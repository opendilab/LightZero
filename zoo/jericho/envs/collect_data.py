
import json
import logging
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

import jericho
from jericho.util import unabbreviate

logger = logging.getLogger("collect_data")

# ================================================================
# 基础配置类定义
# ================================================================
@dataclass
class WalkthroughHyperParams:
    """
        控制 Walkthrough 相关采集与扩展的超参数
        - enabled: 是否启用 walkthrough 模式
        - expansion_mode: 搜索方式（'dfs' 或 'bfs'）
        - reverse_backtrack: 是否从 walkthrough 的末尾回溯生成扩展
        - skip_original_action: 是否跳过原始动作
        - extension_score_threshold: 用于筛选扩展episode的分数阈值
        - max_episode_steps: 限制单个episode的最大步数
        - history_turns: 输入给LLM时包含的历史交互轮数
        - tail_pivot_steps: walkthrough 末尾用于扩展的步数（0 表示全部剩余步数）
        - max_success_expansions: 成功（胜利且分数达标）的扩展 episode 最大数量
        - max_total_expansions: 总扩展 episode 上限（成功与否均计算），用于防止长时间探索
        - progress_path: walkthrough 扩展进度保存文件
    """

    enabled: bool = False
    expansion_mode: str = 'dfs'
    reverse_backtrack: bool = True
    skip_original_action: bool = True
    extension_score_threshold: Optional[float] = None
    max_episode_steps: Optional[int] = None
    history_turns: int = 3
    tail_pivot_steps: int = 1
    max_success_expansions: Optional[int] = None
    max_total_expansions: Optional[int] = None
    progress_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.expansion_mode = (self.expansion_mode or 'dfs').lower()
        if self.expansion_mode not in {'dfs', 'bfs'}:
            raise ValueError(f"Unsupported expansion_mode '{self.expansion_mode}'. Use 'dfs' or 'bfs'.")
        if self.history_turns < 0:
            raise ValueError("history_turns cannot be negative.")
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive when provided.")
        if self.tail_pivot_steps is not None and self.tail_pivot_steps < 0:
            raise ValueError("tail_pivot_steps cannot be negative.")
        if self.max_success_expansions is not None and self.max_success_expansions < 0:
            raise ValueError("max_success_expansions cannot be negative.")
        if self.max_total_expansions is not None and self.max_total_expansions < 0:
            raise ValueError("max_total_expansions cannot be negative.")


@dataclass
class CollectionSwitches:
    """
    控制采集来源开关：
    - use_walkthrough_episode: 是否使用 walkthrough 正常轨迹
    - use_search_episodes: 是否使用 BFS/DFS 搜索得到的episode
    - use_walkthrough_extensions: 是否使用 walkthrough 回溯扩展episode
    """

    use_walkthrough_episode: bool = True
    use_search_episodes: bool = True
    use_walkthrough_extensions: bool = True

# ================================================================
# 工具函数
# ================================================================
def _score_value(score: Optional[float]) -> float:
    """将 None 转换为 -inf，便于比较分数"""
    return score if score is not None else float('-inf')

def load_action_cache(path: Optional[str]) -> Dict[str, List[str]]:
    """从JSON文件加载缓存的有效动作字典 {state_hash: [actions]}"""
    if not path:
        return {}
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # ensure keys/values are strings/lists
        cache = {str(k): list(v) for k, v in data.items()}
        return cache
    except Exception as exc:
        logger.warning("[CACHE] Failed to load action cache from %s: %s", path, exc)
        return {}


def save_action_cache(path: Optional[str], cache: Dict[str, List[str]]) -> None:
    """保存动作缓存，用于后续复用"""
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
        logger.info("[CACHE] Saved action cache with %d entries to %s.", len(cache), path)
    except Exception as exc:
        logger.warning("[CACHE] Failed to save action cache to %s: %s", path, exc)


def load_progress(path: Optional[str]) -> Dict[str, int]:
    """加载 walkthrough 扩展进度 {game: processed_tail_steps}"""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): int(v) for k, v in data.items()}
    except Exception as exc:
        logger.warning("[WALK] Failed to load progress from %s: %s", path, exc)
    return {}


def save_progress(path: Optional[str], progress: Dict[str, int]) -> None:
    """保存 walkthrough 扩展进度"""
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        logger.info("[WALK] Saved progress for %d games to %s.", len(progress), path)
    except Exception as exc:
        logger.warning("[WALK] Failed to save progress to %s: %s", path, exc)

# ================================================================
# 数据结构定义
# ================================================================

@dataclass(frozen=True)
class EpisodeStep:
    """
    表示游戏中一个交互步骤：
    - state_hash: 当前状态哈希
    - observation: 当前观测
    - action: 执行动作
    - current_score: 当前得分
    - valid_actions: 可执行动作列表
    """
    state_hash: str
    observation: str
    action: str
    current_score: Optional[float]  # reward obtained by executing ``action``
    valid_actions: List[str] = field(default_factory=list)


@dataclass
class Sample:
    """
    单条采样数据样本，最终转换为 LLM 训练对话格式
    - history: 包含前若干步交互历史
    """

    game: str
    observation: str
    action: str
    history: List[Tuple[str, str]] = field(default_factory=list)
    current_score: Optional[float] = None
    episode_score: Optional[float] = None
    episode_id: Optional[str] = None
    episode_length: Optional[int] = None
    valid_actions: List[str] = field(default_factory=list)
    cumulative_return: Optional[float] = None

    def to_dialogue_entry(
        self,
        params: WalkthroughHyperParams,
        total_steps: int,
        total_return: Optional[float]
    ) -> List[Dict[str, Any]]:
        """
        将样本转为RLHF格式的对话：
        - human：提示信息（包含历史、当前观察、候选动作等）
        - assistant：ground_truth 动作
        - metadata：额外episode信息
        """
        
        history_turns = params.history_turns
        if history_turns > 0:
            effective_history = self.history[-history_turns:]
        else:
            effective_history = []

        prompt_lines: List[str] = []
        prompt_lines.append(
            "You are a player in a text-based adventure game. Your task is to evaluate and select "
            "actions that are promising based on the given game state."
        )
        if history_turns and history_turns > 0:
            if effective_history:
                prompt_lines.append(
                    f"Recent {history_turns} interactions (oldest to newest):"
                )
                for idx, (obs, act) in enumerate(effective_history, start=1):
                    prompt_lines.append(f"History {idx} observation: {obs.strip()}")
                    prompt_lines.append(f"History {idx} action: {act.strip()}")
            else:
                prompt_lines.append(
                    "No previous interactions are available; this is the beginning of the episode."
                )
        else:
            prompt_lines.append(
                "No interaction history will be provided; decide solely from the current observation."
            )

        prompt_lines.append("Current observation: " + self.observation.strip())
        if self.valid_actions:
            actions_display = ", ".join(act.strip() for act in self.valid_actions)
            prompt_lines.append(
                "Candidate actions (you may choose actions outside this list, but these are often useful): "
                + f"[{actions_display}]"
            )
        prompt_lines.append("The single best next action to maximize your score is:")
        prompt = "\n".join(prompt_lines)

        dialogue_entry = [
            {
                'from': 'human',
                'value': prompt
            },
            {
                'from': 'assistant',
                'ground_truth': {
                    'value': self.action
                }
            },
            {
                'from': 'metadata',
                'episode_info': {
                    'game': self.game,
                'episode_id': self.episode_id,
                'total_steps': total_steps,
                'total_return': total_return,
                'history_step': len(effective_history),
                'current_score': self.current_score,
                'episode_final_return': self.episode_score
            }
        }
        ]
        return dialogue_entry

# ================================================================
# 搜索节点定义（BFS/DFS）
# ================================================================

@dataclass
class SearchNode:
    """Represents a node in the BFS frontier."""

    state: Any
    observation: str
    steps: List[EpisodeStep]
    current_score: Optional[float]
    depth: int
    state_hash: str
    done: bool

# ================================================================
# 核心搜索模块（BFS / DFS）
# ================================================================

# _bfs_collect_episodes 与 _dfs_collect_episodes 的作用是：
# 从游戏初始状态出发，通过 BFS 或 DFS 搜索探索所有可能的路径，
# 收集到达终止状态的完整episode（包括动作、观测、得分）

# BFS：广度优先，优先探索浅层节点；
# DFS：深度优先，优先探索长路径；


# ================================================================
# 构建样本与保存数据集
# ================================================================


def _better_score(existing: Optional[float], candidate: Optional[float]) -> bool:
    existing_val = existing if existing is not None else float('-inf')
    candidate_val = candidate if candidate is not None else float('-inf')
    return candidate_val > existing_val


def _format_score(score: Optional[float]) -> str:
    if score is None:
        return "None"
    return f"{score:.2f}"




def _bfs_collect_episodes(
    rom_path: str,
    score_threshold: Optional[float],
    max_depth: int,
    max_nodes: int,
    max_episodes: int,
    action_cache: Dict[str, List[str]],
    cache_dirty: List[bool],
    max_actions_per_state: int,
    mode_label: str,
    game_name: str
) -> Tuple[List[Tuple[List[EpisodeStep], Optional[float]]], int, int]:
    """Breadth-first traversal to gather terminal trajectories."""
    # 初始化 Jericho 环境（基于 ROM 文件）
    env = jericho.FrotzEnv(rom_path)
    try:
        # 重置环境，获取初始 observation 和 info（含初始分数）
        obs, info = env.reset()
        current_score = info.get('score') if info else None
        # 记录初始状态和哈希，用于后续去重
        initial_state = env.get_state()
        initial_hash = env.get_world_state_hash()

        # 初始化 BFS 队列，起始节点包含初始状态信息
        queue: Deque[SearchNode] = deque([
            SearchNode(
                state=initial_state,
                observation=obs,
                steps=[],
                current_score=current_score,
                depth=0,
                state_hash=initial_hash,
                done=False
            )
        ])
        # 各类辅助表
        seen_hashes = {initial_hash}
        best_terminal_scores: Dict[str, Optional[float]] = {}
        depth_best_scores: Dict[int, float] = {0: _score_value(current_score)}
        depth_node_counts: Dict[int, int] = {0: 1}
        last_depth_logged = -1
        episodes: List[Tuple[List[EpisodeStep], Optional[float]]] = []
        expansions = 0
        logger.debug("[BFS] Start from depth 0 with initial score %s.", _format_score(current_score))

        while queue:
            node = queue.popleft()
            env.set_state(node.state)
            state_hash = node.state_hash
            current_val = _score_value(node.current_score)
            # 更新该深度层的最佳分数
            best_val = depth_best_scores.get(node.depth, float('-inf'))
            if current_val > best_val:
                depth_best_scores[node.depth] = current_val
                logger.debug(
                    "[BFS] Depth %d best score updated to %s (episodes=%d, queue=%d).",
                    node.depth,
                    _format_score(node.current_score),
                    len(episodes),
                    len(queue)
                )
                best_val = current_val

            if node.depth > last_depth_logged:
                best_display = _format_score(None if best_val == float('-inf') else best_val)
                logger.debug(
                    "[BFS] Entering depth %d: current queue size %d, collected episodes %d, best score %s.",
                    node.depth,
                    len(queue),
                    len(episodes),
                    best_display
                )
                last_depth_logged = node.depth
            # 如果该节点是终止状态，尝试记录 episode
            if node.done:
                if _better_score(best_terminal_scores.get(state_hash), node.current_score):
                    best_terminal_scores[state_hash] = node.current_score
                    episodes.append((node.steps, node.current_score))
                    logger.debug(
                        "[BFS] Terminal revisit accepted at depth %d with score %s (episodes=%d).",
                        node.depth,
                        _format_score(node.current_score),
                        len(episodes)
                    )
                    if max_episodes and len(episodes) >= max_episodes:
                        break
                continue
            # 深度或节点数限制
            if node.depth >= max_depth:
                continue
            if max_nodes and expansions >= max_nodes:
                break
            # 从缓存中取有效动作列表，否则从环境获取
            actions = action_cache.get(state_hash)
            if actions is None:
                actions = list(env.get_valid_actions())
                action_cache[state_hash] = actions
                cache_dirty[0] = True
            if not actions:
                continue

            available_actions = list(actions)

            if max_actions_per_state > 0 and len(actions) > max_actions_per_state:
                actions_to_use = actions[:max_actions_per_state]
            else:
                actions_to_use = actions

            expansions += 1
            depth_node_counts[node.depth] = depth_node_counts.get(node.depth, 0) + 1

            for action in actions_to_use:
                env.set_state(node.state)
                saved_state = env.get_state()

                episode_step = EpisodeStep(
                    state_hash=state_hash,
                    observation=node.observation,
                    action=action,
                    current_score=node.current_score,
                    valid_actions=available_actions
                )

                next_obs, reward, done, info = env.step(action)
                score_after = info.get('score') if info else None

                new_steps = node.steps + [episode_step]
                new_state = env.get_state()
                new_hash = env.get_world_state_hash()
                next_depth = node.depth + 1

                if done:
                    if score_after is not None and _better_score(best_terminal_scores.get(new_hash), score_after):
                        best_terminal_scores[new_hash] = score_after
                        episodes.append((new_steps, score_after))
                        logger.debug(
                            "[BFS] Terminal episode depth %d score %s (length=%d, total=%d).",
                            next_depth,
                            _format_score(score_after),
                            len(new_steps),
                            len(episodes)
                        )
                        if max_episodes and len(episodes) >= max_episodes:
                            env.set_state(saved_state)
                            return episodes
                else:
                    if new_hash in seen_hashes:
                        env.set_state(saved_state)
                        continue
                    seen_hashes.add(new_hash)
                    child_val = _score_value(score_after)
                    prev_best = depth_best_scores.get(next_depth, float('-inf'))
                    if child_val > prev_best:
                        depth_best_scores[next_depth] = child_val
                    queue.append(SearchNode(
                        state=new_state,
                        observation=next_obs,
                        steps=new_steps,
                        current_score=score_after,
                        depth=next_depth,
                        state_hash=new_hash,
                        done=False
                    ))

                env.set_state(saved_state)

        logger.debug(
            "[BFS] Finished exploration: depths explored %d, episodes collected %d, expansions %d.",
            len(depth_best_scores),
            len(episodes),
            expansions
        )
        for depth in sorted(depth_best_scores):
            best = depth_best_scores[depth]
            count = depth_node_counts.get(depth, 0)
            best_display = _format_score(None if best == float('-inf') else best)
            logger.debug(
                "[BFS] Depth %d: best score %s, nodes processed %d.",
                depth,
                best_display,
                count
            )
        return episodes
    finally:
        env.close()
        del env


def _walkthrough_extension_threshold(params: WalkthroughHyperParams, fallback: Optional[float]) -> Optional[float]:
    if params.extension_score_threshold is not None:
        return params.extension_score_threshold
    return fallback


def _expand_from_state(
    env: jericho.FrotzEnv,
    params: WalkthroughHyperParams,
    base_state: Any,
    base_observation: str,
    base_score: Optional[float],
    base_hash: str,
    prefix_steps: List[EpisodeStep],
    action_cache: Dict[str, List[str]],
    cache_dirty: List[bool],
    score_threshold: Optional[float],
    game_name: str,
    expansion_label: str,
    skip_action: Optional[str],
    start_depth: int,
    max_total_steps: Optional[int],
    blocked_hashes: Optional[Set[str]],
    walkthrough_target_score: Optional[float],
    walkthrough_target_won: bool,
    success_cap: Optional[int] = None,
    total_cap: Optional[int] = None
) -> Tuple[List[Tuple[List[EpisodeStep], Optional[float]]], int, int]:
    """
    从指定起始状态（通常为 walkthrough 中的一个节点）继续进行探索，
    生成新的扩展 episode（用于数据增强）。
    支持 BFS 或 DFS 两种扩展方式。
    """
    threshold_value = _score_value(score_threshold)
    extension_threshold = _walkthrough_extension_threshold(params, score_threshold)
    extension_threshold_value = _score_value(extension_threshold)

    NodeQueue: Any
    if params.expansion_mode == 'bfs':
        NodeQueue = deque  # type: ignore[assignment]
    else:
        NodeQueue = list  # type: ignore[assignment]

    frontier = NodeQueue()  # type: ignore[call-arg]
    root_depth = start_depth
    visited_depth: Dict[str, int] = {base_hash: root_depth}
    best_states: Dict[str, Tuple[float, int]] = {
        base_hash: (_score_value(base_score), len(prefix_steps))
    }

    root = SearchNode(
        state=base_state,
        observation=base_observation,
        steps=list(prefix_steps),
        current_score=base_score,
        depth=start_depth,
        state_hash=base_hash,
        done=False
    )
    if isinstance(frontier, list):
        frontier.append(root)
    else:
        frontier.append(root)  # type: ignore[attr-defined]

    collected: List[Tuple[List[EpisodeStep], Optional[float]]] = []
    expansions = 0
    success_expansions = 0
    total_expansions = 0

    success_limit = success_cap if success_cap is not None else params.max_success_expansions
    total_limit = total_cap if total_cap is not None else params.max_total_expansions

    if success_limit is not None and success_limit <= 0:
        return collected, success_expansions, total_expansions
    if total_limit is not None and total_limit <= 0:
        return collected, success_expansions, total_expansions

    while frontier:
        if total_limit is not None and total_expansions >= total_limit:
            break
        if success_limit is not None and success_expansions >= success_limit:
            break
        if params.expansion_mode == 'bfs':
            node = frontier.popleft()  # type: ignore[attr-defined]
        else:
            node = frontier.pop()

        env.set_state(node.state)
        state_hash = node.state_hash
        current_score = node.current_score

        actions = action_cache.get(state_hash)
        if actions is None:
            actions = list(env.get_valid_actions())
            action_cache[state_hash] = actions
            cache_dirty[0] = True
        if not actions:
            continue

        available_actions = list(actions)

        if skip_action and node.depth == root_depth:
            candidate_actions = [act for act in actions if act != skip_action]
        else:
            candidate_actions = actions

        if not candidate_actions:
            continue

        expansions += 1

        for action in candidate_actions:
            env.set_state(node.state)
            saved_state = env.get_state()
            next_obs, reward, done, info = env.step(action)
            score_after = info.get('score') if info else None
            if reward is not None:
                step_reward = float(reward)
            elif current_score is not None and score_after is not None:
                step_reward = float(score_after - current_score)
            else:
                step_reward = None

            episode_step = EpisodeStep(
                state_hash=state_hash,
                observation=node.observation,
                action=action,
                current_score=step_reward,
                valid_actions=available_actions
            )

            new_steps = node.steps + [episode_step]
            new_state = env.get_state()
            new_hash = env.get_world_state_hash()
            next_depth = node.depth + 1

            if max_total_steps is not None and len(new_steps) > max_total_steps:
                done = True

            prev_depth = visited_depth.get(new_hash)
            if prev_depth is not None and prev_depth <= next_depth:
                env.set_state(saved_state)
                continue
            visited_depth[new_hash] = next_depth

            if blocked_hashes and new_hash in blocked_hashes:
                env.set_state(saved_state)
                continue

            candidate_score = _score_value(score_after)
            candidate_length = len(new_steps)
            best_entry = best_states.get(new_hash)
            if best_entry is not None:
                best_score, best_length = best_entry
                if candidate_score < best_score or (
                    candidate_score == best_score and candidate_length >= best_length
                ):
                    env.set_state(saved_state)
                    continue
            best_states[new_hash] = (candidate_score, candidate_length)

            if done:
                final_val = _score_value(score_after)
                won_flag = False
                try:
                    won_flag = bool(env.victory())
                except Exception:
                    won_flag = bool(info.get('won')) if isinstance(info, dict) else False
                target_match = (
                    walkthrough_target_score is None
                    or (score_after is not None and score_after == walkthrough_target_score)
                )
                success = (
                    final_val >= extension_threshold_value
                    and final_val >= threshold_value
                    and won_flag
                    and walkthrough_target_won
                    and target_match
                )
                total_expansions += 1
                if total_expansions % 1000 == 0:
                    success_limit_str = success_limit if success_limit is not None else '∞'
                    total_limit_str = total_limit if total_limit is not None else '∞'
                    logger.info(
                        "[WALK] %s progress: attempts=%s/%s, successes=%s/%s.",
                        expansion_label,
                        total_expansions,
                        total_limit_str,
                        success_expansions,
                        success_limit_str
                    )
                if success:
                    collected.append((new_steps, score_after))
                    success_expansions += 1
                    if success_limit is not None and success_expansions >= success_limit:
                        env.set_state(saved_state)
                        return collected, success_expansions, total_expansions
                if total_limit is not None and total_expansions >= total_limit:
                    env.set_state(saved_state)
                    return collected, success_expansions, total_expansions
                env.set_state(saved_state)
                continue

            child = SearchNode(
                state=new_state,
                observation=next_obs,
                steps=new_steps,
                current_score=score_after,
                depth=next_depth,
                state_hash=new_hash,
                done=False
            )
            if params.expansion_mode == 'bfs':
                frontier.append(child)  # type: ignore[attr-defined]
            else:
                frontier.append(child)
            env.set_state(saved_state)

    return collected, success_expansions, total_expansions


def _collect_walkthrough_episodes(
    rom_path: str,
    game_name: str,
    params: WalkthroughHyperParams,
    score_threshold: Optional[float],
    action_cache: Dict[str, List[str]],
    cache_dirty: List[bool],
    include_walkthrough_episode: bool,
    include_extension: bool,
    tail_steps: Optional[int],
    progress_data: Optional[Dict[str, int]],
    progress_dirty: Optional[List[bool]]
) -> List[Tuple[List[EpisodeStep], Optional[float]]]:
    """
    使用 Jericho 自带的 walkthrough（官方通关步骤）
    采集高质量的参考 episode，并可从中回溯扩展新轨迹。
    """
    
    if not params.enabled:
        return []

    env = jericho.FrotzEnv(rom_path)
    try:
        walkthrough_actions = env.get_walkthrough()
        if not walkthrough_actions:
            logger.info("[WALK] No walkthrough available for %s.", game_name)
            return []

        obs, info = env.reset()
        current_score = info.get('score') if info else None
        state = env.get_state()
        state_hash = env.get_world_state_hash()

        trajectory_states: List[Tuple[Any, str, Optional[float], str]] = [
            (state, obs, current_score, state_hash)
        ]
        walkway_steps: List[EpisodeStep] = []

        logger.info("[WALK] Executing walkthrough for %s with %d actions.", game_name, len(walkthrough_actions))

        for action in walkthrough_actions:
            expanded_action = unabbreviate(action) if action else action
            state_snapshot, state_obs, state_score, state_hash = trajectory_states[-1]
            env.set_state(state_snapshot)

            available_actions = list(env.get_valid_actions())
            next_obs, reward, done, info = env.step(expanded_action)
            # print(f'state_hash={state_hash}expanded_action={expanded_action}\treward={reward}\t info={info}')
            next_score = info.get('score') if info else None
            step_reward: Optional[float]
            if reward is not None:
                step_reward = float(reward)
            elif state_score is not None and next_score is not None:
                step_reward = float(next_score - state_score)
            else:
                step_reward = None

            episode_step = EpisodeStep(
                state_hash=state_hash,
                observation=state_obs,
                action=expanded_action,
                current_score=step_reward,
                valid_actions=available_actions
            )
            walkway_steps.append(episode_step)
            next_state = env.get_state()
            next_hash = env.get_world_state_hash()
            trajectory_states.append((next_state, next_obs, next_score, next_hash))
            if done:
                break

        final_state, final_obs, final_score, final_hash = trajectory_states[-1]
        try:
            walkthrough_won = bool(env.victory())
        except Exception:
            walkthrough_won = False
        if not walkway_steps:
            logger.info("[WALK] Walkthrough generated no steps for %s.", game_name)
            return []

        episodes: List[Tuple[List[EpisodeStep], Optional[float]]] = []
        final_val = _score_value(final_score)
        threshold_val = _score_value(score_threshold)

        if include_walkthrough_episode and final_val >= threshold_val and walkthrough_won:
            episodes.append((list(walkway_steps), final_score))
            logger.info(
                "[WALK] Collected walkthrough episode length %d score %s (won=%s).",
                len(walkway_steps),
                _format_score(final_score),
                walkthrough_won
            )
        elif include_walkthrough_episode:
            logger.info(
                "[WALK] Walkthrough score %s below threshold %s or victory=%s; skipping base episode.",
                _format_score(final_score),
                _format_score(score_threshold),
                walkthrough_won
            )
        else:
            logger.info("[WALK] Walkthrough episode excluded by configuration.")

        if (
            not include_extension
            or not params.reverse_backtrack
            or not walkthrough_won
        ):
            return episodes

        walkthrough_hashes: Set[str] = {entry[3] for entry in trajectory_states}

        processed_count = 0
        if progress_data is not None:
            processed_count = int(progress_data.get(game_name, 0))

        total_pivots = len(walkway_steps)
        if processed_count >= total_pivots:
            logger.info("[WALK] All walkthrough pivots already processed for %s (total %d).", game_name, total_pivots)
            return episodes

        available_pivots = list(range(total_pivots - 1, -1, -1))[processed_count:]
        tail_limit = tail_steps if tail_steps and tail_steps > 0 else len(available_pivots)
        pivot_sequence = available_pivots[:tail_limit]

        if not pivot_sequence:
            logger.info("[WALK] No remaining pivots to process for %s.", game_name)
            return episodes

        steps_processed = 0
        total_new_branches = 0
        total_success_expansions = 0
        total_attempt_expansions = 0

        success_limit = params.max_success_expansions
        total_limit = params.max_total_expansions

        for pivot in pivot_sequence:
            state_snapshot, pivot_obs, pivot_score, pivot_hash = trajectory_states[pivot]
            pivot_depth = pivot
            prefix = walkway_steps[:pivot]
            skip_action = walkway_steps[pivot].action if params.skip_original_action else None

            if params.max_episode_steps is not None and len(prefix) >= params.max_episode_steps:
                continue

            blocked_hashes = set(walkthrough_hashes)
            blocked_hashes.discard(pivot_hash)

            env.set_state(state_snapshot)
            extension, pivot_successes, pivot_attempts = _expand_from_state(
                env=env,
                params=params,
                base_state=state_snapshot,
                base_observation=pivot_obs,
                base_score=pivot_score,
                base_hash=pivot_hash,
                prefix_steps=prefix,
                action_cache=action_cache,
                cache_dirty=cache_dirty,
                score_threshold=score_threshold,
                game_name=game_name,
                expansion_label='WalkthroughExt',
                skip_action=skip_action,
                start_depth=pivot_depth,
                max_total_steps=params.max_episode_steps,
                blocked_hashes=blocked_hashes,
                walkthrough_target_score=final_score,
                walkthrough_target_won=walkthrough_won,
                success_cap=success_limit,
                total_cap=total_limit
            )
            episodes.extend(extension)
            branch_count = len(extension)
            total_new_branches += branch_count
            total_success_expansions += pivot_successes
            total_attempt_expansions += pivot_attempts
            steps_processed += 1
            tail_index = processed_count + steps_processed
            logger.info(
                "[WALK] Game=%s pivot#%d (step index %d) produced %d trajectories.",
                game_name,
                tail_index,
                pivot,
                branch_count
            )

        if progress_data is not None and steps_processed:
            if progress_lock is not None:
                with progress_lock:
                    progress_data[game_name] = processed_count + steps_processed
                    if progress_dirty is not None:
                        progress_dirty[0] = True
            else:
                progress_data[game_name] = processed_count + steps_processed
                if progress_dirty is not None:
                    progress_dirty[0] = True

        if steps_processed:
            agg_success_cap = (
                steps_processed * success_limit if success_limit is not None else '∞'
            )
            agg_total_cap = (
                steps_processed * total_limit if total_limit is not None else '∞'
            )
            success_summary = f"[{total_success_expansions}/{agg_success_cap}]"
            total_summary = f"[{total_attempt_expansions}/{agg_total_cap}]"
            logger.info(
                "[WALK] Walkthrough extension for %s: processed %d pivots (total processed %d/%d), "
                "generated %d trajectories; success=%s, attempts=%s.",
                game_name,
                steps_processed,
                processed_count + steps_processed,
                total_pivots,
                total_new_branches,
                success_summary,
                total_summary
            )

        return episodes
    finally:
        env.close()
        del env



def _dfs_collect_episodes(
    rom_path: str,
    score_threshold: Optional[float],
    max_depth: int,
    max_nodes: int,
    max_episodes: int,
    action_cache: Dict[str, List[str]],
    cache_dirty: List[bool],
    max_actions_per_state: int,
    mode_label: str,
    game_name: str
) -> List[Tuple[List[EpisodeStep], Optional[float]]]:
    """Depth-first traversal to gather terminal trajectories."""
    env = jericho.FrotzEnv(rom_path)
    try:
        obs, info = env.reset()
        current_score = info.get('score') if info else None
        initial_state = env.get_state()
        initial_hash = env.get_world_state_hash()

        stack: List[SearchNode] = [
            SearchNode(
                state=initial_state,
                observation=obs,
                steps=[],
                current_score=current_score,
                depth=0,
                state_hash=initial_hash,
                done=False
            )
        ]

        best_progress: Dict[str, Tuple[float, int]] = {initial_hash: (_score_value(current_score), 0)}
        best_terminal_scores: Dict[str, Optional[float]] = {}
        depth_best_scores: Dict[int, float] = {0: _score_value(current_score)}
        depth_node_counts: Dict[int, int] = {0: 1}
        episodes: List[Tuple[List[EpisodeStep], Optional[float]]] = []
        expansions = 0
        leaf_count = 0
        leaf_interval = 100

        logger.debug("[DFS] Start from depth 0 with initial score %s.", _format_score(current_score))

        def report_leaf(depth: int, score: Optional[float]) -> None:
            nonlocal leaf_count
            leaf_count += 1
            if leaf_count % leaf_interval == 0:
                logger.debug(
                    "[DFS] Processed %d leaves; last depth %d; score %s.",
                    leaf_count,
                    depth,
                    _format_score(score)
                )

        while stack:
            node = stack.pop()
            env.set_state(node.state)
            state_hash = node.state_hash
            current_val = _score_value(node.current_score)

            best_val = depth_best_scores.get(node.depth, float('-inf'))
            if current_val > best_val:
                depth_best_scores[node.depth] = current_val
                logger.debug(
                    "[DFS] Best score improved: %s at depth %d (episodes=%d).",
                    _format_score(node.current_score),
                    node.depth,
                    len(episodes)
                )

            if node.done:
                if _better_score(best_terminal_scores.get(state_hash), node.current_score):
                    best_terminal_scores[state_hash] = node.current_score
                    episodes.append((node.steps, node.current_score))
                    logger.debug(
                        "[DFS] Terminal accepted at depth %d with score %s (episodes=%d).",
                        node.depth,
                        _format_score(node.current_score),
                        len(episodes)
                    )
                    if max_episodes and len(episodes) >= max_episodes:
                        break
                report_leaf(node.depth, node.current_score)
                continue

            if node.depth >= max_depth:
                report_leaf(node.depth, node.current_score)
                continue
            if max_nodes and expansions >= max_nodes:
                logger.debug("[DFS] Reached max node limit at %d expansions.", expansions)
                break

            actions = action_cache.get(state_hash)
            if actions is None:
                actions = list(env.get_valid_actions())
                action_cache[state_hash] = actions
                cache_dirty[0] = True
            if not actions:
                report_leaf(node.depth, node.current_score)
                continue

            available_actions = list(actions)

            if max_actions_per_state > 0 and len(actions) > max_actions_per_state:
                actions_to_use = actions[:max_actions_per_state]
            else:
                actions_to_use = actions

            expansions += 1
            depth_node_counts[node.depth] = depth_node_counts.get(node.depth, 0) + 1

            for action in reversed(actions_to_use):
                env.set_state(node.state)
                saved_state = env.get_state()

                episode_step = EpisodeStep(
                    state_hash=state_hash,
                    observation=node.observation,
                    action=action,
                    current_score=node.current_score,
                    valid_actions=available_actions
                )

                next_obs, reward, done, info = env.step(action)
                score_after = info.get('score') if info else None

                new_steps = node.steps + [episode_step]
                new_state = env.get_state()
                new_hash = env.get_world_state_hash()
                next_depth = node.depth + 1

                candidate_val = _score_value(score_after)
                progress_entry = best_progress.get(new_hash)
                if progress_entry is not None:
                    stored_score, stored_depth = progress_entry
                    if candidate_val < stored_score or (candidate_val == stored_score and next_depth >= stored_depth):
                        env.set_state(saved_state)
                        continue

                best_progress[new_hash] = (candidate_val, next_depth)

                if done:
                    if score_after is not None and _better_score(best_terminal_scores.get(new_hash), score_after):
                        best_terminal_scores[new_hash] = score_after
                        episodes.append((new_steps, score_after))
                        logger.debug(
                            "[DFS] Terminal episode depth %d score %s (length=%d, total=%d).",
                            next_depth,
                            _format_score(score_after),
                            len(new_steps),
                            len(episodes)
                        )
                        report_leaf(next_depth, score_after)
                        if max_episodes and len(episodes) >= max_episodes:
                            env.set_state(saved_state)
                            return episodes
                else:
                    prev_best = depth_best_scores.get(next_depth, float('-inf'))
                    if candidate_val > prev_best:
                        depth_best_scores[next_depth] = candidate_val
                    stack.append(SearchNode(
                        state=new_state,
                        observation=next_obs,
                        steps=new_steps,
                        current_score=score_after,
                        depth=next_depth,
                        state_hash=new_hash,
                        done=False
                    ))

                env.set_state(saved_state)

        logger.debug(
            "[DFS] Finished exploration: depths explored %d, episodes collected %d, expansions %d, leaves %d.",
            len(depth_best_scores),
            len(episodes),
            expansions,
            leaf_count
        )
        for depth in sorted(depth_best_scores):
            best = depth_best_scores[depth]
            count = depth_node_counts.get(depth, 0)
            best_display = _format_score(None if best == float('-inf') else best)
            logger.debug(
                "[DFS] Depth %d: best score %s, nodes processed %d.",
                depth,
                best_display,
                count
            )
        return episodes
    finally:
        env.close()
        del env

def _generate_samples_from_episode(
    game_name: str,
    episode_id: str,
    steps: List[EpisodeStep],
    episode_score: Optional[float],
    history_window: Optional[int]
) -> List[Tuple[str, float, Sample]]:
    """Generate one sample per step, attaching up to ``history_window`` prior interactions."""
    samples: List[Tuple[str, float, Sample]] = []
    compare_score = episode_score if episode_score is not None else float('-inf')

    running_return = 0.0

    for idx, step in enumerate(steps, start=1):
        if idx > 1:
            if history_window is None or history_window <= 0:
                start = 0
            else:
                start = max(0, idx - 1 - history_window)
            prior_steps = steps[start:idx - 1]
            history = [(pst.observation, pst.action) for pst in prior_steps]
        else:
            history = []

        if step.current_score is not None:
            running_return += float(step.current_score)

        sample = Sample(
            game=game_name,
            observation=step.observation,
            action=step.action,
            history=history,
            current_score=step.current_score,
            episode_score=episode_score,
            episode_id=f"{episode_id}-step{idx}",
            episode_length=len(steps),
            valid_actions=list(step.valid_actions),
            cumulative_return=running_return
        )
        samples.append((step.state_hash, compare_score, sample))

    return samples



def build_dataset_for_game(
    rom_path: str,
    game_name: str,
    samples_per_game: Optional[int],
    score_threshold: Optional[float] = 300,
    bfs_max_depth: int = 40,
    bfs_max_nodes: int = 5000,
    bfs_max_episodes: int = 100,
    history_window: Optional[int] = 3,
    search_mode: str = 'bfs',
    action_cache: Optional[Dict[str, List[str]]] = None,
    cache_dirty: Optional[List[bool]] = None,
    max_actions_per_state: int = 0,
    walkthrough_params: Optional[WalkthroughHyperParams] = None,
    collection_switches: Optional[CollectionSwitches] = None,
    walkthrough_progress: Optional[Dict[str, int]] = None,
    progress_dirty: Optional[List[bool]] = None,
    progress_lock: Optional[threading.Lock] = None
) -> List[List[Dict[str, Any]]]:
    """Collect samples for a single game using the configured search strategy."""
    """
    为单个游戏（如 zork1）构建完整数据集。
    步骤：
      1. 执行 BFS/DFS 搜索得到探索 episode。
      2. 可选：执行 walkthrough + 扩展采集高质量 episode。
      3. 将所有 episode 转换为样本（带历史窗口）。
      4. 格式化为 LLM 对话训练样式并返回。
    """
    if action_cache is None:
        action_cache = {}
    if cache_dirty is None:
        cache_dirty = [False]

    if walkthrough_params is None:
        walkthrough_params = WalkthroughHyperParams()
    if collection_switches is None:
        collection_switches = CollectionSwitches()

    if max_actions_per_state < 0:
        max_actions_per_state = 0

    mode = (search_mode or 'bfs').lower()
    if mode == 'dfs':
        collector = _dfs_collect_episodes
        mode_label = 'DFS'
    else:
        collector = _bfs_collect_episodes
        mode_label = 'BFS'

    logger.info(
        "[DATA] Collecting game='%s' mode=%s threshold=%s depth=%d nodes=%d history_window=%s.",
        game_name,
        mode_label,
        score_threshold,
        bfs_max_depth,
        bfs_max_nodes,
        history_window
    )

    samples_by_state: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    episodes: List[Tuple[List[EpisodeStep], Optional[float]]] = []
    if collection_switches.use_search_episodes:
        episodes = collector(
            rom_path=rom_path,
            score_threshold=score_threshold,
            max_depth=bfs_max_depth,
            max_nodes=bfs_max_nodes,
            max_episodes=bfs_max_episodes,
            action_cache=action_cache,
            cache_dirty=cache_dirty,
            max_actions_per_state=max_actions_per_state,
            mode_label=mode_label,
            game_name=game_name
        )
    else:
        logger.info(
            "[DATA] Skipping search-based %s collection for %s per configuration.",
            mode_label,
            game_name
        )

    if walkthrough_params.enabled and (
        collection_switches.use_walkthrough_episode or collection_switches.use_walkthrough_extensions
    ):
        walk_eps, _, _ = _collect_walkthrough_episodes(
            rom_path=rom_path,
            game_name=game_name,
            params=walkthrough_params,
            score_threshold=score_threshold,
            action_cache=action_cache,
            cache_dirty=cache_dirty,
            include_walkthrough_episode=collection_switches.use_walkthrough_episode,
            include_extension=collection_switches.use_walkthrough_extensions,
            tail_steps=walkthrough_params.tail_pivot_steps,
            progress_data=walkthrough_progress,
            progress_dirty=progress_dirty
        )
        episodes.extend(walk_eps)
        if walk_eps:
            logger.info("[DATA] Added %d walkthrough-derived trajectories for %s.", len(walk_eps), game_name)
    elif walkthrough_params.enabled:
        logger.info(
            "[DATA] Walkthrough processing enabled but outputs disabled; skipping for %s.",
            game_name
        )

    logger.info(
        "[DATA] %s produced %d terminal trajectories (search_enabled=%s, mode=%s).",
        game_name,
        len(episodes),
        collection_switches.use_search_episodes,
        mode_label
    )

    for idx, (steps, episode_score) in enumerate(episodes):
        if score_threshold is not None and _score_value(episode_score) < _score_value(score_threshold):
            continue
        episode_id = f"{game_name}-{mode_label.lower()}-{idx}"
        logger.debug(
            "[DATA] Episode %s: length=%d score=%s.",
            episode_id,
            len(steps),
            _format_score(episode_score)
        )
        for state_hash, compare_score, sample in _generate_samples_from_episode(
            game_name=game_name,
            episode_id=episode_id,
            steps=steps,
            episode_score=episode_score,
            history_window=history_window
        ):
            key = (game_name, state_hash, sample.action)
            existing = samples_by_state.get(key)
            if existing is None or compare_score > existing['compare_score']:
                samples_by_state[key] = {
                    'compare_score': compare_score,
                    'sample': sample
                }

    samples = [entry['sample'] for entry in samples_by_state.values()]
    if samples_per_game is not None and samples_per_game > 0:
        samples = samples[:samples_per_game]

    dataset: List[List[Dict[str, Any]]] = []
    for sample in samples:
        total_steps = sample.episode_length if sample.episode_length is not None else 0
        entry = sample.to_dialogue_entry(
            params=walkthrough_params,
            total_steps=total_steps,
            total_return=sample.cumulative_return
        )
        dataset.append(entry)
    return dataset


def save_dataset(dataset: Iterable[Iterable[Dict[str, Any]]], output_path: str) -> None:
    """Persist dataset entries to a JSON file (list of dialog records)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    new_entries = list(dataset)
    existing_entries: List[Any] = []

    if os.path.isfile(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    existing_entries = loaded
                else:
                    logger.warning("[DATA] Existing dataset at %s is not a list; overwriting.", output_path)
        except json.JSONDecodeError:
            logger.warning("[DATA] Failed to parse existing dataset at %s; starting fresh.", output_path)

    existing_entries.extend(new_entries)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(existing_entries, f, ensure_ascii=False, indent=2)



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build a Jericho RLHF dataset.")
    parser.add_argument('--samples-per-game', type=int, default=None,
                        help="Optional cap for samples per game (omit for all).")
    parser.add_argument('--bfs-max-depth', type=int, default=500,
                        help="Maximum depth explored during search.")
    parser.add_argument('--bfs-max-nodes', type=int, default=1000,
                        help="Maximum number of node expansions during search.")
    parser.add_argument('--bfs-max-episodes', type=int, default=1000,
                        help="Maximum number of terminal episodes to record.")
    parser.add_argument('--search-mode', type=str, choices=['bfs', 'dfs'], default='dfs',
                        help="Search strategy to use (breadth-first or depth-first).")
    parser.add_argument('--max-actions-per-state', type=int, default=55,
                        help="Limit the number of actions expanded per state (0 for no limit).")
    parser.add_argument('--action-cache', type=str, default='/mnt/afs/wanzunian/niuyazhe/xiongjyu/jericho/LightZero/zoo/jericho/envs/rft_datasets/action_cache.json',
                        help="Path to persist and reuse the valid action cache.")
    parser.add_argument('--output', type=str,
                        default='/mnt/afs/wanzunian/niuyazhe/xiongjyu/jericho/LightZero/zoo/jericho/envs/rft_datasets/parallel/jericho_dataset',
                        help="Output file prefix; window suffixes will be appended.")
    parser.add_argument('--parallel-games', type=int, default=11,
                        help="Number of games to process in parallel for each history window.")
    args = parser.parse_args()

    output_prefix = args.output
    output_dir = os.path.dirname(os.path.abspath(output_prefix))
    base_name = os.path.splitext(os.path.basename(output_prefix))[0]
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    general_log_path = os.path.join(logs_dir, f"{base_name}.log")
    general_handler = logging.FileHandler(general_log_path, encoding='utf-8')
    general_handler.setFormatter(formatter)
    logger.addHandler(general_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    parallel_games = max(1, args.parallel_games or 1)
    shared_cache = parallel_games == 1

    if shared_cache:
        action_cache = load_action_cache(args.action_cache)
        cache_dirty = [False]
    else:
        logger.info(
            "[MAIN] Parallel mode (%d games): shared action cache disabled to avoid contention.",
            parallel_games
        )
        action_cache = {}
        cache_dirty = [False]

    history_windows = [4]

    progress_lock = threading.Lock() if parallel_games > 1 else None

    progress_path = os.path.join(output_dir, f"{base_name}_progress.json") if output_dir else None

    walkthrough_params = WalkthroughHyperParams(
        enabled=True,
        expansion_mode='dfs',
        reverse_backtrack=True,
        skip_original_action=True,
        tail_pivot_steps=3,
        max_success_expansions=2000,
        max_total_expansions=500000,
        progress_path=progress_path
    )

    collection_switches = CollectionSwitches(
        use_walkthrough_episode=True,
        use_search_episodes=False,
        use_walkthrough_extensions=True,
    )

    progress_data = load_progress(walkthrough_params.progress_path) if walkthrough_params.progress_path else {}
    progress_dirty = [False]

    per_game_thresholds: Dict[str, float] = {
        'acorncourt': 25.0,
        'zork1': 330.0,
        'detective': 320.0,
        'omniquest': 40.0,
        'pentari': 60,
        'ludicorp': 130,
        'balances': 40,
        'library': 25,
        'deephome': 280,
        'temple': 30,
        'ztuu': 80
    }
    per_game_max_steps: Dict[str, Optional[int]] = {
        'zork1': 500,
        'detective': 100,
        'acorncourt': 50,
        'omniquest': 100,
        'pentari': 100,
        'ludicorp': 400,
        'balances': 300,
        'library': 150, 
        'deephome': 400,
        'temple': 300,
        'ztuu': 100, 
    }

    for history_window in history_windows:
        datasets: List[List[Dict[str, Any]]] = []
        walkthrough_params.history_turns = history_window

        def collect_single_game(game: str, game_threshold: float) -> Tuple[str, List[List[Dict[str, Any]]]]:
            local_cache = action_cache if shared_cache else {}
            local_cache_dirty = cache_dirty if shared_cache else [False]
            local_progress_dirty = progress_dirty
            local_params = replace(walkthrough_params)
            local_params.history_turns = history_window
            local_params.extension_score_threshold = game_threshold
            local_params.max_episode_steps = per_game_max_steps.get(game)
            rom_path = (
                '/mnt/afs/wanzunian/niuyazhe/xiongjyu/jericho/LightZero/zoo/jericho/'
                f'envs/z-machine-games-master/jericho-game-suite/{game}.z5'
            )
            thread_id = threading.get_ident()

            class ThreadFilter(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    return record.thread == thread_id

            game_log_path = os.path.join(logs_dir, f"{base_name}_his_{history_window}_{game}.log")
            game_handler = logging.FileHandler(game_log_path, encoding='utf-8')
            game_handler.setFormatter(formatter)
            game_handler.addFilter(ThreadFilter())
            logger.addHandler(game_handler)

            logger.info(
                "[MAIN] Collecting data for %s (threshold=%s, history=%s) from %s.",
                game,
                game_threshold,
                history_window,
                rom_path
            )

            try:
                game_data = build_dataset_for_game(
                    rom_path=rom_path,
                    game_name=game,
                    samples_per_game=args.samples_per_game,
                    score_threshold=game_threshold,
                    bfs_max_depth=args.bfs_max_depth,
                bfs_max_nodes=args.bfs_max_nodes,
                bfs_max_episodes=args.bfs_max_episodes,
                history_window=history_window,
                search_mode=args.search_mode,
                action_cache=local_cache,
                cache_dirty=local_cache_dirty,
                max_actions_per_state=args.max_actions_per_state,
                    walkthrough_params=local_params,
                    collection_switches=collection_switches,
                    walkthrough_progress=progress_data,
                    progress_dirty=local_progress_dirty,
                    progress_lock=progress_lock
                )
                per_game_output = os.path.join(output_dir, f"{base_name}_his_{history_window}_{game}.json")
                save_dataset(game_data, per_game_output)
                logger.info("[MAIN] Saved %d samples to %s.", len(game_data), per_game_output)
                return game, game_data
            finally:
                logger.removeHandler(game_handler)
                game_handler.close()

        with ThreadPoolExecutor(max_workers=parallel_games) as executor:
            futures = [
                executor.submit(collect_single_game, game, game_threshold)
                for game, game_threshold in per_game_thresholds.items()
            ]
            for future in as_completed(futures):
                _, game_dataset = future.result()
                datasets.extend(game_dataset)

        output_path = os.path.join(output_dir, f"{base_name}_his_{history_window}.json")
    save_dataset(datasets, output_path)
    logger.info("[MAIN] Saved %d samples to %s.", len(datasets), output_path)

    if walkthrough_params.progress_path and progress_dirty[0]:
        save_progress(walkthrough_params.progress_path, progress_data)

    if shared_cache and cache_dirty[0]:
        save_action_cache(args.action_cache, action_cache)



if __name__ == '__main__':
    main()
