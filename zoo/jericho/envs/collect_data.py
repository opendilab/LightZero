
import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

import jericho
from jericho.util import unabbreviate

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
        - extension_episode_limit: 最大扩展episode数量
        - extension_max_nodes: 最大节点数限制
        - max_episode_steps: 限制单个episode的最大步数
        - history_turns: 输入给LLM时包含的历史交互轮数
    """

    enabled: bool = False
    expansion_mode: str = 'dfs'
    reverse_backtrack: bool = True
    skip_original_action: bool = True
    extension_score_threshold: Optional[float] = None
    extension_episode_limit: int = 10
    extension_max_nodes: int = 5000
    max_episode_steps: Optional[int] = None
    history_turns: int = 3

    def __post_init__(self) -> None:
        self.expansion_mode = (self.expansion_mode or 'dfs').lower()
        if self.expansion_mode not in {'dfs', 'bfs'}:
            raise ValueError(f"Unsupported expansion_mode '{self.expansion_mode}'. Use 'dfs' or 'bfs'.")
        if self.extension_episode_limit < 0:
            raise ValueError("extension_episode_limit cannot be negative.")
        if self.extension_max_nodes < 0:
            raise ValueError("extension_max_nodes cannot be negative.")
        if self.history_turns < 0:
            raise ValueError("history_turns cannot be negative.")
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive when provided.")


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
        print(f"[CACHE] Failed to load action cache from {path}: {exc}")
        return {}


def save_action_cache(path: Optional[str], cache: Dict[str, List[str]]) -> None:
    """保存动作缓存，用于后续复用"""
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"[CACHE] Saved action cache with {len(cache)} entries to {path}.")
    except Exception as exc:
        print(f"[CACHE] Failed to save action cache to {path}: {exc}")

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
        prompt_lines.append(
            "You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. "
            "And your final answer will be extracted automatically by the \\boxed{} tag."
        )
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
                    'current_score': self.current_score
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
) -> List[Tuple[List[EpisodeStep], Optional[float]]]:
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
        print(f"[BFS] Start from depth 0 with initial score {_format_score(current_score)}.")

        while queue:
            node = queue.popleft()
            env.set_state(node.state)
            state_hash = node.state_hash
            current_val = _score_value(node.current_score)
            # 更新该深度层的最佳分数
            best_val = depth_best_scores.get(node.depth, float('-inf'))
            if current_val > best_val:
                depth_best_scores[node.depth] = current_val
                print(
                    f"[BFS] Depth {node.depth} best score updated to {_format_score(node.current_score)} "
                    f"(episodes={len(episodes)}, queue={len(queue)})."
                )
                best_val = current_val

            if node.depth > last_depth_logged:
                best_display = _format_score(None if best_val == float('-inf') else best_val)
                print(
                    f"[BFS] Entering depth {node.depth}: current queue size {len(queue)}, "
                    f"collected episodes {len(episodes)}, best score {best_display}."
                )
                last_depth_logged = node.depth
            # 如果该节点是终止状态，尝试记录 episode
            if node.done:
                if _better_score(best_terminal_scores.get(state_hash), node.current_score):
                    best_terminal_scores[state_hash] = node.current_score
                    episodes.append((node.steps, node.current_score))
                    print(
                        f"[BFS] Terminal revisit accepted at depth {node.depth} with score "
                        f"{_format_score(node.current_score)} (episodes={len(episodes)})."
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
                        print(
                            f"[BFS] Terminal episode depth {next_depth} score {_format_score(score_after)} "
                            f"(length={len(new_steps)}, total={len(episodes)})."
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

        print(
            f"[BFS] Finished exploration: depths explored {len(depth_best_scores)}, "
            f"episodes collected {len(episodes)}, expansions {expansions}."
        )
        for depth in sorted(depth_best_scores):
            best = depth_best_scores[depth]
            count = depth_node_counts.get(depth, 0)
            best_display = _format_score(None if best == float('-inf') else best)
            print(
                f"[BFS] Depth {depth}: best score {best_display}, nodes processed {count}."
            )
        return episodes
    finally:
        env.close()


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
    already_collected: int,
    start_depth: int,
    max_total_steps: Optional[int],
    blocked_hashes: Optional[Set[str]]
) -> List[Tuple[List[EpisodeStep], Optional[float]]]:
    """
    从指定起始状态（通常为 walkthrough 中的一个节点）继续进行探索，
    生成新的扩展 episode（用于数据增强）。
    支持 BFS 或 DFS 两种扩展方式。
    """
    if params.extension_episode_limit and already_collected >= params.extension_episode_limit:
        return []

    max_nodes = params.extension_max_nodes
    remaining_limit = (
        params.extension_episode_limit - already_collected
        if params.extension_episode_limit
        else None
    )
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

    while frontier:
        if params.expansion_mode == 'bfs':
            node = frontier.popleft()  # type: ignore[attr-defined]
        else:
            node = frontier.pop()

        if max_nodes and expansions >= max_nodes:
            print(f"[WALK] Reached expansion cap ({expansions}).")
            break

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
                env.set_state(saved_state)
                continue

            prev_depth = visited_depth.get(new_hash)
            if prev_depth is not None and prev_depth <= next_depth:
                env.set_state(saved_state)
                continue
            visited_depth[new_hash] = next_depth

            if blocked_hashes and new_hash in blocked_hashes:
                env.set_state(saved_state)
                continue

            if done:
                final_val = _score_value(score_after)
                if final_val >= extension_threshold_value and final_val >= threshold_value:
                    collected.append((new_steps, score_after))
                    print(
                        f"[WALK] Collected extension episode depth {next_depth} score {_format_score(score_after)} "
                        f"(len={len(new_steps)})."
                    )
                env.set_state(saved_state)
                if remaining_limit and len(collected) + already_collected >= params.extension_episode_limit:
                    return collected
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

        if remaining_limit and len(collected) + already_collected >= params.extension_episode_limit:
            break

    return collected


def _collect_walkthrough_episodes(
    rom_path: str,
    game_name: str,
    params: WalkthroughHyperParams,
    score_threshold: Optional[float],
    action_cache: Dict[str, List[str]],
    cache_dirty: List[bool],
    include_walkthrough_episode: bool,
    include_extension: bool
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
            print(f"[WALK] No walkthrough available for {game_name}.")
            return []

        obs, info = env.reset()
        current_score = info.get('score') if info else None
        state = env.get_state()
        state_hash = env.get_world_state_hash()

        trajectory_states: List[Tuple[Any, str, Optional[float], str]] = [
            (state, obs, current_score, state_hash)
        ]
        walkway_steps: List[EpisodeStep] = []

        print(f"[WALK] Executing walkthrough for {game_name} with {len(walkthrough_actions)} actions.")

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
        if not walkway_steps:
            print(f"[WALK] Walkthrough generated no steps for {game_name}.")
            return []

        episodes: List[Tuple[List[EpisodeStep], Optional[float]]] = []
        final_val = _score_value(final_score)
        threshold_val = _score_value(score_threshold)

        if include_walkthrough_episode and final_val >= threshold_val:
            episodes.append((list(walkway_steps), final_score))
            print(
                f"[WALK] Collected walkthrough episode length {len(walkway_steps)} "
                f"score {_format_score(final_score)}."
            )
        elif include_walkthrough_episode:
            print(
                f"[WALK] Walkthrough score {_format_score(final_score)} below threshold "
                f"{_format_score(score_threshold)}; skipping base episode."
            )
        else:
            print("[WALK] Walkthrough episode excluded by configuration.")

        if (
            not include_extension
            or not params.reverse_backtrack
            or params.extension_episode_limit == 0
        ):
            return episodes

        walkthrough_hashes: Set[str] = {entry[3] for entry in trajectory_states}

        collected_extension = 0
        for pivot in range(len(walkway_steps) - 1, -1, -1):
            if params.extension_episode_limit and collected_extension >= params.extension_episode_limit:
                break
            state_snapshot, pivot_obs, pivot_score, pivot_hash = trajectory_states[pivot]
            # Adjust pivot depth so that prior steps are reflected correctly when extending.
            pivot_depth = pivot
            prefix = walkway_steps[:pivot]
            skip_action = walkway_steps[pivot].action if params.skip_original_action else None

            if params.max_episode_steps is not None and len(prefix) >= params.max_episode_steps:
                continue

            blocked_hashes = set(walkthrough_hashes)
            blocked_hashes.discard(pivot_hash)

            env.set_state(state_snapshot)
            extension = _expand_from_state(
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
                already_collected=collected_extension,
                start_depth=pivot_depth,
                max_total_steps=params.max_episode_steps,
                blocked_hashes=blocked_hashes
            )
            episodes.extend(extension)
            collected_extension += len(extension)
            if collected_extension and params.extension_episode_limit and collected_extension >= params.extension_episode_limit:
                break

        if collected_extension:
            print(f"[WALK] Generated {collected_extension} extension episodes for {game_name}.")

        return episodes
    finally:
        env.close()



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

        print(f"[DFS] Start from depth 0 with initial score {_format_score(current_score)}.")

        def report_leaf(depth: int, score: Optional[float]) -> None:
            nonlocal leaf_count
            leaf_count += 1
            if leaf_count % leaf_interval == 0:
                print(
                    f"[DFS] Processed {leaf_count} leaves; last depth {depth}; score {_format_score(score)}."
                )

        while stack:
            node = stack.pop()
            env.set_state(node.state)
            state_hash = node.state_hash
            current_val = _score_value(node.current_score)

            best_val = depth_best_scores.get(node.depth, float('-inf'))
            if current_val > best_val:
                depth_best_scores[node.depth] = current_val
                print(
                    f"[DFS] Best score improved: {_format_score(node.current_score)} at depth {node.depth} (episodes={len(episodes)})."
                )

            if node.done:
                if _better_score(best_terminal_scores.get(state_hash), node.current_score):
                    best_terminal_scores[state_hash] = node.current_score
                    episodes.append((node.steps, node.current_score))
                    print(
                        f"[DFS] Terminal accepted at depth {node.depth} with score "
                        f"{_format_score(node.current_score)} (episodes={len(episodes)})."
                    )
                    if max_episodes and len(episodes) >= max_episodes:
                        break
                report_leaf(node.depth, node.current_score)
                continue

            if node.depth >= max_depth:
                report_leaf(node.depth, node.current_score)
                continue
            if max_nodes and expansions >= max_nodes:
                print(f"[DFS] Reached max node limit at {expansions} expansions.")
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
                        print(
                            f"[DFS] Terminal episode depth {next_depth} score {_format_score(score_after)} "
                            f"(length={len(new_steps)}, total={len(episodes)})."
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

        print(
            f"[DFS] Finished exploration: depths explored {len(depth_best_scores)}, "
            f"episodes collected {len(episodes)}, expansions {expansions}, leaves {leaf_count}."
        )
        for depth in sorted(depth_best_scores):
            best = depth_best_scores[depth]
            count = depth_node_counts.get(depth, 0)
            best_display = _format_score(None if best == float('-inf') else best)
            print(
                f"[DFS] Depth {depth}: best score {best_display}, nodes processed {count}."
            )
        return episodes
    finally:
        env.close()

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

        sample = Sample(
            game=game_name,
            observation=step.observation,
            action=step.action,
            history=history,
            current_score=step.current_score,
            episode_score=episode_score,
            episode_id=f"{episode_id}-step{idx}",
            episode_length=len(steps),
            valid_actions=list(step.valid_actions)
        )
        samples.append((step.state_hash, compare_score, sample))

    return samples



def build_dataset_for_game(
    rom_path: str,
    game_name: str,
    samples_per_game: int,
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
    collection_switches: Optional[CollectionSwitches] = None
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

    print(
        f"[DATA] Collecting game='{game_name}' mode={mode_label} threshold={score_threshold} "
        f"depth={bfs_max_depth} nodes={bfs_max_nodes} history_window={history_window}."
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
        print(f"[DATA] Skipping search-based {mode_label} collection for {game_name} per configuration.")

    if walkthrough_params.enabled and (
        collection_switches.use_walkthrough_episode or collection_switches.use_walkthrough_extensions
    ):
        walk_eps = _collect_walkthrough_episodes(
            rom_path=rom_path,
            game_name=game_name,
            params=walkthrough_params,
            score_threshold=score_threshold,
            action_cache=action_cache,
            cache_dirty=cache_dirty,
            include_walkthrough_episode=collection_switches.use_walkthrough_episode,
            include_extension=collection_switches.use_walkthrough_extensions
        )
        episodes.extend(walk_eps)
        if walk_eps:
            print(f"[DATA] Added {len(walk_eps)} walkthrough-derived trajectories for {game_name}.")
    elif walkthrough_params.enabled:
        print(f"[DATA] Walkthrough processing enabled but all walkthrough-related outputs disabled; skipping for {game_name}.")

    print(
        f"[DATA] {game_name} produced {len(episodes)} terminal trajectories "
        f"(search_enabled={collection_switches.use_search_episodes}, mode={mode_label})."
    )

    for idx, (steps, episode_score) in enumerate(episodes):
        if score_threshold is not None and _score_value(episode_score) < _score_value(score_threshold):
            continue
        episode_id = f"{game_name}-{mode_label.lower()}-{idx}"
        print(
            f"[DATA] Episode {episode_id}: length={len(steps)} score={_format_score(episode_score)}."
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
    if samples_per_game and samples_per_game > 0:
        samples = samples[:samples_per_game]

    dataset: List[List[Dict[str, Any]]] = []
    for sample in samples:
        total_steps = sample.episode_length if sample.episode_length is not None else 0
        entry = sample.to_dialogue_entry(
            params=walkthrough_params,
            total_steps=total_steps,
            total_return=sample.episode_score
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
                    print(f"[DATA] Existing dataset at {output_path} is not a list; overwriting.")
        except json.JSONDecodeError:
            print(f"[DATA] Failed to parse existing dataset at {output_path}; starting fresh.")

    existing_entries.extend(new_entries)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(existing_entries, f, ensure_ascii=False, indent=2)



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build a Jericho RLHF dataset.")
    parser.add_argument('--samples-per-game', type=int, default=1000,
                        help="Maximum number of samples to include per game.")
    parser.add_argument('--bfs-max-depth', type=int, default=500,
                        help="Maximum depth explored during search.")
    parser.add_argument('--bfs-max-nodes', type=int, default=1000,
                        help="Maximum number of node expansions during search.")
    parser.add_argument('--bfs-max-episodes', type=int, default=1000,
                        help="Maximum number of terminal episodes to record.")
    parser.add_argument('--history-windows', type=List, default=[4, 10],
                        help="JSON list of history lengths (e.g., [4, 10]).")
    parser.add_argument('--search-mode', type=str, choices=['bfs', 'dfs'], default='dfs',
                        help="Search strategy to use (breadth-first or depth-first).")
    parser.add_argument('--max-actions-per-state', type=int, default=55,
                        help="Limit the number of actions expanded per state (0 for no limit).")
    parser.add_argument('--action-cache', type=str, default='/mnt/afs/wanzunian/niuyazhe/xiongjyu/jericho/LightZero/zoo/jericho/envs/rft_datasets/action_cache.json',
                        help="Path to persist and reuse the valid action cache.")
    parser.add_argument('--output', type=str,
                        default='/mnt/afs/wanzunian/niuyazhe/xiongjyu/jericho/LightZero/zoo/jericho/envs/rft_datasets/jericho_dataset',
                        help="Output file prefix; window suffixes will be appended.")
    args = parser.parse_args()

    action_cache = load_action_cache(args.action_cache)
    cache_dirty = [False]

    history_windows = args.history_windows
    if not history_windows:
        history_windows = [4]

    walkthrough_params = WalkthroughHyperParams(
        enabled=True,
        expansion_mode='dfs',
        reverse_backtrack=True,
        skip_original_action=True,
        extension_score_threshold=None,
        extension_episode_limit=200,
        extension_max_nodes=1500,
        history_turns=history_windows[0],
    )

    collection_switches = CollectionSwitches(
        use_walkthrough_episode=True,
        use_search_episodes=False,
        use_walkthrough_extensions=True,
    )

    per_game_thresholds: Dict[str, float] = {
        'zork1': 330.0,
        'detective': 320.0,
        'acorncourt': 25.0,
        'omniquest': 40.0,
    }
    per_game_max_steps: Dict[str, Optional[int]] = {
        'zork1': 500,
        'detective': 100,
        'acorncourt': 50,
        'omniquest': 100,
    }

    output_dir = os.path.dirname(os.path.abspath(args.output))
    base_name = os.path.splitext(os.path.basename(args.output))[0]

    for history_window in history_windows:
        datasets: List[List[Dict[str, Any]]] = []
        walkthrough_params.history_turns = history_window

        for game, game_threshold in per_game_thresholds.items():
            walkthrough_params.extension_score_threshold = game_threshold
            walkthrough_params.max_episode_steps = per_game_max_steps.get(game)
            rom_path = (
                '/mnt/afs/wanzunian/niuyazhe/xiongjyu/jericho/LightZero/zoo/jericho/'
                f'envs/z-machine-games-master/jericho-game-suite/{game}.z5'
            )
            print(
                f"[MAIN] Collecting data for {game} (threshold={game_threshold}, history={history_window}) "
                f"from {rom_path}."
            )
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
                action_cache=action_cache,
                cache_dirty=cache_dirty,
                max_actions_per_state=args.max_actions_per_state,
                walkthrough_params=walkthrough_params,
                collection_switches=collection_switches
            )
            datasets.extend(game_data)

        output_path = os.path.join(output_dir, f"{base_name}_his_{history_window}.json")
        save_dataset(datasets, output_path)
        print(f"[MAIN] Saved {len(datasets)} samples to {output_path}.")

    # if cache_dirty[0]:
    #     save_action_cache(args.action_cache, action_cache)



if __name__ == '__main__':
    main()
