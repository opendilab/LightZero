
import json
import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import jericho



def _score_value(score: Optional[float]) -> float:
    return score if score is not None else float('-inf')



def steps_to_payload(steps: List['EpisodeStep']) -> List[Dict[str, Any]]:
    return [{
        'observation': step.observation,
        'action': step.action,
        'score': step.current_score
    } for step in steps]

def load_action_cache(path: Optional[str]) -> Dict[str, List[str]]:
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
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"[CACHE] Saved action cache with {len(cache)} entries to {path}.")
    except Exception as exc:
        print(f"[CACHE] Failed to save action cache to {path}: {exc}")


def append_high_score_episode(path: Optional[str], episode: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(episode, ensure_ascii=False) + '\n')
    except Exception as exc:
        print(f"[CACHE] Failed to append high score episode to {path}: {exc}")


@dataclass(frozen=True)
class EpisodeStep:
    """A single step within an episode trajectory."""

    state_hash: str
    observation: str
    action: str
    current_score: Optional[float]


@dataclass
class Sample:
    """Represents a single dataset sample with history-aware prompting."""

    game: str
    observation: str
    action: str
    history: List[Tuple[str, str]] = field(default_factory=list)
    current_score: Optional[float] = None
    episode_score: Optional[float] = None
    episode_id: Optional[str] = None

    def to_prompt(
        self,
        reasoning_instructions: List[str],
        context_turns: Optional[int] = None
    ) -> str:
        """Compose a prompt with recent interaction history (observation + action)."""
        if reasoning_instructions:
            tip = random.choice(reasoning_instructions)
        else:
            tip = (
                "Think carefully about what to do next. Identify objects, exits "
                "and possible actions before deciding."
            )

        prompt_lines: List[str] = []
        prompt_lines.append(f"You are playing the text adventure game '{self.game}'.")
        prompt_lines.append(
            "Use the observation and recent interaction history to choose the "
            "best possible text command."
        )
        prompt_lines.append("Heuristic to follow: " + tip + ".")
        prompt_lines.append(
            "Respond in the format <think>reasoning</think><answer>action</answer>."
        )

        if self.history:
            prompt_lines.append("")
            prompt_lines.append("Recent interaction history (older to newer):")
            turns = self.history if context_turns is None or context_turns <= 0 else self.history[-context_turns:]
            for idx, (obs, act) in enumerate(turns, start=1):
                prompt_lines.append(f"Turn {idx} observation: {obs.strip()}")
                prompt_lines.append(f"Turn {idx} action: {act.strip()}")
        else:
            prompt_lines.append("")
            prompt_lines.append("No prior turns. This is the start of the episode.")

        prompt_lines.append("")
        prompt_lines.append("Observation: " + self.observation.strip())
        prompt_lines.append("Provide your response now:")
        return "\n".join(prompt_lines)


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
    high_score_threshold: Optional[float],
    high_score_path: Optional[str],
    mode_label: str,
    game_name: str
) -> List[Tuple[List[EpisodeStep], Optional[float]]]:
    """Breadth-first traversal to gather terminal trajectories."""
    env = jericho.FrotzEnv(rom_path)
    try:
        obs, info = env.reset()
        current_score = info.get('score') if info else None
        initial_state = env.get_state()
        initial_hash = env.get_world_state_hash()

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

        seen_hashes = {initial_hash}
        best_terminal_scores: Dict[str, Optional[float]] = {}
        depth_best_scores: Dict[int, float] = {0: _score_value(current_score)}
        depth_node_counts: Dict[int, int] = {0: 1}
        last_depth_logged = -1
        episodes: List[Tuple[List[EpisodeStep], Optional[float]]] = []
        expansions = 0
        high_marker = high_score_threshold if high_score_threshold is not None else float('-inf')

        print(f"[BFS] Start from depth 0 with initial score {_format_score(current_score)}.")

        while queue:
            node = queue.popleft()
            env.set_state(node.state)
            state_hash = node.state_hash
            current_val = _score_value(node.current_score)

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

            if node.done:
                if _better_score(best_terminal_scores.get(state_hash), node.current_score):
                    best_terminal_scores[state_hash] = node.current_score
                    episodes.append((node.steps, node.current_score))
                    print(
                        f"[BFS] Terminal revisit accepted at depth {node.depth} with score "
                        f"{_format_score(node.current_score)} (episodes={len(episodes)})."
                    )
                    if high_score_path and _score_value(node.current_score) >= high_marker:
                        append_high_score_episode(high_score_path, {
                            'game': game_name,
                            'mode': mode_label,
                            'score': node.current_score,
                            'length': len(node.steps),
                            'steps': steps_to_payload(node.steps),
                            'terminal_observation': node.observation
                        })
                    if max_episodes and len(episodes) >= max_episodes:
                        break
                continue

            if node.depth >= max_depth:
                continue
            if max_nodes and expansions >= max_nodes:
                break

            actions = action_cache.get(state_hash)
            if actions is None:
                actions = list(env.get_valid_actions())
                action_cache[state_hash] = actions
                cache_dirty[0] = True
            if not actions:
                continue

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
                    current_score=node.current_score
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
                        if high_score_path and _score_value(score_after) >= high_marker:
                            append_high_score_episode(high_score_path, {
                                'game': game_name,
                                'mode': mode_label,
                                'score': score_after,
                                'length': len(new_steps),
                                'steps': steps_to_payload(new_steps),
                                'terminal_observation': next_obs
                            })
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



def _dfs_collect_episodes(
    rom_path: str,
    score_threshold: Optional[float],
    max_depth: int,
    max_nodes: int,
    max_episodes: int,
    action_cache: Dict[str, List[str]],
    cache_dirty: List[bool],
    max_actions_per_state: int,
    high_score_threshold: Optional[float],
    high_score_path: Optional[str],
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
        high_marker = high_score_threshold if high_score_threshold is not None else float('-inf')
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
                    if high_score_path and _score_value(node.current_score) >= high_marker:
                        append_high_score_episode(high_score_path, {
                            'game': game_name,
                            'mode': mode_label,
                            'score': node.current_score,
                            'length': len(node.steps),
                            'steps': steps_to_payload(node.steps),
                            'terminal_observation': node.observation
                        })
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
                    current_score=node.current_score
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
                        if high_score_path and _score_value(score_after) >= high_marker:
                            append_high_score_episode(high_score_path, {
                                'game': game_name,
                                'mode': mode_label,
                                'score': score_after,
                                'length': len(new_steps),
                                'steps': steps_to_payload(new_steps),
                                'terminal_observation': next_obs
                            })
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
    slice_size: int
) -> List[Tuple[str, float, Sample]]:
    """Slice an episode into samples every ``slice_size`` steps."""
    if slice_size <= 0:
        slice_size = 1

    samples: List[Tuple[str, float, Sample]] = []
    compare_score = episode_score if episode_score is not None else float('-inf')
    history_turns = max(slice_size - 1, 0)

    for idx, step in enumerate(steps, start=1):
        if idx % slice_size != 0:
            continue

        if history_turns > 0:
            start = max(0, idx - history_turns - 1)
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
            episode_id=f"{episode_id}-step{idx}"
        )
        samples.append((step.state_hash, compare_score, sample))

    return samples



def build_dataset_for_game(
    rom_path: str,
    game_name: str,
    samples_per_game: int,
    score_threshold: Optional[float] = 300,
    high_score_threshold: Optional[float] = 300,
    high_score_output: Optional[str] = './rft_datasets/high_score_episodes.jsonl',
    bfs_max_depth: int = 40,
    bfs_max_nodes: int = 5000,
    bfs_max_episodes: int = 100,
    sample_slice_size: int = 3,
    search_mode: str = 'bfs',
    action_cache: Optional[Dict[str, List[str]]] = None,
    cache_dirty: Optional[List[bool]] = None,
    max_actions_per_state: int = 0
) -> List[Dict[str, Any]]:
    """Collect samples for a single game using the configured search strategy."""
    if action_cache is None:
        action_cache = {}
    if cache_dirty is None:
        cache_dirty = [False]

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
        f"high-threshold={high_score_threshold} depth={bfs_max_depth} nodes={bfs_max_nodes} slice={sample_slice_size}."
    )

    samples_by_state: Dict[Tuple[str, str], Dict[str, Any]] = {}

    episodes = collector(
        rom_path=rom_path,
        score_threshold=score_threshold,
        max_depth=bfs_max_depth,
        max_nodes=bfs_max_nodes,
        max_episodes=bfs_max_episodes,
        action_cache=action_cache,
        cache_dirty=cache_dirty,
        max_actions_per_state=max_actions_per_state,
        high_score_threshold=high_score_threshold,
        high_score_path=high_score_output,
        mode_label=mode_label,
        game_name=game_name
    )

    print(f"[DATA] {game_name} produced {len(episodes)} terminal trajectories ({mode_label}).")

    for idx, (steps, episode_score) in enumerate(episodes):
        episode_id = f"{game_name}-{mode_label.lower()}-{idx}"
        print(
            f"[DATA] Episode {episode_id}: length={len(steps)} score={_format_score(episode_score)}."
        )
        for state_hash, compare_score, sample in _generate_samples_from_episode(
            game_name=game_name,
            episode_id=episode_id,
            steps=steps,
            episode_score=episode_score,
            slice_size=sample_slice_size
        ):
            key = (game_name, state_hash)
            existing = samples_by_state.get(key)
            if existing is None or compare_score > existing['compare_score']:
                samples_by_state[key] = {
                    'compare_score': compare_score,
                    'sample': sample
                }

    samples = [entry['sample'] for entry in samples_by_state.values()]
    samples.sort(
        key=lambda sample: (
            sample.episode_score if sample.episode_score is not None else float('-inf'),
            sample.current_score if sample.current_score is not None else float('-inf')
        ),
        reverse=True
    )
    if samples_per_game and samples_per_game > 0:
        samples = samples[:samples_per_game]

    reasoning_tips = [
        "Identify the main objects mentioned and think about how you can interact with them.",
        "Consider your exits. If no obvious object interactions present themselves, explore a new direction.",
        "Maintain awareness of your inventory. Use items when they seem useful, otherwise continue exploring.",
        "If the scene describes dialogue or characters, think about talking to them or responding appropriately.",
        "Picture the layout of the map; retrace your steps or try alternative routes if progress stalls."
    ]

    dataset: List[Dict[str, Any]] = []
    context_turns = max(sample_slice_size - 1, 0)
    for sample in samples:
        prompt = sample.to_prompt(reasoning_tips, context_turns=context_turns)
        metadata = {
            'game': sample.game,
            'episode_id': sample.episode_id,
            'episode_score': sample.episode_score,
            'current_score': sample.current_score
        }
        dataset.append({
            'prompt': prompt,
            'answer': sample.action,
            'metadata': metadata
        })
    return dataset


def save_dataset(dataset: Iterable[Dict[str, Any]], output_path: str) -> None:
    """Write the dataset to a JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build a Jericho RLHF dataset.")
    parser.add_argument('--samples-per-game', type=int, default=1000000,
                        help="Maximum number of samples to include per game.")
    parser.add_argument('--score-threshold', type=float, default=300,
                        help="Minimum score for a trajectory to be kept.")
    parser.add_argument('--bfs-max-depth', type=int, default=500,
                        help="Maximum depth explored during search.")
    parser.add_argument('--bfs-max-nodes', type=int, default=1000000,
                        help="Maximum number of node expansions during search.")
    parser.add_argument('--bfs-max-episodes', type=int, default=10000000,
                        help="Maximum number of terminal episodes to record.")
    parser.add_argument('--slice-size', type=int, default=3,
                        help="Number of steps per dataset sample when slicing trajectories.")
    parser.add_argument('--high-score-threshold', type=float, default=300,
                        help="Score threshold for logging high-quality trajectories.")
    parser.add_argument('--high-score-output', type=str, default='./rft_datasets/high_score_episodes.jsonl',
                        help="Path to append high-scoring trajectories as JSONL.")
    parser.add_argument('--search-mode', type=str, choices=['bfs', 'dfs'], default='dfs',
                        help="Search strategy to use (breadth-first or depth-first).")
    parser.add_argument('--max-actions-per-state', type=int, default=55,
                        help="Limit the number of actions expanded per state (0 for no limit).")
    parser.add_argument('--action-cache', type=str, default='./rft_datasets/action_cache.json',
                        help="Path to persist and reuse the valid action cache.")
    parser.add_argument('--output', type=str,
                        default='./rft_datasets/jericho_dataset.jsonl',
                        help="Output JSONL file.")
    args = parser.parse_args()

    action_cache = load_action_cache(args.action_cache)
    cache_dirty = [False]

    datasets: List[Dict[str, Any]] = []

    games_list = ['zork1']
    for game in games_list:
        rom_path = (
            '/mnt/afs/wanzunian/niuyazhe/xiongjyu/jericho/LightZero/zoo/jericho/'
            f'envs/z-machine-games-master/jericho-game-suite/{game}.z5'
        )
        print(f"[MAIN] Collecting data for {game} from {rom_path}.")
        game_data = build_dataset_for_game(
            rom_path=rom_path,
            game_name=game,
            samples_per_game=args.samples_per_game,
            score_threshold=args.score_threshold,
            high_score_threshold=args.high_score_threshold,
            high_score_output=args.high_score_output,
            bfs_max_depth=args.bfs_max_depth,
            bfs_max_nodes=args.bfs_max_nodes,
            bfs_max_episodes=args.bfs_max_episodes,
            sample_slice_size=args.slice_size,
            search_mode=args.search_mode,
            action_cache=action_cache,
            cache_dirty=cache_dirty,
            max_actions_per_state=args.max_actions_per_state
        )
        datasets.extend(game_data)

    datasets.sort(
        key=lambda entry: (
            entry.get('metadata', {}).get('episode_score')
            if entry.get('metadata', {}).get('episode_score') is not None else float('-inf'),
            entry.get('metadata', {}).get('current_score')
            if entry.get('metadata', {}).get('current_score') is not None else float('-inf')
        ),
        reverse=True
    )
    save_dataset(datasets, args.output)
    print(f"[MAIN] Saved {len(datasets)} samples to {args.output}.")

    if cache_dirty[0]:
        save_action_cache(args.action_cache, action_cache)



if __name__ == '__main__':
    main()
