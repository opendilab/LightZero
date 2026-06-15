#!/usr/bin/env python3
"""
Evaluate VLM prior quality by running episodes with different policies:
  - random: uniform random action selection
  - vlm:    VLM prior (greedy argmax from VL model output)

Usage (on GPU worker):
  cd zoo/jericho/priorzero
  python scripts/eval_vl_prior.py --vl_model Qwen2.5-VL-7b --num_episodes 20
  python scripts/eval_vl_prior.py --vl_model Qwen2.5-VL-7b --num_episodes 20 --prompt_style legacy
  python scripts/eval_vl_prior.py --vl_model Qwen2.5-VL-7b --num_episodes 20 --vlm_image_mode first_and_current
  python scripts/eval_vl_prior.py --policies random  # random-only baseline (no GPU needed)
"""
import argparse
import sys
import os
import time
import json
import glob
import numpy as np
from collections import defaultdict, deque
from pathlib import Path


# # ---------------------------------------------------------------------------
# # Fix NVIDIA driver visibility in containers (must run before torch import)
# # ---------------------------------------------------------------------------
# def _fix_nvidia_env():
#     """Auto-detect NVIDIA driver libs and force-load libcuda before torch init."""
#     import ctypes

#     # 1. Patch LD_LIBRARY_PATH for child processes / nvidia-smi
#     candidate_lib_dirs = [
#         "/usr/local/nvidia/lib64",
#         "/usr/local/nvidia/lib",
#         "/usr/lib/x86_64-linux-gnu",
#         "/usr/lib64",
#     ]
#     candidate_bin_dirs = [
#         "/usr/local/nvidia/bin",
#         "/usr/local/cuda/bin",
#     ]
#     for pattern in ["/usr/**/libcuda.so.1", "/lib/**/libcuda.so.1"]:
#         for p in glob.glob(pattern, recursive=True):
#             d = os.path.dirname(p)
#             if d not in candidate_lib_dirs:
#                 candidate_lib_dirs.append(d)

#     ld_path = os.environ.get("LD_LIBRARY_PATH", "")
#     for d in candidate_lib_dirs:
#         if os.path.isdir(d) and d not in ld_path:
#             ld_path = d + ":" + ld_path
#     os.environ["LD_LIBRARY_PATH"] = ld_path

#     path = os.environ.get("PATH", "")
#     for d in candidate_bin_dirs:
#         if os.path.isdir(d) and d not in path:
#             path = d + ":" + path
#     os.environ["PATH"] = path

#     # 2. Force-load libcuda.so.1 into the current process so torch can find it.
#     #    Setting LD_LIBRARY_PATH alone is too late — the dynamic linker only
#     #    reads it at process start. ctypes.CDLL loads it immediately.
#     for d in candidate_lib_dirs:
#         libcuda = os.path.join(d, "libcuda.so.1")
#         if os.path.isfile(libcuda):
#             try:
#                 ctypes.CDLL(libcuda)
#             except OSError:
#                 continue
#             break

# _fix_nvidia_env()

# # Now safe to check CUDA
import torch
if not torch.cuda.is_available():
    print("[WARN] torch.cuda.is_available() = False. VLM policy will fail.")
    print(f"  LD_LIBRARY_PATH = {os.environ.get('LD_LIBRARY_PATH', '(unset)')}")
    print(f"  Searching libcuda.so.1 ...")
    found = glob.glob("/usr/**/libcuda.so*", recursive=True) + \
            glob.glob("/lib/**/libcuda.so*", recursive=True)
    print(f"  Found: {found or 'NONE — this node has no GPU driver'}")
    print("  If running on a GPU node, check that the NVIDIA driver is mounted into the container.")
else:
    print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")


# ── ensure project root is importable ──
SCRIPT_DIR = Path(__file__).resolve().parent.parent  # zoo/jericho/priorzero
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "src"))
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent       # LightZero root
sys.path.insert(0, str(PROJECT_ROOT))

# ── PLACEHOLDER_MORE_IMPORTS ──


# ---------------------------------------------------------------------------
# Environment wrapper (thin, no DI-engine dependency)
# ---------------------------------------------------------------------------
class LunarLanderImageWrapper:
    """Minimal wrapper around gymnasium LunarLander with image obs."""

    ACTION_NAMES = ["NOOP", "LEFT_ENGINE", "MAIN_ENGINE", "RIGHT_ENGINE"]

    def __init__(self, image_size: int = 64, seed: int = 0):
        try:
            import gymnasium as gym
        except ImportError:
            import gym
        import cv2
        self._cv2 = cv2
        self._env = gym.make("LunarLander-v2", render_mode="rgb_array")
        self._image_size = image_size
        self._seed = seed
        self._timestep = 0

    def reset(self):
        self._env.reset(seed=self._seed)
        self._timestep = 0
        return self._render()

    def step(self, action_idx: int):
        _, reward, terminated, truncated, info = self._env.step(action_idx)
        self._timestep += 1
        done = terminated or truncated
        obs = self._render()
        return obs, reward, done, info

    def _render(self) -> np.ndarray:
        frame = self._env.render()  # (H, W, 3) uint8
        frame = self._cv2.resize(frame, (self._image_size, self._image_size),
                                 interpolation=self._cv2.INTER_AREA)
        # CHW float32 [0,1] — same as LunarLanderImageEnv
        return np.transpose(frame, (2, 0, 1)).astype(np.float32) / 255.0

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Policy: Random
# ---------------------------------------------------------------------------
class RandomPolicy:
    name = "random"

    def select_action(self, obs, history, valid_actions):
        idx = np.random.randint(len(valid_actions))
        return idx, valid_actions[idx]


# ---------------------------------------------------------------------------
# Policy: VLM Prior (greedy)
# ---------------------------------------------------------------------------
class VLMPolicy:
    """Wraps VLPriorGenerator for greedy action selection."""

    def __init__(self, prior_generator):
        self.pg = prior_generator
        self.name = "vlm"

    def select_action(self, obs, history, valid_actions):
        result = self.pg.generate_prior(
            observation=obs,
            action_candidates=valid_actions,
            history=history,
            temperature=0.01,  # near-greedy
        )
        idx = int(np.argmax(result["action_probs"]))
        return idx, valid_actions[idx]


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(env, policy, history_maxlen: int = 3, max_steps: int = 1000):
    """Run one episode, return (total_reward, steps, action_counts)."""
    obs = env.reset()
    history = deque(maxlen=history_maxlen)
    total_reward = 0.0
    action_counts = defaultdict(int)

    for step in range(max_steps):
        action_idx, action_name = policy.select_action(
            obs, list(history), LunarLanderImageWrapper.ACTION_NAMES
        )
        action_counts[action_name] += 1
        next_obs, reward, done, info = env.step(action_idx)
        history.append((obs, action_name, float(reward), step))
        total_reward += reward
        obs = next_obs
        if done:
            break

    return total_reward, step + 1, dict(action_counts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_vl_policy(args):
    """Build VLPriorGenerator from args. Requires GPU."""
    from vl_config import VL_MODEL_CONFIGS, GAME_DESCRIPTIONS
    from vl_engine import VLLMVLEngine
    from prior_generator import VLPriorGenerator

    model_cfg = VL_MODEL_CONFIGS[args.vl_model]
    limit_mm = {"image": 4 if args.vlm_image_mode != "current_only" else 1}
    print(f"Loading VL model: {args.vl_model} ({model_cfg['model_path']})")

    # Use high-level VLLMVLEngine with standalone=True (no DDP)
    vl_engine = VLLMVLEngine(
        model_name="qwen2.5-vl",
        model_path=model_cfg["model_path"],
        tensor_parallel_size=model_cfg["tensor_parallel_size"],
        gpu_memory_utilization=model_cfg["gpu_memory_utilization"],
        max_model_len=4096,
        enable_sleep=False,
        limit_mm_per_prompt=limit_mm,
        standalone=True,
    )

    pg = VLPriorGenerator(
        vl_engine=vl_engine,
        model_name=model_cfg["model_path"],
        use_cot=args.use_cot,
        game_description=GAME_DESCRIPTIONS.get("LunarLander-v2", ""),
        vlm_image_mode=args.vlm_image_mode,
        prompt_style=args.prompt_style,
    )
    return VLMPolicy(pg)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM prior vs random on LunarLander")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--history_length", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=64)
    # VLM settings
    parser.add_argument("--vl_model", type=str, default="Qwen3-VL-8b")
    # parser.add_argument("--vl_model", type=str, default="Qwen2.5-VL-7b")
    parser.add_argument("--use_cot", action="store_true", default=True)
    parser.add_argument("--no_cot", action="store_true")
    parser.add_argument("--vlm_image_mode", type=str, default="current_only",
                        choices=["current_only", "first_and_current", "all_history"])
    parser.add_argument("--prompt_style", type=str, default="concise",
                        choices=["concise", "legacy"])
    # Which policies to run
    parser.add_argument("--policies", type=str, nargs="+", default=["random", "vlm"],
                        choices=["random", "vlm"])
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results (default: stdout only)")
    args = parser.parse_args()

    if args.no_cot:
        args.use_cot = False

    # Build policies
    policies = []
    for p in args.policies:
        if p == "random":
            policies.append(RandomPolicy())
        elif p == "vlm":
            if not torch.cuda.is_available():
                print("[SKIP] vlm policy requires GPU but CUDA is not available. Skipping.")
                continue
            policies.append(build_vl_policy(args))

    if not policies:
        print("[ERROR] No policies to evaluate. Exiting.")
        sys.exit(1)

    # Run evaluation
    all_results = {}
    for policy in policies:
        tag = f"{policy.name}"
        if hasattr(policy, "pg"):
            tag += f"_{args.prompt_style}_{args.vlm_image_mode}"
            if args.use_cot:
                tag += "_cot"

        print(f"\n{'='*60}")
        print(f"Policy: {tag}  |  Episodes: {args.num_episodes}")
        print(f"{'='*60}")

        rewards = []
        steps_list = []
        action_totals = defaultdict(int)

        for ep in range(args.num_episodes):
            env = LunarLanderImageWrapper(image_size=args.image_size,
                                          seed=args.seed + ep)
            t0 = time.time()
            ep_reward, ep_steps, ep_actions = run_episode(
                env, policy,
                history_maxlen=args.history_length,
                max_steps=args.max_steps,
            )
            elapsed = time.time() - t0
            env.close()

            rewards.append(ep_reward)
            steps_list.append(ep_steps)
            for k, v in ep_actions.items():
                action_totals[k] += v

            print(f"  ep {ep:3d}: reward={ep_reward:8.2f}  steps={ep_steps:4d}  "
                  f"time={elapsed:.1f}s  actions={dict(ep_actions)}")

        # Summary
        r = np.array(rewards)
        summary = {
            "policy": tag,
            "num_episodes": args.num_episodes,
            "reward_mean": float(r.mean()),
            "reward_std": float(r.std()),
            "reward_min": float(r.min()),
            "reward_max": float(r.max()),
            "steps_mean": float(np.mean(steps_list)),
            "action_distribution": dict(action_totals),
        }
        all_results[tag] = summary

        print(f"\n  Summary: mean={r.mean():.2f} ± {r.std():.2f}  "
              f"min={r.min():.2f}  max={r.max():.2f}  "
              f"avg_steps={np.mean(steps_list):.0f}")

    # Final comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    for tag, s in all_results.items():
        print(f"  {tag:40s}  reward={s['reward_mean']:8.2f} ± {s['reward_std']:.2f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
