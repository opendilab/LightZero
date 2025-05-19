import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F
from lzero.model.unizero_world_models.world_model import WorldModel
from zoo.atari.envs import

def evaluate_and_log_one_trajectory_metrics_only(
    wm: WorldModel,
    env_name: str,
    num_simulations: int = 50,
    max_steps: int = 50,
    run_name: str = None
):
    """
    Rollout one episode of `env_name` using MCTS+world-model,
    collect at each step:
      - true vs pred reward
      - prior policy (world-model)
      - mcts policy (visit-count distribution)
    and log a 3×T plot to wandb (no image frames).
    """
    wandb.init(project="unizero-worldmodel", name=run_name, reinit=True)
    env = make_env(env_name)
    state = env.reset()
    last_action = -1

    true_rewards, pred_rewards = [], []
    prior_policies, mcts_policies = [], []

    for t in range(max_steps):
        # 1) world-model prior pass
        # build the dict your forward_initial_inference expects:
        obs_act = {
          "obs":         torch.from_numpy(env.render(mode="rgb_array"))
                             .permute(2,0,1).unsqueeze(0)
                             .float().to(wm.device),
          "action":      last_action,
          "current_obs": torch.from_numpy(env.render(mode="rgb_array"))
                             .permute(2,0,1).unsqueeze(0)
                             .float().to(wm.device)
        }
        _, _, logits_r, logits_prior, _ = wm.forward_initial_inference(obs_act, start_pos=t)

        # 2) record predicted & true reward
        pr = wm.scalar_transform.inverse(logits_r).item()
        pred_rewards.append(pr)

        # 3) record prior policy
        pp = F.softmax(logits_prior, dim=-1).cpu().numpy().squeeze(0)
        prior_policies.append(pp)

        # 4) MCTS search → improved policy
        #root = MCTS(wm, state, num_simulations=num_simulations)
        counts = np.array([c.visit_count for c in root.children], dtype=np.float32)
        pi_mcts = counts / counts.sum()
        mcts_policies.append(pi_mcts)

        # 5) step env
        action = int(pi_mcts.argmax())
        state, rew, done, _ = env.step(action)
        last_action = action
        true_rewards.append(rew)
        if done:
            break

    T = len(pred_rewards)
    action_count = prior_policies[0].shape[-1]
    colors = [f"C{i}" for i in range(action_count)]

    # create a 3×T grid
    fig, axes = plt.subplots(3, T, figsize=(2*T, 6),
                             gridspec_kw={'height_ratios':[1,2,2]})

    # Row 1: rewards
    for t in range(T):
        axes[0,t].scatter(t, true_rewards[t],  c='green', marker='x')
        axes[0,t].scatter(t, pred_rewards[t],  c='blue',  marker='o')
        axes[0,t].set_xticks([])
        axes[0,t].set_yticks([])
    axes[0,0].legend(['true','pred'], loc='upper left')
    axes[0,0].set_ylabel("Reward")

    # Row 2: prior policy bar charts
    for t in range(T):
        axes[1,t].bar(np.arange(action_count), prior_policies[t], color=colors)
        axes[1,t].set_ylim(0,1.0)
        axes[1,t].set_xticks([])
        axes[1,t].set_yticks([])
    axes[1,0].set_ylabel("Prior π")

    # Row 3: MCTS policy bar charts
    for t in range(T):
        axes[2,t].bar(np.arange(action_count), mcts_policies[t], color=colors)
        axes[2,t].set_ylim(0,1.0)
        axes[2,t].set_xticks([])
        axes[2,t].set_yticks([])
    axes[2,0].set_ylabel("MCTS π")

    plt.tight_layout()

    # log to W&B
    wandb.log({
      "trajectory_metrics_only": wandb.Image(fig,
        caption=f"Unizero world-model metrics on {env_name}")
    })
    plt.close(fig)
    env.close()
    wandb.finish()
