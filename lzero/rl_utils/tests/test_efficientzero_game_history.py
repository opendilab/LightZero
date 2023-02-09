"""
The following code is adapted from https://github.com/YeWR/EfficientZero/blob/main/core/test.py
"""
import numpy as np
import torch
from torch.cuda.amp import autocast as autocast

from lzero.rl_utils.mcts.ctree_efficientzero import ez_tree
from lzero.rl_utils.mcts.game import GameHistory
from lzero.rl_utils.mcts.utils import select_action, prepare_observation_list

# args = ['PongNoFrameskip-v4', 'tictactoe']

args = ['PongNoFrameskip-v4']


@pytest.mark.unittest
@pytest.mark.parametrize('env_name', args)
def test_game_history(env_name):
    if env_name == 'PongNoFrameskip-v4':
        from lzero.rl_utils.mcts.mcts_ctree import EfficientZeroMCTSCtree as MCTS
        from lzero.model.efficientzero.efficientzero_model import EfficientZeroNet as Model
        from zoo.atari.config.pong_efficientzero_config import pong_efficientzero_config as config
    elif env_name == 'tictactoe':
        from lzero.rl_utils.mcts.mcts_ctree import MuZeroMCTSCtree as MCTS
        from lzero.model.muzero.muzero_model import MuZeroNet as Model
        from zoo.board_games.tictactoe.config.tictactoe_muzero_1pm_config import tictactoe_muzero_config as config

    # set some additional config for test
    config.device = 'cpu'
    config.device = config.device

    config.env.evaluator_env_num = 2
    config.env.render_mode_human = False
    config.render = False
    config.env.num_simulations = 2
    config.env.game_history_length = 20

    # to obtain model
    model = Model(**config.policy.model)
    model.to(config.device)
    model.eval()

    with torch.no_grad():
        if env_name == 'PongNoFrameskip-v4':
            from zoo.atari.envs.atari_lightzero_env import AtariLightZeroEnv
            envs = [AtariLightZeroEnv(config.env) for i in range(config.env.evaluator_env_num)]
        elif env_name == 'tictactoe':
            from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv
            envs = [TicTacToeEnv(config.env) for i in range(config.env.evaluator_env_num)]

        # initializations
        init_obses = [env.reset() for env in envs]
        dones = np.array([False for _ in range(config.env.evaluator_env_num)])
        game_histories = [
            GameHistory(
                envs[i].action_space, game_history_length=config.policy.game_history_length, config=config.policy
            ) for i in range(config.env.evaluator_env_num)
        ]
        for i in range(config.env.evaluator_env_num):
            game_histories[i].init([init_obses[i]['observation'] for _ in range(config.policy.frame_stack_num)])

        ep_ori_rewards = np.zeros(config.env.evaluator_env_num)
        ep_clip_rewards = np.zeros(config.env.evaluator_env_num)
        # loop
        while not dones.all():
            if config.render:
                for i in range(config.env.evaluator_env_num):
                    envs[i].render()
            stack_obs = [game_history.step_obs() for game_history in game_histories]
            stack_obs = prepare_observation_list(stack_obs)
            if config.policy.image_based:
                stack_obs = torch.from_numpy(stack_obs).to(config.device).float() / 255.0
            else:
                stack_obs = torch.from_numpy(np.array(stack_obs)).to(config.device)
            with autocast():
                # stack_obs {Tensor:(2,12,96,96)}
                network_output = model.initial_inference(stack_obs.float())

            value_prefix_pool = network_output.value_prefix  # {list: 2}->{float}
            policy_logits_pool = network_output.policy_logits

            hidden_state_roots = network_output.hidden_state  # {ndarray:（2, 64, 6, 6）}
            if env_name == 'PongNoFrameskip-v4':
                reward_hidden_state_roots = network_output.reward_hidden_state  # {tuple:2}->{ndarray:(1,2,512)}
            elif env_name == 'tictactoe':
                # TODO
                reward_hidden_state_roots = network_output.reward  # {tuple:2}->{ndarray:(1,2,512)}

            # network output process
            hidden_state_roots = hidden_state_roots.detach().cpu().numpy()
            reward_hidden_state_roots = (
                reward_hidden_state_roots[0].detach().cpu().numpy(), reward_hidden_state_roots[1].detach().cpu().numpy()
            )
            policy_logits_pool = policy_logits_pool.detach().cpu().numpy().tolist()

            legal_actions_list = [
                [i for i in range(config.policy.action_space_size)] for _ in range(config.env.evaluator_env_num)
            ]  # all action
            roots = ez_tree.Roots(config.env.evaluator_env_num, config.policy.action_space_size, legal_actions_list)
            to_play = [-1 for _ in range(config.env.evaluator_env_num)]
            roots.prepare_no_noise(value_prefix_pool, policy_logits_pool, to_play)
            # do MCTS for a policy (argmax in testing)
            MCTS(config.policy).search(roots, model, hidden_state_roots, reward_hidden_state_roots, to_play)
            roots_distributions = roots.get_distributions()  # {list: 1}->{list:6}
            roots_values = roots.get_values()  # {list: 1}
            for i in range(config.env.evaluator_env_num):
                if dones[i]:
                    continue
                distributions, value, env = roots_distributions[i], roots_values[i], envs[i]
                # select the argmax, not sampling
                action, _ = select_action(distributions, temperature=1, deterministic=True)
                obs, ori_reward, done, info = env.step(action)
                obs = obs['observation']
                if config.policy.clip_reward:
                    clip_reward = np.sign(ori_reward)
                else:
                    clip_reward = ori_reward

                game_histories[i].store_search_stats(distributions, value)
                game_histories[i].append(action, obs, clip_reward)

                dones[i] = done
                ep_ori_rewards[i] += ori_reward
                ep_clip_rewards[i] += clip_reward

        for env in envs:
            env.close()


# debug
test_game_history('PongNoFrameskip-v4')