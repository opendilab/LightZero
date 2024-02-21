import numpy as np
import pytest
import torch

from lzero.mcts.buffer.game_segment import GameSegment
from lzero.mcts.utils import prepare_observation
from lzero.policy import select_action

# args = ['EfficientZero', 'MuZero']
args = ["MuZero"]


@pytest.mark.unittest
@pytest.mark.parametrize('test_algo', args)
def test_game_segment(test_algo):
    # import different modules according to ``test_algo``
    if test_algo == 'EfficientZero':
        from lzero.mcts.tree_search.mcts_ctree import EfficientZeroMCTSCtree as MCTSCtree
        from lzero.model.efficientzero_model import EfficientZeroModel as Model
        from lzero.mcts.tests.config.atari_efficientzero_config_for_test import atari_efficientzero_config as config
        from zoo.atari.envs.atari_lightzero_env import AtariLightZeroEnv
        envs = [AtariLightZeroEnv(config.env) for _ in range(config.env.evaluator_env_num)]

    elif test_algo == 'MuZero':
        from lzero.mcts.tree_search.mcts_ctree import MuZeroMCTSCtree as MCTSCtree
        from lzero.model.muzero_model import MuZeroModel as Model
        from lzero.mcts.tests.config.tictactoe_muzero_bot_mode_config_for_test import tictactoe_muzero_config as config
        from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv
        envs = [TicTacToeEnv(config.env) for _ in range(config.env.evaluator_env_num)]

    # create model
    model = Model(**config.policy.model)
    if config.policy.cuda and torch.cuda.is_available():
        config.policy.device = 'cuda'
    else:
        config.policy.device = 'cpu'
    model.to(config.policy.device)
    model.eval()

    with torch.no_grad():
        # initializations
        init_observations = [env.reset() for env in envs]
        dones = np.array([False for _ in range(config.env.evaluator_env_num)])
        game_segments = [
            GameSegment(
                envs[i].action_space, game_segment_length=config.policy.game_segment_length, config=config.policy
            ) for i in range(config.env.evaluator_env_num)
        ]
        for i in range(config.env.evaluator_env_num):
            game_segments[i].reset(
                [init_observations[i]['observation'] for _ in range(config.policy.model.frame_stack_num)]
            )
        episode_rewards = np.zeros(config.env.evaluator_env_num)

        while not dones.all():
            stack_obs = [game_segment.get_obs() for game_segment in game_segments]
            stack_obs = prepare_observation(stack_obs, config.policy.model.model_type)
            stack_obs = torch.from_numpy(np.array(stack_obs)).to(config.policy.device)

            # ==============================================================
            # the core initial_inference.
            # ==============================================================
            network_output = model.initial_inference(stack_obs)

            # process the network output
            policy_logits_pool = network_output.policy_logits.detach().cpu().numpy().tolist()
            latent_state_roots = network_output.latent_state.detach().cpu().numpy()

            if test_algo == 'EfficientZero':
                reward_hidden_state_roots = network_output.reward_hidden_state
                value_prefix_pool = network_output.value_prefix
                reward_hidden_state_roots = (
                    reward_hidden_state_roots[0].detach().cpu().numpy(),
                    reward_hidden_state_roots[1].detach().cpu().numpy()
                )
                # for atari env, all actions are legal_action
                legal_actions_list = [
                    [i for i in range(config.policy.model.action_space_size)]
                    for _ in range(config.env.evaluator_env_num)
                ]
            elif test_algo == 'MuZero':
                reward_pool = network_output.reward
                # for board games, we use the all actions is legal_action
                legal_actions_list = [
                    [a for a, x in enumerate(init_observations[i]['action_mask']) if x == 1]
                    for i in range(config.env.evaluator_env_num)
                ]

            # null padding for the atari games and board_games in vs_bot_mode
            to_play = [-1 for _ in range(config.env.evaluator_env_num)]

            if test_algo == 'EfficientZero':
                roots = MCTSCtree.roots(config.env.evaluator_env_num, legal_actions_list)
                roots.prepare_no_noise(value_prefix_pool, policy_logits_pool, to_play)
                MCTSCtree(config.policy).search(roots, model, latent_state_roots, reward_hidden_state_roots, to_play)

            elif test_algo == 'MuZero':
                roots = MCTSCtree.roots(config.env.evaluator_env_num, legal_actions_list)
                roots.prepare_no_noise(reward_pool, policy_logits_pool, to_play)
                MCTSCtree(config.policy).search(roots, model, latent_state_roots, to_play)

            roots_distributions = roots.get_distributions()
            roots_values = roots.get_values()

            for i in range(config.env.evaluator_env_num):
                distributions, value, env = roots_distributions[i], roots_values[i], envs[i]
                # ``deterministic=True`` indicates that we select the argmax action instead of sampling.
                action, _ = select_action(distributions, temperature=1, deterministic=True)
                # ==============================================================
                # the core initial_inference.
                # ==============================================================
                obs, reward, done, info = env.step(action)
                obs = obs['observation']

                game_segments[i].store_search_stats(distributions, value)
                game_segments[i].append(action, obs, reward)

                dones[i] = done
                episode_rewards[i] += reward
                if dones[i]:
                    continue

        for env in envs:
            env.close()
