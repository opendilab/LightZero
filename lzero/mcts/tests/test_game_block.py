import numpy as np
import pytest
import torch

from lzero.mcts.buffer.game_block import GameBlock
from lzero.mcts.utils import prepare_observation_list
from lzero.policy import select_action


args = ['EfficientZero', 'MuZero']
# args = ['MuZero']


@pytest.mark.unittest
@pytest.mark.parametrize('test_algo', args)
def test_game_block(test_algo):
    # import different modules according to ``test_algo``
    if test_algo == 'EfficientZero':
        from lzero.mcts.tree_search.mcts_ctree import EfficientZeroMCTSCtree as MCTSCtree
        from lzero.model.efficientzero_model import EfficientZeroModel as Model
        from lzero.mcts.tests.pong_efficientzero_config_test import pong_efficientzero_config as config
        from zoo.atari.envs.atari_lightzero_env import AtariLightZeroEnv
        envs = [AtariLightZeroEnv(config.env) for _ in range(config.env.evaluator_env_num)]

    elif test_algo == 'MuZero':
        from lzero.mcts.tree_search.mcts_ctree import MuZeroMCTSCtree as MCTSCtree
        from lzero.model.muzero_model import MuZeroModel as Model
        from lzero.mcts.tests.tictactoe_muzero_bot_mode_config_test import tictactoe_muzero_config as config
        from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv
        envs = [TicTacToeEnv(config.env) for _ in range(config.env.evaluator_env_num)]

    # set config for test
    config.env.evaluator_env_num = 2
    config.env.num_simulations = 2
    config.env.game_block_length = 20
    # config.env.render_mode_human = True

    # create model
    model = Model(**config.policy.model)
    model.to(config.policy.device)
    model.eval()

    with torch.no_grad():
        # initializations
        init_observations = [env.reset() for env in envs]
        dones = np.array([False for _ in range(config.env.evaluator_env_num)])
        game_blocks = [
            GameBlock(
                envs[i].action_space, game_block_length=config.policy.game_block_length, config=config.policy
            ) for i in range(config.env.evaluator_env_num)
        ]
        for i in range(config.env.evaluator_env_num):
            game_blocks[i].init([init_observations[i]['observation'] for _ in range(config.policy.model.frame_stack_num)])
        episode_rewards = np.zeros(config.env.evaluator_env_num)

        while not dones.all():
            stack_obs = [game_block.step_obs() for game_block in game_blocks]
            stack_obs = prepare_observation_list(stack_obs)
            stack_obs = torch.from_numpy(np.array(stack_obs)).to(config.policy.device)

            # ==============================================================
            # the core initial_inference.
            # ==============================================================
            network_output = model.initial_inference(stack_obs)

            # process the network output
            policy_logits_pool = network_output.policy_logits.detach().cpu().numpy().tolist()
            hidden_state_roots = network_output.hidden_state.detach().cpu().numpy()

            if test_algo == 'EfficientZero':
                reward_hidden_state_roots = network_output.reward_hidden_state
                value_prefix_pool = network_output.value_prefix
                reward_hidden_state_roots = (
                    reward_hidden_state_roots[0].detach().cpu().numpy(),
                    reward_hidden_state_roots[1].detach().cpu().numpy()
                )
                # for atari env, all actions is legal_action
                legal_actions_list = [
                    [i for i in range(config.policy.model.action_space_size)] for _ in
                    range(config.env.evaluator_env_num)
                ]
            elif test_algo == 'MuZero':
                reward_pool = network_output.reward
                # for board games, we use the all actions is legal_action
                legal_actions_list = [
                    [a for a, x in enumerate(init_observations[i]['action_mask']) if x == 1] for i in range(config.env.evaluator_env_num)
                ]

            # null padding for the atari games and board_games in vs_bot_mode
            to_play = [-1 for _ in range(config.env.evaluator_env_num)]

            if test_algo == 'EfficientZero':
                roots = MCTSCtree.Roots(config.env.evaluator_env_num,
                                      legal_actions_list)
                roots.prepare_no_noise(value_prefix_pool, policy_logits_pool, to_play)
                MCTSCtree(config.policy).search(roots, model, hidden_state_roots, reward_hidden_state_roots, to_play)

            elif test_algo == 'MuZero':
                roots = MCTSCtree.Roots(config.env.evaluator_env_num, legal_actions_list)
                roots.prepare_no_noise(reward_pool, policy_logits_pool, to_play)
                MCTSCtree(config.policy).search(roots, model, hidden_state_roots, to_play)

            roots_distributions = roots.get_distributions()
            roots_values = roots.get_values()

            for i in range(config.env.evaluator_env_num):
                distributions, value, env = roots_distributions[i], roots_values[i], envs[i]
                # ``deterministic=True``  indicates that we select the argmax action instead of sampling.
                action, _ = select_action(distributions, temperature=1, deterministic=True)
                # ==============================================================
                # the core initial_inference.
                # ==============================================================
                obs, reward, done, info = env.step(action)
                obs = obs['observation']

                game_blocks[i].store_search_stats(distributions, value)
                game_blocks[i].append(action, obs, reward)

                dones[i] = done
                episode_rewards[i] += reward
                if dones[i]:
                    continue

        for env in envs:
            env.close()


# debug
test_game_block('EfficientZero')
