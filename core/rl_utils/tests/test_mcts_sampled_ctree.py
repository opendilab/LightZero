from pandas import to_pickle
import pytest
import torch
from easydict import EasyDict
import sys
# sys.path.append('/Users/yangzhenjie/code/jayyoung0802/LightZero/')
sys.path.append('/Users/puyuan/code/LightZero')


from core.rl_utils import inverse_scalar_transform


class MuZeroModelFake(torch.nn.Module):

    def __init__(self, action_num):
        super().__init__()
        self.action_num = action_num

    def initial_inference(self, observation):
        encoded_state = observation
        batch_size = encoded_state.shape[0]

        value = torch.zeros(size=(batch_size, 601))
        value_prefix = [0. for _ in range(batch_size)]
        # policy_logits = torch.zeros(size=(batch_size, self.action_num))
        policy_logits = 0.1*torch.ones(size=(batch_size, self.action_num))

        hidden_state = torch.zeros(size=(batch_size, 12, 3, 3))
        reward_hidden_state_state = (torch.zeros(size=(1, batch_size, 16)), torch.zeros(size=(1, batch_size, 16)))

        output = {
            'value': value,
            'value_prefix': value_prefix,
            'policy_logits': policy_logits,
            'hidden_state': hidden_state,
            'reward_hidden_state': reward_hidden_state_state
        }

        return EasyDict(output)

    def recurrent_inference(self, hidden_states, reward_hidden_states, actions):
        batch_size = hidden_states.shape[0]
        hidden_state = torch.zeros(size=(batch_size, 12, 3, 3))
        reward_hidden_state_state = (torch.zeros(size=(1, batch_size, 16)), torch.zeros(size=(1, batch_size, 16)))
        value = torch.zeros(size=(batch_size, 601))
        value_prefix = torch.zeros(size=(batch_size, 601))
        policy_logits = 0.1*torch.ones(size=(batch_size, self.action_num))

        # policy_logits = torch.zeros(size=(batch_size, self.action_num))

        output = {
            'value': value,
            'value_prefix': value_prefix,
            'policy_logits': policy_logits,
            'hidden_state': hidden_state,
            'reward_hidden_state': reward_hidden_state_state
        }

        return EasyDict(output)


# @pytest.mark.unittest
def test_mcts():
    import core.rl_utils.mcts.ctree_sampled_efficientzero.cytree as ctree
    import numpy as np
    from core.rl_utils.mcts.mcts_ctree_sampled_efficientzero import MCTSSampledCtree as MCTS

    game_config = EasyDict(
        dict(
            lstm_horizon_len=5,
            support_size=300,
            action_space_size=2,
            num_of_sampled_actions=6,
            num_simulations=100,
            batch_size=5,
            pb_c_base=1,
            pb_c_init=1,
            discount=0.9,
            root_dirichlet_alpha=0.3,
            root_exploration_fraction=0.2,
            dirichlet_alpha=0.3,
            exploration_fraction=1,
            device='cpu',
            value_delta_max=0,
        )
    )

    batch_size = env_nums = game_config.batch_size
    action_space_size = game_config.action_space_size

    model = MuZeroModelFake(action_num=game_config.action_space_size*2)
    stack_obs = torch.zeros(
        size=(
            batch_size,
            game_config.action_space_size*2,
        ), dtype=torch.float
    )

    network_output = model.initial_inference(stack_obs.float())

    hidden_state_roots = network_output['hidden_state']
    reward_hidden_state_state = network_output['reward_hidden_state']
    pred_values_pool = network_output['value']
    value_prefix_pool = network_output['value_prefix']
    policy_logits_pool = network_output['policy_logits']

    # network output process
    pred_values_pool = inverse_scalar_transform(pred_values_pool, game_config.support_size).detach().cpu().numpy()
    hidden_state_roots = hidden_state_roots.detach().cpu().numpy()
    reward_hidden_state_state = (
        reward_hidden_state_state[0].detach().cpu().numpy(), reward_hidden_state_state[1].detach().cpu().numpy()
    )
    policy_logits_pool = policy_logits_pool.detach().cpu().numpy().tolist()

    legal_actions_list = [[-1 for i in range(5)] for _ in range(env_nums)]
    # game_config.num_simulations
    # roots = ctree_efficientzero.Roots(env_nums, legal_actions_list, action_space_size=2, num_of_sampled_actions=20)
    roots = ctree.Roots(env_nums, legal_actions_list, game_config.action_space_size, game_config.num_of_sampled_actions)

    noises = [
        np.random.dirichlet([game_config.root_dirichlet_alpha] * game_config.num_of_sampled_actions
                            ).astype(np.float32).tolist() for _ in range(env_nums)
    ]
    to_play_batch = [int(np.random.randint(1, 2, 1)) for _ in range(env_nums)]
    roots.prepare(game_config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool, to_play_batch)

    MCTS(game_config).search(roots, model, hidden_state_roots, reward_hidden_state_state, to_play_batch)
    roots_distributions = roots.get_distributions()
    assert np.array(roots_distributions).shape == (batch_size, game_config.num_of_sampled_actions)
