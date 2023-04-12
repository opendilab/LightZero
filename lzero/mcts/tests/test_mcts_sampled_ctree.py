import pytest
import torch
from easydict import EasyDict
from lzero.policy import inverse_scalar_transform


class MuZeroModelFake(torch.nn.Module):
    """
    Overview:
        Fake MuZero model just for test EfficientZeroMCTSPtree.
    Interfaces:
        __init__, initial_inference, recurrent_inference
    """

    def __init__(self, action_num):
        super().__init__()
        self.action_num = action_num

    def initial_inference(self, observation):
        encoded_state = observation
        batch_size = encoded_state.shape[0]

        value = torch.zeros(size=(batch_size, 601))
        value_prefix = [0. for _ in range(batch_size)]
        # policy_logits = torch.zeros(size=(batch_size, self.action_num))
        policy_logits = 0.1 * torch.ones(size=(batch_size, self.action_num))

        latent_state = torch.zeros(size=(batch_size, 12, 3, 3))
        reward_hidden_state_state = (torch.zeros(size=(1, batch_size, 16)), torch.zeros(size=(1, batch_size, 16)))

        output = {
            'value': value,
            'value_prefix': value_prefix,
            'policy_logits': policy_logits,
            'latent_state': latent_state,
            'reward_hidden_state': reward_hidden_state_state
        }

        return EasyDict(output)

    def recurrent_inference(self, hidden_states, reward_hidden_states, actions):
        batch_size = hidden_states.shape[0]
        latent_state = torch.zeros(size=(batch_size, 12, 3, 3))
        reward_hidden_state_state = (torch.zeros(size=(1, batch_size, 16)), torch.zeros(size=(1, batch_size, 16)))
        value = torch.zeros(size=(batch_size, 601))
        value_prefix = torch.zeros(size=(batch_size, 601))
        policy_logits = 0.1 * torch.ones(size=(batch_size, self.action_num))
        # policy_logits = torch.zeros(size=(batch_size, self.action_num))

        output = {
            'value': value,
            'value_prefix': value_prefix,
            'policy_logits': policy_logits,
            'latent_state': latent_state,
            'reward_hidden_state': reward_hidden_state_state
        }

        return EasyDict(output)


@pytest.mark.unittest
def test_mcts():
    import numpy as np
    from lzero.mcts.tree_search.mcts_ctree_sampled import SampledEfficientZeroMCTSCtree as MCTSCtree

    policy_config = EasyDict(
        dict(
            lstm_horizon_len=5,
            num_of_sampled_actions=6,
            num_simulations=100,
            batch_size=5,
            pb_c_base=1,
            pb_c_init=1,
            discount_factor=0.9,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.2,
            dirichlet_alpha=0.3,
            exploration_fraction=1,
            device='cpu',
            value_delta_max=0,
            model=dict(
                continuous_action_space=True,
                support_scale=300,
                action_space_size=2,
                categorical_distribution=True,
            ),
        )
    )

    batch_size = env_nums = policy_config.batch_size
    model = MuZeroModelFake(action_num=policy_config.model.action_space_size * 2)
    stack_obs = torch.zeros(
        size=(
            batch_size,
            policy_config.model.action_space_size * 2,
        ), dtype=torch.float
    )

    network_output = model.initial_inference(stack_obs.float())

    latent_state_roots = network_output['latent_state']
    reward_hidden_state_state = network_output['reward_hidden_state']
    pred_values_pool = network_output['value']
    value_prefix_pool = network_output['value_prefix']
    policy_logits_pool = network_output['policy_logits']

    # network output process
    pred_values_pool = inverse_scalar_transform(pred_values_pool,
                                                policy_config.model.support_scale).detach().cpu().numpy()
    latent_state_roots = latent_state_roots.detach().cpu().numpy()
    reward_hidden_state_state = (
        reward_hidden_state_state[0].detach().cpu().numpy(), reward_hidden_state_state[1].detach().cpu().numpy()
    )
    policy_logits_pool = policy_logits_pool.detach().cpu().numpy().tolist()

    legal_actions_list = [[-1 for i in range(5)] for _ in range(env_nums)]
    roots = MCTSCtree.roots(
        env_nums,
        legal_actions_list,
        policy_config.model.action_space_size,
        policy_config.num_of_sampled_actions,
        continuous_action_space=True
    )

    noises = [
        np.random.dirichlet([policy_config.root_dirichlet_alpha] * policy_config.num_of_sampled_actions
                            ).astype(np.float32).tolist() for _ in range(env_nums)
    ]
    to_play_batch = [int(np.random.randint(1, 2, 1)) for _ in range(env_nums)]
    roots.prepare(policy_config.root_noise_weight, noises, value_prefix_pool, policy_logits_pool, to_play_batch)

    MCTSCtree(policy_config).search(roots, model, latent_state_roots, reward_hidden_state_state, to_play_batch)
    roots_distributions = roots.get_distributions()
    assert np.array(roots_distributions).shape == (batch_size, policy_config.num_of_sampled_actions)
