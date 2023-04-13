import numpy as np
import pytest
import torch
from easydict import EasyDict

from lzero.policy import inverse_scalar_transform, select_action
from lzero.mcts.tree_search.mcts_ctree import EfficientZeroMCTSCtree as MCTSCtree


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
        policy_logits = torch.zeros(size=(batch_size, self.action_num))
        latent_state = torch.zeros(size=(batch_size, 12, 3, 3))
        reward_hidden_state_roots = (torch.zeros(size=(1, batch_size, 16)), torch.zeros(size=(1, batch_size, 16)))

        output = {
            'value': value,
            'value_prefix': value_prefix,
            'policy_logits': policy_logits,
            'latent_state': latent_state,
            'reward_hidden_state': reward_hidden_state_roots
        }

        return EasyDict(output)

    def recurrent_inference(self, latent_states, reward_hidden_states, actions):
        batch_size = latent_states.shape[0]
        latent_state = torch.zeros(size=(batch_size, 12, 3, 3))
        reward_hidden_state_roots = (torch.zeros(size=(1, batch_size, 16)), torch.zeros(size=(1, batch_size, 16)))
        value = torch.zeros(size=(batch_size, 601))
        value_prefix = torch.zeros(size=(batch_size, 601))
        policy_logits = torch.zeros(size=(batch_size, self.action_num))

        output = {
            'value': value,
            'value_prefix': value_prefix,
            'policy_logits': policy_logits,
            'latent_state': latent_state,
            'reward_hidden_state': reward_hidden_state_roots
        }

        return EasyDict(output)


policy_config = EasyDict(
    lstm_horizon_len=5,
    num_simulations=8,
    batch_size=16,
    pb_c_base=1,
    pb_c_init=1,
    discount_factor=0.9,
    root_dirichlet_alpha=0.3,
    root_noise_weight=0.2,
    dirichlet_alpha=0.3,
    exploration_fraction=1,
    device='cpu',
    value_delta_max=0.01,
    model=dict(
        action_space_size=9,
        support_scale=300,
        categorical_distribution=True,
    ),
)

batch_size = env_nums = policy_config.batch_size
action_space_size = policy_config.model.action_space_size

model = MuZeroModelFake(action_num=9)
stack_obs = torch.zeros(size=(batch_size, 8), dtype=torch.float)

network_output = model.initial_inference(stack_obs.float())

latent_state_roots = network_output['latent_state']
reward_hidden_state_roots = network_output['reward_hidden_state']
pred_values_pool = network_output['value']
value_prefix_pool = network_output['value_prefix']
policy_logits_pool = network_output['policy_logits']

# network output process
pred_values_pool = inverse_scalar_transform(pred_values_pool, policy_config.model.support_scale).detach().cpu().numpy()
latent_state_roots = latent_state_roots.detach().cpu().numpy()
reward_hidden_state_roots = (
    reward_hidden_state_roots[0].detach().cpu().numpy(), reward_hidden_state_roots[1].detach().cpu().numpy()
)
policy_logits_pool = policy_logits_pool.detach().cpu().numpy().tolist()

action_mask = [
    [0, 0, 0, 1, 0, 1, 1, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 1, 1, 0],
    [1, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 0, 0, 1],
]
assert len(action_mask) == batch_size
assert len(action_mask[0]) == action_space_size

action_num = [
    int(np.array(action_mask[i]).sum()) for i in range(env_nums)
]  # [3, 3, 5, 4, 3, 3, 6, 6, 3, 6, 6, 5, 2, 5, 1, 4]
legal_actions_list = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(env_nums)]
# legal_actions_list =
# [[3, 5, 6], [0, 3, 6], [0, 1, 4, 6, 8], [0, 3, 4, 5],
# [2, 5, 8], [1, 2, 4], [0, 2, 3, 4, 7, 8], [0, 1, 2, 3, 4, 8],
# [3, 5, 6], [1, 2, 4, 5, 6, 7], [0, 1, 2, 6, 7, 8], [0, 1, 3, 5, 6],
# [2, 5], [0, 2, 3, 6, 7], [1], [0, 4, 5, 8]]
to_play = [2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1]
assert len(to_play) == batch_size


@pytest.mark.unittest
def test_mcts_vs_bot_to_play():
    legal_actions_list = [[i for i in range(action_space_size)] for _ in range(env_nums)]  # all action
    roots = MCTSCtree.roots(env_nums, legal_actions_list)
    noises = [
        np.random.dirichlet([policy_config.root_dirichlet_alpha] * policy_config.model.action_space_size
                            ).astype(np.float32).tolist() for _ in range(env_nums)
    ]
    # In ctree, to_play must be list, not None
    roots.prepare(
        policy_config.root_noise_weight, noises, value_prefix_pool, policy_logits_pool, [0 for _ in range(env_nums)]
    )
    MCTSCtree(policy_config
              ).search(roots, model, latent_state_roots, reward_hidden_state_roots, [0 for _ in range(env_nums)])
    roots_distributions = roots.get_distributions()
    roots_values = roots.get_values()
    assert np.array(roots_distributions).shape == (batch_size, action_space_size)
    assert np.array(roots_values).shape == (batch_size, )


@pytest.mark.unittest
def test_mcts_vs_bot_to_play_large():
    policy_config.obs_space_size = 100
    policy_config.model.action_space_size = 20

    policy_config.num_simulations = 500
    policy_config.batch_size = 256
    env_nums = policy_config.batch_size

    model = MuZeroModelFake(action_num=policy_config.model.action_space_size)
    # stack_obs = torch.zeros(size=(policy_config.batch_size, policy_config.obs_space_size), dtype=torch.float)
    stack_obs = torch.randn(size=(policy_config.batch_size, policy_config.obs_space_size), dtype=torch.float)

    network_output = model.initial_inference(stack_obs.float())

    latent_state_roots = network_output['latent_state']
    reward_hidden_state_roots = network_output['reward_hidden_state']
    pred_values_pool = network_output['value']
    value_prefix_pool = network_output['value_prefix']
    policy_logits_pool = network_output['policy_logits']

    # network output process
    pred_values_pool = inverse_scalar_transform(pred_values_pool,
                                                policy_config.model.support_scale).detach().cpu().numpy()
    latent_state_roots = latent_state_roots.detach().cpu().numpy()
    reward_hidden_state_roots = (
        reward_hidden_state_roots[0].detach().cpu().numpy(), reward_hidden_state_roots[1].detach().cpu().numpy()
    )
    policy_logits_pool = policy_logits_pool.detach().cpu().numpy().tolist()

    # all actions are legal
    legal_actions_list = [[i for i in range(policy_config.model.action_space_size)] for _ in range(env_nums)]

    roots = MCTSCtree.roots(env_nums, legal_actions_list)
    noises = [
        np.random.dirichlet([policy_config.root_dirichlet_alpha] * policy_config.model.action_space_size
                            ).astype(np.float32).tolist() for _ in range(env_nums)
    ]
    # In ctree, to_play must be list, not None
    roots.prepare(
        policy_config.root_noise_weight, noises, value_prefix_pool, policy_logits_pool, [0 for _ in range(env_nums)]
    )
    MCTSCtree(policy_config
              ).search(roots, model, latent_state_roots, reward_hidden_state_roots, [0 for _ in range(env_nums)])
    roots_distributions = roots.get_distributions()
    roots_values = roots.get_values()
    assert np.array(roots_distributions).shape == (policy_config.batch_size, policy_config.model.action_space_size)
    assert np.array(roots_values).shape == (policy_config.batch_size, )


@pytest.mark.unittest
def test_mcts_vs_bot_to_play_legal_action():
    for i in range(env_nums):
        assert action_num[i] == len(legal_actions_list[i])

    roots = MCTSCtree.roots(env_nums, legal_actions_list)
    noises = [
        np.random.dirichlet([policy_config.root_dirichlet_alpha] * int(sum(action_mask[j]))).astype(np.float32).tolist()
        for j in range(env_nums)
    ]

    # In ctree, to_play must be list, not None
    roots.prepare(
        policy_config.root_noise_weight, noises, value_prefix_pool, policy_logits_pool, [0 for _ in range(env_nums)]
    )
    MCTSCtree(policy_config
              ).search(roots, model, latent_state_roots, reward_hidden_state_roots, [0 for _ in range(env_nums)])
    roots_distributions = roots.get_distributions()
    roots_values = roots.get_values()
    assert len(roots_values) == env_nums
    assert len(roots_values) == env_nums
    for i in range(env_nums):
        assert len(roots_distributions[i]) == action_num[i]

    temperature = [1 for _ in range(env_nums)]
    for i in range(env_nums):
        distributions = roots_distributions[i]
        action_index, visit_count_distribution_entropy = select_action(
            distributions, temperature=temperature[i], deterministic=False
        )
        action = np.where(np.array(action_mask[i]) == 1.0)[0][action_index]
        assert action_index < action_num[i]
        assert action == legal_actions_list[i][action_index]
        print('\n action_index={}, legal_action={}, action={}'.format(action_index, legal_actions_list[i], action))


@pytest.mark.unittest
def test_mcts_self_play():
    legal_actions_list = [[i for i in range(action_space_size)] for _ in range(env_nums)]  # all action
    roots = MCTSCtree.roots(env_nums, legal_actions_list)
    noises = [
        np.random.dirichlet([policy_config.root_dirichlet_alpha] * policy_config.model.action_space_size
                            ).astype(np.float32).tolist() for _ in range(env_nums)
    ]
    # In ctree, to_play must be list, not None
    roots.prepare(policy_config.root_noise_weight, noises, value_prefix_pool, policy_logits_pool, to_play)
    MCTSCtree(policy_config).search(roots, model, latent_state_roots, reward_hidden_state_roots, to_play)
    roots_distributions = roots.get_distributions()
    roots_values = roots.get_values()
    assert np.array(roots_distributions).shape == (batch_size, action_space_size)
    assert np.array(roots_values).shape == (batch_size, )


@pytest.mark.unittest
def test_mcts_self_play_legal_action():
    for i in range(env_nums):
        assert action_num[i] == len(legal_actions_list[i])

    roots = MCTSCtree.roots(env_nums, legal_actions_list)
    noises = [
        np.random.dirichlet([policy_config.root_dirichlet_alpha] * int(sum(action_mask[j]))).astype(np.float32).tolist()
        for j in range(env_nums)
    ]
    # In ctree, to_play must be list, not None
    roots.prepare(policy_config.root_noise_weight, noises, value_prefix_pool, policy_logits_pool, to_play)
    MCTSCtree(policy_config).search(roots, model, latent_state_roots, reward_hidden_state_roots, to_play)
    roots_distributions = roots.get_distributions()
    roots_values = roots.get_values()
    assert len(roots_values) == env_nums
    assert len(roots_values) == env_nums
    for i in range(env_nums):
        assert len(roots_distributions[i]) == action_num[i]

    temperature = [1 for _ in range(env_nums)]
    for i in range(env_nums):
        distributions = roots_distributions[i]
        action_index, visit_count_distribution_entropy = select_action(
            distributions, temperature=temperature[i], deterministic=False
        )
        action = np.where(np.array(action_mask[i]) == 1.0)[0][action_index]
        assert action_index < action_num[i]
        assert action == legal_actions_list[i][action_index]
        print('\n action_index={}, legal_action={}, action={}'.format(action_index, legal_actions_list[i], action))
