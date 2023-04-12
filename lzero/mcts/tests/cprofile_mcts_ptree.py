import torch
from easydict import EasyDict

from lzero.policy.scaling_transform import inverse_scalar_transform


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
        policy_logits = torch.zeros(size=(batch_size, self.action_num))

        output = {
            'value': value,
            'value_prefix': value_prefix,
            'policy_logits': policy_logits,
            'latent_state': latent_state,
            'reward_hidden_state': reward_hidden_state_state
        }

        return EasyDict(output)


def check_mcts():
    import numpy as np
    from lzero.mcts.tree_search.mcts_ptree import EfficientZeroMCTSPtree as MCTSPtree

    policy_config = EasyDict(
        dict(
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
                categorical_distribution=True,
                support_scale=300,
            ),
        )
    )

    env_nums = policy_config.batch_size

    model = MuZeroModelFake(action_num=100)
    stack_obs = torch.zeros(
        size=(
            policy_config.batch_size,
            100,
        ), dtype=torch.float
    )

    network_output = model.initial_inference(stack_obs.float())

    latent_state_roots = network_output['hidden_state']
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

    legal_actions_list = [
        [i for i in range(policy_config.model.action_space_size)] for _ in range(env_nums)
    ]  # all action
    roots = MCTSPtree.roots(env_nums, legal_actions_list)
    noises = [
        np.random.dirichlet([policy_config.root_dirichlet_alpha] * policy_config.model.action_space_size
                            ).astype(np.float32).tolist() for _ in range(env_nums)
    ]
    roots.prepare(policy_config.root_noise_weight, noises, value_prefix_pool, policy_logits_pool)
    MCTSPtree(policy_config).search(roots, model, latent_state_roots, reward_hidden_state_state)
    roots_distributions = roots.get_distributions()
    assert np.array(roots_distributions).shape == (policy_config.batch_size, policy_config.model.action_space_size)


if __name__ == '__main__':
    import cProfile

    run_num = 10

    def profile_mcts(run_num):
        for i in range(run_num):
            check_mcts()

    # Save the analysis results to a file.
    cProfile.run(f"profile_mcts({run_num})", filename="result.out")
