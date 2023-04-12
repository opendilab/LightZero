import torch
from easydict import EasyDict
from lzero.policy import inverse_scalar_transform, select_action
import numpy as np
import random

from lzero.mcts.tree_search.mcts_ptree import EfficientZeroMCTSPtree as MCTSPtree
from lzero.mcts.tree_search.mcts_ctree import EfficientZeroMCTSCtree as MCTSCtree
import time


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


def ptree_func(policy_config, num_simulations):
    """
        Overview:
            Search on the tree of the Python implementation and record the time spent at different stages.
        Arguments:
            - policy_config: config of game.
            - num_simulations: Number of simulations.
        Returns:
            - build_time: Type builds take time.
            - prepare_time: time for prepare.
            - search_time.
            - total_time.
        """
    batch_size = env_nums = policy_config.batch_size
    action_space_size = policy_config.action_space_size

    build_time = []
    prepare_time = []
    search_time = []
    total_time = []

    for n_s in num_simulations:
        t0 = time.time()
        model = MuZeroModelFake(action_num=action_space_size)
        stack_obs = torch.zeros(
            size=(
                batch_size,
                n_s,
            ), dtype=torch.float
        )

        policy_config.num_simulations = n_s
        network_output = model.initial_inference(stack_obs.float())

        hidden_state_roots = network_output['hidden_state']
        reward_hidden_state_state = network_output['reward_hidden_state']
        pred_values_pool = network_output['value']
        value_prefix_pool = network_output['value_prefix']
        policy_logits_pool = network_output['policy_logits']

        # network output process
        pred_values_pool = inverse_scalar_transform(pred_values_pool,
                                                    policy_config.model.support_scale).detach().cpu().numpy()
        hidden_state_roots = hidden_state_roots.detach().cpu().numpy()
        reward_hidden_state_state = (
            reward_hidden_state_state[0].detach().cpu().numpy(), reward_hidden_state_state[1].detach().cpu().numpy()
        )
        policy_logits_pool = policy_logits_pool.detach().cpu().numpy().tolist()

        action_mask = [[random.randint(0, 1) for _ in range(action_space_size)] for _ in range(env_nums)]
        assert len(action_mask) == batch_size
        assert len(action_mask[0]) == action_space_size

        action_num = [int(np.array(action_mask[i]).sum()) for i in range(env_nums)]
        legal_actions_list = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(env_nums)]
        to_play = [np.random.randint(1, 3) for i in range(env_nums)]
        assert len(to_play) == batch_size
        # ============================================ptree=====================================#
        for i in range(env_nums):
            assert action_num[i] == len(legal_actions_list[i])
        t1 = time.time()
        roots = MCTSPtree.roots(env_nums, legal_actions_list)
        build_time.append(time.time() - t1)
        noises = [
            np.random.dirichlet([policy_config.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                ).astype(np.float32).tolist() for j in range(env_nums)
        ]
        t1 = time.time()
        roots.prepare(policy_config.root_noise_weight, noises, value_prefix_pool, policy_logits_pool, to_play)
        prepare_time.append(time.time() - t1)
        t1 = time.time()
        MCTSPtree(policy_config).search(roots, model, hidden_state_roots, reward_hidden_state_state, to_play)
        search_time.append(time.time() - t1)
        total_time.append(time.time() - t0)
        roots_distributions = roots.get_distributions()
        roots_values = roots.get_values()
        assert len(roots_values) == env_nums
        assert len(roots_values) == env_nums
        for i in range(env_nums):
            assert len(roots_distributions[i]) == action_num[i]

        temperature = [1 for _ in range(env_nums)]
        for i in range(env_nums):
            distributions, value = roots_distributions[i], roots_values[i]
            action_index, visit_count_distribution_entropy = select_action(
                distributions, temperature=temperature[i], deterministic=False
            )
            action = np.where(np.array(action_mask[i]) == 1.0)[0][action_index]
            assert action_index < action_num[i]
            assert action == legal_actions_list[i][action_index]
            print('\n action_index={}, legal_action={}, action={}'.format(action_index, legal_actions_list[i], action))
    return build_time, prepare_time, search_time, total_time


def ctree_func(policy_config, num_simulations):
    """
        Overview:
            Search on the tree of the C++ implementation and record the time spent at different stages.
        Arguments:
            - policy_config: config of game.
            - num_simulations: Number of simulations.
        Returns:
            - build_time: Type builds take time.
            - prepare_time: time for prepare.
            - search_time.
            - total_time.
        """
    batch_size = env_nums = policy_config.batch_size
    action_space_size = policy_config.action_space_size

    build_time = []
    prepare_time = []
    search_time = []
    total_time = []

    for n_s in num_simulations:
        t0 = time.time()
        model = MuZeroModelFake(action_num=action_space_size)
        stack_obs = torch.zeros(
            size=(
                batch_size,
                n_s,
            ), dtype=torch.float
        )
        policy_config.num_simulations = n_s

        network_output = model.initial_inference(stack_obs.float())

        hidden_state_roots = network_output['hidden_state']
        reward_hidden_state_state = network_output['reward_hidden_state']
        pred_values_pool = network_output['value']
        value_prefix_pool = network_output['value_prefix']
        policy_logits_pool = network_output['policy_logits']

        # network output process
        pred_values_pool = inverse_scalar_transform(pred_values_pool,
                                                    policy_config.model.support_scale).detach().cpu().numpy()
        hidden_state_roots = hidden_state_roots.detach().cpu().numpy()
        reward_hidden_state_state = (
            reward_hidden_state_state[0].detach().cpu().numpy(), reward_hidden_state_state[1].detach().cpu().numpy()
        )
        policy_logits_pool = policy_logits_pool.detach().cpu().numpy().tolist()

        action_mask = [[random.randint(0, 1) for _ in range(action_space_size)] for _ in range(env_nums)]
        assert len(action_mask) == batch_size
        assert len(action_mask[0]) == action_space_size

        action_num = [int(np.array(action_mask[i]).sum()) for i in range(env_nums)]
        legal_actions_list = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(env_nums)]
        to_play = [np.random.randint(1, 3) for i in range(env_nums)]
        assert len(to_play) == batch_size
        # ============================================ctree=====================================#
        for i in range(env_nums):
            assert action_num[i] == len(legal_actions_list[i])

        t1 = time.time()
        roots = MCTSCtree.roots(env_nums, legal_actions_list)
        build_time.append(time.time() - t1)
        noises = [
            np.random.dirichlet([policy_config.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                ).astype(np.float32).tolist() for j in range(env_nums)
        ]
        t1 = time.time()
        roots.prepare(policy_config.root_noise_weight, noises, value_prefix_pool, policy_logits_pool, to_play)
        prepare_time.append(time.time() - t1)
        t1 = time.time()
        MCTSCtree(policy_config).search(roots, model, hidden_state_roots, reward_hidden_state_state, to_play)
        search_time.append(time.time() - t1)
        total_time.append(time.time() - t0)
        roots_distributions = roots.get_distributions()
        roots_values = roots.get_values()
        assert len(roots_values) == env_nums
        assert len(roots_values) == env_nums
        for i in range(env_nums):
            assert len(roots_distributions[i]) == action_num[i]

        temperature = [1 for _ in range(env_nums)]
        for i in range(env_nums):
            distributions, value = roots_distributions[i], roots_values[i]
            action_index, visit_count_distribution_entropy = select_action(
                distributions, temperature=temperature[i], deterministic=False
            )
            action = np.where(np.array(action_mask[i]) == 1.0)[0][action_index]
            assert action_index < action_num[i]
            assert action == legal_actions_list[i][action_index]
            print('\n action_index={}, legal_action={}, action={}'.format(action_index, legal_actions_list[i], action))
    return build_time, prepare_time, search_time, total_time


def plot(ctree_time, ptree_time, iters, label):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import pyplot
    plt.style.use('seaborn-whitegrid')
    palette = pyplot.get_cmap('Set1')
    font1 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 18,
    }

    fig = plt.figure(figsize=(20, 10))
    # ctree
    color = palette(0)
    avg = np.mean(ctree_time, axis=0)
    std = np.std(ctree_time, axis=0)
    r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  # 上方差
    r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  # 下方差
    plt.plot(iters, avg, color=color, label="ctree", linewidth=3.0)
    plt.fill_between(iters, r1, r2, color=color, alpha=0.2)

    # ptree
    ptree_time = np.array(ptree_time)
    color = palette(1)
    avg = np.mean(ptree_time, axis=0)
    std = np.std(ptree_time, axis=0)
    r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
    r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
    plt.plot(iters, avg, color=color, label="ptree", linewidth=3.0)
    plt.fill_between(iters, r1, r2, color=color, alpha=0.2)

    plt.legend(loc='lower right', prop=font1)
    plt.title('{}'.format(label))
    plt.xlabel('simulations', fontsize=22)
    plt.ylabel('time', fontsize=22)
    plt.savefig('{}-time.png'.format(label))


if __name__ == "__main__":

    # cProfile.run("ctree_func()", filename="ctree_result.out", sort="cumulative")
    # cProfile.run("ptree_func()", filename="ptree_result.out", sort="cumulative")

    policy_config = EasyDict(
        dict(
            lstm_horizon_len=5,
            model=dict(
                support_scale=300,
                categorical_distribution=True,
            ),
            action_space_size=100,
            num_simulations=100,
            batch_size=512,
            pb_c_base=1,
            pb_c_init=1,
            discount_factor=0.9,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.2,
            dirichlet_alpha=0.3,
            exploration_fraction=1,
            device='cpu',
            value_delta_max=0.01,
        )
    )

    ACTION_SPCAE_SIZE = [16, 50]
    BATCH_SIZE = [8, 64, 512]
    NUM_SIMULATIONS = [i for i in range(20, 200, 20)]

    # ACTION_SPCAE_SIZE = [50]
    # BATCH_SIZE = [512]
    # NUM_SIMULATIONS =  [i for i in range(10, 50, 10)]

    for action_space_size in ACTION_SPCAE_SIZE:
        for batch_size in BATCH_SIZE:
            policy_config.batch_size = batch_size
            policy_config.action_space_size = action_space_size
            ctree_build_time = []
            ctree_prepare_time = []
            ctree_search_time = []
            ptree_build_time = []
            ptree_prepare_time = []
            ptree_search_time = []
            ctree_total_time = []
            ptree_total_time = []
            num_simulations = NUM_SIMULATIONS
            for i in range(3):
                build_time, prepare_time, search_time, total_time = ctree_func(
                    policy_config, num_simulations=num_simulations
                )
                ctree_build_time.append(build_time)
                ctree_prepare_time.append(prepare_time)
                ctree_search_time.append(search_time)
                ctree_total_time.append(total_time)

            for i in range(3):
                build_time, prepare_time, search_time, total_time = ptree_func(
                    policy_config, num_simulations=num_simulations
                )
                ptree_build_time.append(build_time)
                ptree_prepare_time.append(prepare_time)
                ptree_search_time.append(search_time)
                ptree_total_time.append(total_time)
            label = 'action_space_size_{}_batch_size_{}'.format(action_space_size, batch_size)
            plot(ctree_build_time, ptree_build_time, iters=num_simulations, label=label + '_bulid_time')
            plot(ctree_prepare_time, ptree_prepare_time, iters=num_simulations, label=label + '_prepare_time')
            plot(ctree_search_time, ptree_search_time, iters=num_simulations, label=label + '_search_time')
            plot(ctree_total_time, ptree_total_time, iters=num_simulations, label=label + '_total_time')
