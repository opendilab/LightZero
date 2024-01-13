import os
from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch
from tensorboardX import SummaryWriter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from ding.config import compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.torch_utils import to_tensor, to_device, to_ndarray
from ding.worker import BaseLearner
from lzero.worker import MuZeroEvaluator
from lzero.policy import InverseScalarTransform, mz_network_output_unpack

from zoo.classic_control.mountain_car.config.mtcar_muzero_config import main_config, create_config
# from lzero.entry import eval_muzero
import numpy as np

from typing import Optional, Tuple, List

# ==============================================================
# 原空间枚举状态
def create_grid(v_mins: List, v_maxs: List, resolution: int) -> np.ndarray:
    data = list(map(lambda r: np.linspace(*r, resolution), zip(v_mins, v_maxs)))
    grid = np.asarray(np.meshgrid(*data, indexing="ij")).T.reshape(-1, len(v_mins))
    return grid

def get_state_space(env, resolution: int = 25) -> np.ndarray:
    obs_space = env.observation_space
    state_space = create_grid(obs_space.low, obs_space.high, resolution)
    return state_space


def embedding_manifold(state_space, model, return_pis: bool = False, policy_cfg = None) -> Tuple:
    with torch.no_grad():
        network_output = model.initial_inference(state_space)
    latent_state, reward, value, policy_logits = mz_network_output_unpack(network_output)
    inverse_scalar_transform_handler = InverseScalarTransform(
        policy_cfg.model.support_scale,
        policy_cfg.device,
        policy_cfg.model.categorical_distribution)
    value_real = inverse_scalar_transform_handler(value)

    if return_pis:
        return to_ndarray(latent_state.cpu()), to_ndarray(value_real.cpu()), to_ndarray(policy_logits.cpu())
    
    return to_ndarray(latent_state.cpu()), to_ndarray(value_real.cpu())

# ==============================================================
# PCA
def embedding_PCA(latent_states: np.ndarray, standardize: bool = False):   
    x = latent_states
    if standardize:
        x = (x - x.mean(axis=0)) / x.std(axis=0)
    
    # Perform PCA on latent dimensions
    pca = PCA(n_components=x.shape[-1])
    pca.fit(x)
    spcs = pca.fit_transform(x)
    
    # Create barchart
    ns = list(range(x.shape[-1]))
    var = pca.explained_variance_ratio_
    
    bar = plt.bar(ns, var)
    
    plt.title(f"PCA on latent-states (standardize={standardize})")
    plt.ylabel("Explained Variance Ratio")
    plt.xlabel("Principal Component")
    
    
    for i in range(len(var)):
        plt.annotate(f'{var[i]:.3f}', xy=(ns[i],var[i]), ha='center', va='bottom')

    plt.show()
    
    # Create violinplot
    plt.violinplot(spcs)
    plt.xticks(range(1, x.shape[-1]+1), range(1, x.shape[-1]+1))
    
    plt.title(f"Projected values distribution (standardize={standardize})")
    plt.ylabel("PC values")
    plt.xlabel("Principal Component")
    
    plt.show()
    
    return pca

# ==============================================================
# 状态空间可视化
def to_grid(x: np.ndarray, delta: int) -> np.ndarray:
    return x.reshape(delta, delta)


def simple_PC_value_contour(pc_1: np.ndarray, pc_2: np.ndarray, z: np.ndarray) -> None:
    # 画 PCA 后的 latent state
    plt.scatter(pc_1, pc_2, c=z, alpha=0.5, s=5, cmap='rainbow')

    cbar = plt.colorbar()
    cbar.set_label(r'$V_\theta(o_t)$')

    plt.title("Value Contour MuZero PC-Space")
    plt.ylabel(r"First PCA component $h_\theta(o_t)$")
    plt.xlabel(r"Second PCA component $h_\theta(o_t)$")


def simple_MC_value_contour(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    # 画原始的 state
    # Simple Example Figure for a 2-d env
    plt.title("Value Contour MuZero MountainCar")
    plt.ylabel("Velocity")
    plt.xlabel("Position")

    plt.contourf(x, y, z, levels=100, cmap='rainbow')

    cbar = plt.colorbar()
    cbar.set_label(r'$V_\theta(o_t)$')
    
# ==============================================================
# 游戏轨迹dynamics处理
def get_latent_trajectory(embeddings: torch.Tensor, actions: torch.Tensor, model) -> np.ndarray:
    latent_state = embeddings[0].unsqueeze(0)
    
    latent_states = list()
    latent_states.append(to_ndarray(latent_state.cpu()))
    with torch.no_grad():
        for i in range(len(actions)):
            
            network_output = model.recurrent_inference(latent_state, actions[i].unsqueeze(0))    # 这里action注意
            latent_state, reward, value, policy_logits = mz_network_output_unpack(network_output)
        
            # memory = latent_state
            latent_states.append(to_ndarray(latent_state.cpu()))

    stacked = np.concatenate(latent_states)
    return stacked

# ==============================================================
# 3D轨迹可视化
def generate_3d_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray, clim=None):
    return go.Surface(
        x=x, y=y, z=z,
        opacity=1, 
        surfacecolor=colors,
        colorscale='Viridis',
        cmin=colors.min() if clim is None else clim[0],
        cmax=colors.max() if clim is None else clim[1],
        colorbar=dict(title=dict(text='V',side='top'), thickness=50, tickmode='array')
    )

def generate_3d_trajectory(x: np.ndarray, y: np.ndarray, z: np.ndarray, color: str):
    return go.Scatter3d(
        x=x + np.random.rand()*0.01,
        y=y + np.random.rand()*0.01,
        z=z + np.random.rand()*0.01,
        mode='lines+markers',
        marker=dict(
            size=3,
            symbol='x',
            color=color,
            opacity=1
        ),
        line=dict(
            color=color,
            width=20
        )
    )

def generate_3d_valuefield(x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray, clim=None):
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=colors,
            colorscale='Viridis',
            cmin=colors.min() if clim is None else clim[0],
            cmax=colors.max() if clim is None else clim[1],
            opacity=1,
            colorbar=dict(title=dict(text='V',side='top'), thickness=50, tickmode='array')
        ),
    )
# ==============================================================

def vis_muzero(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        num_episodes_each_seed: int = 1,
        print_seed_details: int = False,
        # return_trajectory: bool = False,
) -> 'Policy':  # noqa
    """
    Overview:
        The eval entry for MCTS+RL algorithms, including MuZero, EfficientZero, Sampled EfficientZero.
    Arguments:
        - input_cfg (:obj:`Tuple[dict, dict]`): Config in dict type.
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - model_path (:obj:`Optional[str]`): The pretrained model path, which should
            point to the ckpt file of the pretrained model, and an absolute path is recommended.
            In LightZero, the path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    cfg, create_cfg = input_cfg
    assert create_cfg.policy.type in ['efficientzero', 'muzero', 'stochastic_muzero', 'gumbel_muzero', 'sampled_efficientzero'], \
        "LightZero now only support the following algo.: 'efficientzero', 'muzero', 'stochastic_muzero', 'gumbel_muzero', 'sampled_efficientzero'"

    if cfg.policy.cuda and torch.cuda.is_available():
        cfg.policy.device = 'cuda'
    else:
        cfg.policy.device = 'cpu'

    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # load pretrained model
    if model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # ==============================================================
    # MCTS+RL algorithms related core code
    # ==============================================================
    policy_config = cfg.policy
    evaluator = MuZeroEvaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=policy_config
    )

    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    # 只进行一次eval
    # ==============================================================
    # eval trained model
    # ==============================================================
    # returns = []
    # trajectorys = []
    # for i in range(num_episodes_each_seed):
    stop_flag, episode_info = evaluator.eval(learner.save_checkpoint, learner.train_iter, return_trajectory=True)
    trajectorys = episode_info['trajectory']
    returns = episode_info['eval_episode_return']

    returns = np.array(returns)

    if print_seed_details:
        print("=" * 20)
        print(f'In seed {seed}, returns: {returns}')
        if cfg.policy.env_type == 'board_games':
            print(
                f'win rate: {len(np.where(returns == 1.)[0]) / num_episodes_each_seed}, draw rate: {len(np.where(returns == 0.)[0]) / num_episodes_each_seed}, lose rate: {len(np.where(returns == -1.)[0]) / num_episodes_each_seed}'
            )
        print("=" * 20)
    
    # 原空间枚举
    delta = 250
    state_space = get_state_space(evaluator_env, delta)
    state_space_tensor = to_device(to_tensor(state_space), policy_config.device)
    latent_state_space, v_state_space = embedding_manifold(state_space_tensor, policy._model, policy_cfg=policy_config)
    print(state_space.shape, latent_state_space.shape)
    # pca
    pca = embedding_PCA(latent_state_space, False)
    pca_norm = embedding_PCA(latent_state_space, True)
    pca_latent_state_space = pca.transform(latent_state_space)
    # 原空间/PCA latent state可视化
    simple_PC_value_contour(pca_latent_state_space[:, 0], pca_latent_state_space[:, 1], v_state_space)
    plt.show()
    simple_MC_value_contour(to_grid(state_space[:,0], delta), to_grid(state_space[:,1], delta), to_grid(v_state_space, delta))
    plt.show()
    # 获取eval的轨迹
    #   1 轨迹加载
    real_state = np.array(trajectorys[0].obs_segment)
    real_state_tensor = to_device(to_tensor(real_state), state_space_tensor.device)
    actions = np.array(trajectorys[0].action_segment)
    actions_tensor = to_device(to_tensor(actions).unsqueeze(1), state_space_tensor.device)
    with torch.no_grad():
        network_output = policy._model.initial_inference(real_state_tensor)
    latent_state_represent_tensor, reward, v_trajectorys_tensor, policy_logits = mz_network_output_unpack(network_output) 
    latent_state_represent = to_ndarray(latent_state_represent_tensor.cpu())
    v_trajectorys = to_ndarray(v_trajectorys_tensor.cpu())

    #   2 获取latent state轨迹
    latent_state_dynamics = get_latent_trajectory(latent_state_represent_tensor, actions_tensor, policy._model)

    #   3 Project to PC-space
    pc_embedding_trajectory = pca.transform(latent_state_represent.reshape(len(latent_state_represent), -1))
    pc_dynamics_trajectory = pca.transform(latent_state_dynamics.reshape(len(latent_state_dynamics), -1))

    # 观察分布
    #   1 latent state轨迹分布
    plt.violinplot(latent_state_represent.reshape(len(latent_state_represent), -1), np.arange(1, latent_state_space.shape[-1] + 1))
    plt.violinplot(latent_state_dynamics.reshape(len(latent_state_dynamics), -1), np.arange(1, latent_state_space.shape[-1] + 1))

    plt.scatter([], [], label='embedding')
    plt.scatter([], [], label='dynamics')

    plt.title("Value Distributions within latent-space")

    plt.ylabel("Values")
    plt.xlabel("Latent Dimension")
    plt.xticks(range(1, latent_state_space.shape[-1] + 1), [f'dim {i}' for i in range(1, latent_state_space.shape[-1] + 1)], rotation=45)

    plt.legend()
    plt.show()
    #   2 PCA后的latent state轨迹分布
    plt.violinplot(pc_embedding_trajectory, np.arange(1, latent_state_space.shape[-1] + 1))
    plt.violinplot(pc_dynamics_trajectory, np.arange(1, latent_state_space.shape[-1] + 1))

    plt.scatter([], [], label='embedding')
    plt.scatter([], [], label='dynamics')

    plt.title("Value Distributions within latent PC-space")

    plt.ylabel("Values")
    plt.xlabel("Latent Dimension")
    plt.xticks(range(1, latent_state_space.shape[-1] + 1), [f'dim {i}' for i in range(1, latent_state_space.shape[-1] + 1)], rotation=45)

    plt.legend()
    plt.show()

    # 3D轨迹可视化
    x = 3
    dynamics_trajectory =  generate_3d_trajectory(
        pc_dynamics_trajectory[:, 0].ravel(),
        pc_dynamics_trajectory[:, 1].ravel(), 
        pc_dynamics_trajectory[:, 2].ravel(), 'grey')

    embedding_trajectory = generate_3d_trajectory(
        pc_embedding_trajectory[:, 0].ravel(), 
        pc_embedding_trajectory[:, 1].ravel(), 
        pc_embedding_trajectory[:, 2].ravel(), 'black')

    surface = generate_3d_valuefield(pca_latent_state_space[:,0], pca_latent_state_space[:,1], pca_latent_state_space[:,2], v_state_space)

    fig = go.Figure(data=[embedding_trajectory, dynamics_trajectory, surface])

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    fig.show()

    



if __name__ == "__main__":
    """
    Entry point for the evaluation of the MuZero model on the CartPole environment. 

    Variables:
        - model_path (:obj:`Optional[str]`): The pretrained model path, which should point to the ckpt file of the 
        pretrained model. An absolute path is recommended. In LightZero, the path is usually something like 
        ``exp_name/ckpt/ckpt_best.pth.tar``.
        - returns_mean_seeds (:obj:`List[float]`): List to store the mean returns for each seed.
        - returns_seeds (:obj:`List[float]`): List to store the returns for each seed.
        - seeds (:obj:`List[int]`): List of seeds for the environment.
        - num_episodes_each_seed (:obj:`int`): Number of episodes to run for each seed.
        - total_test_episodes (:obj:`int`): Total number of test episodes, computed as the product of the number of 
        seeds and the number of episodes per seed.
    """
    # model_path = "./ckpt/ckpt_best.pth.tar"
    model_path = "/home/nighoodRen/lz_result/debug/mountain_car_muzero_ns25_upc100_rr0_seed0_240110_201124/ckpt/ckpt_best.pth.tar"
    returns_mean_seeds = []
    returns_seeds = []
    seed = 0
    num_episodes_each_seed = 1
    total_test_episodes = num_episodes_each_seed
    create_config.env_manager.type = 'base'  # Visualization requires the 'type' to be set as base
    main_config.env.evaluator_env_num = 1  # Visualization requires the 'env_num' to be set as 1
    main_config.env.n_evaluator_episode = total_test_episodes
    main_config.env.replay_path = 'lz_result/video/mtcar_mz'
    main_config.exp_name = f'lz_result/eval/muzero_eval_ls{main_config.policy.model.latent_state_dim}'

    # for seed in seeds:
    """
    - returns_mean (:obj:`float`): The mean return of the evaluation.
    - returns (:obj:`List[float]`): The returns of the evaluation.
    """
    returns_mean, returns = vis_muzero(
        [main_config, create_config],
        seed=seed,
        num_episodes_each_seed=num_episodes_each_seed,
        print_seed_details=False,
        model_path=model_path,
        # return_trajectory=True,
    )

    returns_mean = np.array(returns_mean)
    returns = np.array(returns)

    # Print evaluation results
    print("=" * 20)
    print(f"We evaluated a total of 1 seeds. For each seed, we evaluated {num_episodes_each_seed} episode(s).")
    print(f"For seed {seed}, the mean return is {returns_mean}, and the returns are {returns}.")
    print("Across all seeds, the mean reward is:", returns_mean.mean())
    print("=" * 20)

    # 可视化

