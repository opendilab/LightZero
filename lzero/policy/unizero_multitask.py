import copy
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY

from lzero.entry.utils import initialize_zeros_batch
from lzero.mcts import UniZeroMCTSCtree as MCTSCtree
from lzero.model import ImageTransforms
from lzero.policy import prepare_obs_stack_for_unizero
from lzero.policy import scalar_transform, InverseScalarTransform, phi_transform, \
    DiscreteSupport, to_torch_float_tensor, mz_network_output_unpack, select_action, prepare_obs
from lzero.policy.unizero import UniZeroPolicy, scale_module_weights_vectorized
from .utils import configure_optimizers_nanogpt
import sys

# Please replace the path with the actual location of your LibMTL library.
sys.path.append('/path/to/your/LibMTL')

from LibMTL.weighting.MoCo_unizero import MoCo as GradCorrect
from LibMTL.weighting.moco_fast_mem_eff import FastMoCoMemEff as FastMoCo
from LibMTL.weighting.moco_fast_mem_eff import MoCoCfg

import torch.distributed as dist



def generate_task_loss_dict(multi_task_losses: List[Union[torch.Tensor, float]], task_name_template: str, task_id: int) -> Dict[str, float]:
    """
    Overview:
        Generates a dictionary for the losses of each task.
    Arguments:
        - multi_task_losses (:obj:`List[Union[torch.Tensor, float]]`): A list containing the loss for each task.
        - task_name_template (:obj:`str`): The template for the task name, e.g., 'obs_loss_task{}'.
        - task_id (:obj:`int`): The starting ID of the tasks.
    Returns:
        - task_loss_dict (:obj:`Dict[str, float]`): A dictionary where keys are formatted task names and values are the corresponding losses.
    """
    task_loss_dict = {}
    for task_idx, task_loss in enumerate(multi_task_losses):
        task_name = task_name_template.format(task_idx + task_id)
        try:
            # Get the scalar value of the loss if it's a tensor.
            task_loss_dict[task_name] = task_loss.item() if hasattr(task_loss, 'item') else task_loss
        except Exception as e:
            task_loss_dict[task_name] = task_loss
    return task_loss_dict



class WrappedModel:
    """
    Overview:
        A wrapper class for the world model to conveniently access its parameters and zero its gradients.
        This version wraps the entire world model.
    """
    def __init__(self, world_model: torch.nn.Module):
        """
        Arguments:
            - world_model (:obj:`torch.nn.Module`): The world model instance.
        """
        self.world_model = world_model

    def parameters(self) -> iter:
        """
        Overview:
            Returns an iterator over the parameters of the entire world model.
        """
        return self.world_model.parameters()

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Overview:
            Sets the gradients of all world model parameters to zero.
        Arguments:
            - set_to_none (:obj:`bool`): Whether to set gradients to None instead of zero.
        """
        self.world_model.zero_grad(set_to_none=set_to_none)


class WrappedModelV2:
    """
    Overview:
        A wrapper for specific components of the world model.
        This version is designed to group parameters that are considered "shared"
        across tasks for gradient correction methods like MoCo, excluding the prediction heads.
    """
    def __init__(self, tokenizer: torch.nn.Module, transformer: torch.nn.Module, pos_emb: torch.nn.Module, task_emb: torch.nn.Module, act_embedding_table: torch.nn.Module):
        """
        Arguments:
            - tokenizer (:obj:`torch.nn.Module`): The tokenizer module.
            - transformer (:obj:`torch.nn.Module`): The transformer backbone.
            - pos_emb (:obj:`torch.nn.Module`): The positional embedding module.
            - task_emb (:obj:`torch.nn.Module`): The task embedding module.
            - act_embedding_table (:obj:`torch.nn.Module`): The action embedding table.
        """
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.pos_emb = pos_emb
        self.task_emb = task_emb
        self.act_embedding_table = act_embedding_table

    def parameters(self) -> iter:
        """
        Overview:
            Returns an iterator over the parameters of the wrapped components (tokenizer, transformer, embeddings).
            These are typically the shared parts of the model whose gradients need to be managed for multi-task learning.
        """
        return (list(self.tokenizer.parameters()) +
                list(self.transformer.parameters()) +
                list(self.pos_emb.parameters()) +
                # list(self.task_emb.parameters()) + # TODO: Decide whether to include task embeddings in shared parameters.
                list(self.act_embedding_table.parameters()))

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Overview:
            Sets the gradients of all wrapped components to zero.
        Arguments:
            - set_to_none (:obj:`bool`): Whether to set gradients to None instead of zero.
        """
        self.tokenizer.zero_grad(set_to_none=set_to_none)
        self.transformer.zero_grad(set_to_none=set_to_none)
        self.pos_emb.zero_grad(set_to_none=set_to_none)
        # self.task_emb.zero_grad(set_to_none=set_to_none)  # TODO: Match the decision made in the parameters() method.
        self.act_embedding_table.zero_grad(set_to_none=set_to_none)


class WrappedModelV3:
    """
    Overview:
        An alternative wrapper for world model components.
        This version excludes the tokenizer from the shared parameters, focusing gradient correction
        on the transformer and embedding layers.
    """
    def __init__(self, transformer: torch.nn.Module, pos_emb: torch.nn.Module, task_emb: torch.nn.Module, act_embedding_table: torch.nn.Module):
        """
        Arguments:
            - transformer (:obj:`torch.nn.Module`): The transformer backbone.
            - pos_emb (:obj:`torch.nn.Module`): The positional embedding module.
            - task_emb (:obj:`torch.nn.Module`): The task embedding module.
            - act_embedding_table (:obj:`torch.nn.Module`): The action embedding table.
        """
        self.transformer = transformer
        self.pos_emb = pos_emb
        self.task_emb = task_emb
        self.act_embedding_table = act_embedding_table

    def parameters(self) -> iter:
        """
        Overview:
            Returns an iterator over the parameters of the transformer and various embedding layers.
        """
        return (list(self.transformer.parameters()) +
                list(self.pos_emb.parameters()) +
                list(self.task_emb.parameters()) +
                list(self.act_embedding_table.parameters()))

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Overview:
            Sets the gradients of the wrapped components to zero.
        Arguments:
            - set_to_none (:obj:`bool`): Whether to set gradients to None instead of zero.
        """
        self.transformer.zero_grad(set_to_none=set_to_none)
        self.pos_emb.zero_grad(set_to_none=set_to_none)
        self.task_emb.zero_grad(set_to_none=set_to_none)
        self.act_embedding_table.zero_grad(set_to_none=set_to_none)



def configure_optimizer_unizero(model, learning_rate, weight_decay, device_type, betas):
    """
    Configure optimizer with differentiated learning rates for UniZero model.
    (Corrected version ensuring parameter groups are mutually exclusive)
    """
    # 1. Create empty parameter lists for grouping
    transformer_params = []
    tokenizer_params = []
    head_params = []

    # 2. Iterate through all trainable parameters, using if/elif/else structure to ensure each parameter is assigned to only one group
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'transformer' in name:
            transformer_params.append(param)
        elif 'tokenizer' in name:
            tokenizer_params.append(param)
        else:
            head_params.append(param)

    # 3. Set different optimizer parameters for each group
    #    We still use AdamW here, but with more reasonable learning rate settings
    optim_groups = [
        {
            'params': transformer_params,
            'lr': learning_rate,  # 1e-4
            'weight_decay': weight_decay
        },
        {
            'params': tokenizer_params,
            'lr': learning_rate,  # Tokenizer uses base learning rate, e.g., 1e-4
            'weight_decay': weight_decay
        },
        {
            'params': head_params,
            'lr': learning_rate,  # Heads also use base learning rate, e.g., 1e-4
            'weight_decay': weight_decay

        }
    ]

    print("--- Optimizer Groups ---")
    # Print parameter count for each group for debugging
    print(f"Transformer params: {len(transformer_params)}")
    print(f"Tokenizer params: {len(tokenizer_params)}")
    print(f"Head params: {len(head_params)}")
    print(f"Transformer LR: {learning_rate}")
    print(f"Tokenizer/Heads LR: {learning_rate}")

    optimizer = torch.optim.AdamW(optim_groups, betas=betas)
    return optimizer

@POLICY_REGISTRY.register('unizero_multitask')
class UniZeroMTPolicy(UniZeroPolicy):
    """
    Overview:
        The policy class for multi-task UniZero, an official implementation for the paper "UniZero: Generalized and Efficient Planning
        with Scalable Latent World Models". UniZero aims to enhance the planning capabilities of reinforcement learning agents
        by addressing the limitations of MuZero-style algorithms, particularly in environments requiring the
        capture of long-term dependencies. More details can be found at: https://arxiv.org/abs/2406.10667.
    """

    # The default_config for UniZero multi-task policy.
    config = dict(
        type='unizero_multitask',
        model=dict(
            # (str) The model type. For 1-dimensional vector obs, we use mlp model. For the image obs, we use conv model.
            model_type='conv',  # options={'mlp', 'conv'}
            # (bool) If True, the action space of the environment is continuous, otherwise discrete.
            continuous_action_space=False,
            # (tuple) The obs shape.
            observation_shape=(3, 64, 64),
            # (bool) Whether to use the self-supervised learning loss.
            self_supervised_learning_loss=True,
            # (bool) Whether to use discrete support to represent categorical distribution for value/reward/value_prefix.
            categorical_distribution=True,
            # (int) The image channel in image observation.
            image_channel=3,
            # (int) The number of frames to stack together.
            frame_stack_num=1,
            # (int) The number of res blocks in MuZero model.
            num_res_blocks=1,
            # (int) The number of channels of hidden states in MuZero model.
            num_channels=64,
            # (int) The scale of supports used in categorical distribution.
            # This variable is only effective when ``categorical_distribution=True``.
            support_scale=50,
            # (bool) whether to learn bias in the last linear layer in value and policy head.
            bias=True,
            # (bool) whether to use res connection in dynamics.
            res_connection_in_dynamics=True,
            # (str) The type of normalization in MuZero model. Options are ['BN', 'LN']. Default to 'BN'.
            norm_type='LN',  # NOTE: LayerNorm is used in the transformer-based world model.
            # (bool) Whether to analyze simulation normalization.
            analysis_sim_norm=False,
            # (int) The save interval of the model.
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=10000, ), ), ),
            world_model_cfg=dict(
                # (int) The number of tokens per block.
                tokens_per_block=2,
                # (int) The maximum number of blocks.
                max_blocks=10,
                # (int) The maximum number of tokens, calculated as tokens per block multiplied by max blocks.
                max_tokens=2 * 10,
                # (int) The context length, usually calculated as twice the number of some base unit.
                context_length=2 * 4,
                # (bool) Whether to use GRU gating mechanism.
                gru_gating=False,
                # (str) The device to be used for computation, e.g., 'cpu' or 'cuda'.
                device='cpu',
                # (bool) Whether to analyze simulation normalization.
                analysis_sim_norm=False,
                # (bool) Whether to analyze dormant ratio.
                analysis_dormant_ratio=False,
                # (int) The shape of the action space.
                action_space_size=6,
                # (int) The size of the group, related to simulation normalization.
                group_size=8,  # NOTE: for sim_norm
                # (str) The type of attention mechanism used. Options could be ['causal'].
                attention='causal',
                # (int) The number of layers in the model.
                num_layers=2,
                # (int) The number of attention heads.
                num_heads=8,
                # (int) The dimension of the embedding.
                embed_dim=768,
                # (float) The dropout probability for the embedding layer.
                embed_pdrop=0.1,
                # (float) The dropout probability for the residual connections.
                resid_pdrop=0.1,
                # (float) The dropout probability for the attention mechanism.
                attn_pdrop=0.1,
                # (int) The size of the support set for value and reward heads.
                support_size=101,
                # (int) The maximum size of the cache.
                max_cache_size=5000,
                # (int) The number of environments.
                env_num=8,
                # (float) The weight of the latent reconstruction loss.
                latent_recon_loss_weight=0.,
                # (float) The weight of the perceptual loss.
                perceptual_loss_weight=0.,
                # (float) The weight of the policy entropy.
                policy_entropy_weight=1e-4,
                # (str) The type of loss for predicting latent variables. Options could be ['group_kl', 'mse'].
                predict_latent_loss_type='group_kl',
                # (str) The type of observation. Options are ['image', 'vector'].
                obs_type='image',
                # (float) The discount factor for future rewards.
                gamma=1,
                # (bool) Whether to analyze dormant ratio, average_weight_magnitude of net, effective_rank of latent.
                analysis_dormant_ratio_weight_rank=False,
                # (float) The threshold for a dormant neuron.
                dormant_threshold=0.01,

            ),
        ),
        # ****** common ******
        # (bool) whether to use rnd model.
        use_rnd_model=False,
        # (bool) Whether to use multi-gpu training.
        multi_gpu=True,
        # (bool) Whether to enable the sampled-based algorithm (e.g. Sampled EfficientZero)
        # this variable is used in ``collector``.
        sampled_algo=False,
        # (bool) Whether to enable the gumbel-based algorithm (e.g. Gumbel Muzero)
        gumbel_algo=False,
        # (bool) Whether to use C++ MCTS in policy. If False, use Python implementation.
        mcts_ctree=True,
        # (bool) Whether to use cuda for network.
        cuda=True,
        # (int) The number of environments used in collecting data.
        collector_env_num=8,
        # (int) The number of environments used in evaluating policy.
        evaluator_env_num=3,
        # (str) The type of environment. Options are ['not_board_games', 'board_games'].
        env_type='not_board_games',
        # (str) The type of action space. Options are ['fixed_action_space', 'varied_action_space'].
        action_type='fixed_action_space',
        # (str) The type of battle mode. Options are ['play_with_bot_mode', 'self_play_mode'].
        battle_mode='play_with_bot_mode',
        # (bool) Whether to monitor extra statistics in tensorboard.
        monitor_extra_statistics=True,
        # (int) The transition number of one ``GameSegment``.
        game_segment_length=400,
        # (bool) Whether to analyze simulation normalization.
        analysis_sim_norm=False,
        # (bool) Whether to use the pure policy to collect data.
        collect_with_pure_policy=False,
        # (int) The evaluation frequency.
        eval_freq=int(5e3),
        # (str) The sample type. Options are ['episode', 'transition'].
        sample_type='transition',

        # ****** observation ******
        # (bool) Whether to transform image to string to save memory.
        transform2string=False,
        # (bool) Whether to use gray scale image.
        gray_scale=False,
        # (bool) Whether to use data augmentation.
        use_augmentation=False,
        # (list) The style of augmentation.
        augmentation=['shift', 'intensity'],

        # ******* learn ******
        # (bool) Whether to ignore the done flag in the training data. Typically, this value is set to False.
        # However, for some environments with a fixed episode length, to ensure the accuracy of Q-value calculations,
        # we should set it to True to avoid the influence of the done flag.
        ignore_done=False,
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        # For different env, we have different episode_length,
        # we usually set update_per_collect = collector_env_num * episode_length / batch_size * reuse_factor.
        # If we set update_per_collect=None, we will set update_per_collect = collected_transitions_num * cfg.policy.replay_ratio automatically.
        update_per_collect=None,
        # (float) The ratio of the collected data used for training. Only effective when ``update_per_collect`` is not None.
        replay_ratio=0.25,
        # (int) Minibatch size for one gradient descent.
        batch_size=256,
        # (str) Optimizer for training policy network.
        optim_type='AdamW',
        # (float) Learning rate for training policy network. Initial lr for manually decay schedule.
        learning_rate=0.0001,
        # ==================== Norm Monitoring Frequency ====================
        # How often (in training iteration steps) to monitor model parameter norms. Set to 0 to disable.
        monitor_norm_freq=5000,
        # ====================================================================
        # (int) Frequency of hard target network update.
        target_update_freq=100,
        # (int) Frequency of soft target network update.
        target_update_theta=0.05,
        # (int) Frequency of target network update.
        target_update_freq_for_intrinsic_reward=1000,
        # (float) Weight decay for training policy network.
        weight_decay=1e-4,
        # (float) One-order Momentum in optimizer, which stabilizes the training process (gradient direction).
        momentum=0.9,
        # (float) The maximum constraint value of gradient norm clipping.
        grad_clip_value=5,
        # (int) The number of episodes in each collecting stage when use muzero_collector.
        n_episode=8,
        # (int) The number of num_segments in each collecting stage when use muzero_segment_collector.
        num_segments=8,
        # # (int) the number of simulations in MCTS for renalyze.
        num_simulations=50,
        # (int) The number of simulations in MCTS for the collect phase.
        collect_num_simulations=25,
        # (int) The number of simulations in MCTS for the eval phase.
        eval_num_simulations=50,
        # (float) Discount factor (gamma) for returns.
        discount_factor=0.997,
        # (int) The number of steps for calculating target q_value.
        td_steps=5,
        # (int) The number of unroll steps in dynamics network.
        num_unroll_steps=10,
        # (float) The weight of reward loss.
        reward_loss_weight=1,
        # (float) The weight of value loss.
        value_loss_weight=0.25,
        # (float) The weight of policy loss.
        policy_loss_weight=1,
        # (float) The weight of ssl (self-supervised learning) loss.
        ssl_loss_weight=0,
        cos_lr_scheduler=False,
        piecewise_decay_lr_scheduler=False,
        # (bool) Whether to use piecewise constant learning rate decay.
        # i.e. lr: 0.2 -> 0.02 -> 0.002
        lr_piecewise_constant_decay=False,
        # (int) The number of final training iterations to control lr decay, which is only used for manually decay.
        threshold_training_steps_for_final_lr=int(5e4),
        # (bool) Whether to use manually decayed temperature.
        manual_temperature_decay=False,
        # (int) The number of final training iterations to control temperature, which is only used for manually decay.
        threshold_training_steps_for_final_temperature=int(1e5),
        # (float) The fixed temperature value for MCTS action selection, which is used to control the exploration.
        # The larger the value, the more exploration. This value is only used when manual_temperature_decay=False.
        fixed_temperature_value=0.25,
        # (bool) Whether to use the true chance in MCTS in some environments with stochastic dynamics, such as 2048.
        use_ture_chance_label_in_chance_encoder=False,

        # ****** Priority ******
        # (bool) Whether to use priority when sampling training data from the buffer.
        use_priority=False,
        # (float) The degree of prioritization to use. A value of 0 means no prioritization,
        # while a value of 1 means full prioritization.
        priority_prob_alpha=0.6,
        # (float) The degree of correction to use. A value of 0 means no correction,
        # while a value of 1 means full correction.
        priority_prob_beta=0.4,
        # (int) The initial Env Steps for training.
        train_start_after_envsteps=int(0),

        # ****** UCB ******
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of search tree.
        root_dirichlet_alpha=0.3,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,

        # ****** Explore by random collect ******
        # (int) The number of episodes to collect data randomly before training.
        random_collect_episode_num=0,

        # ****** Explore by eps greedy ******
        eps=dict(
            # (bool) Whether to use eps greedy exploration in collecting data.
            eps_greedy_exploration_in_collect=False,
            # (str) The type of decaying epsilon. Options are 'linear', 'exp'.
            type='linear',
            # (float) The start value of eps.
            start=1.,
            # (float) The end value of eps.
            end=0.05,
            # (int) The decay steps from start to end eps.
            decay=int(1e5),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm's default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): A tuple containing the model name and a list of import paths.
                - model_type (:obj:`str`): The model type used in this algorithm, registered in ModelRegistry.
                - import_names (:obj:`List[str]`): The list of model class paths used in this algorithm.
        .. note::
            Users can define and use customized network models, but they must adhere to the same interface definition
            as indicated by the import_names path. For multi-task UniZero, this is ``lzero.model.unizero_model_multitask.UniZeroMTModel``.
        """
        # NOTE: This specifies the default multi-task model.
        return 'UniZeroMTModel', ['lzero.model.unizero_model_multitask']

    # ==================== Model Norm Monitoring Function ====================
    def _monitor_model_norms(self) -> Dict[str, float]:
        """
        Overview:
            Calculate and return parameter matrix norms for key model components (Encoder, Transformer, Heads).
            This function should be called within a torch.no_grad() context for efficiency.
        Returns:
            - norm_metrics (:obj:`Dict[str, float]`): Dictionary containing all norm metrics for logging.
        """
        world_model = self._learn_model.world_model
        norm_metrics = {}

        # Define module groups to monitor
        module_groups = {
            'encoder': world_model.tokenizer.encoder,
            'transformer': world_model.transformer,
            'head_value': world_model.head_value_multi_task,  # Note: multi-task uses head_value (plural)
            'head_reward': world_model.head_rewards_multi_task,
            'head_policy': world_model.head_policy_multi_task,  # Note: multi-task uses head_policies (plural)
        }

        for group_name, group_module in module_groups.items():
            # Handle ModuleList (for multi-task heads)
            if isinstance(group_module, torch.nn.ModuleList):
                for task_idx, task_module in enumerate(group_module):
                    total_norm_sq = 0.0
                    for param_name, param in task_module.named_parameters():
                        if param.requires_grad:
                            param_norm = param.data.norm(2).item()
                            log_name = f'norm/{group_name}_task{task_idx}/{param_name.replace(".", "/")}'
                            norm_metrics[log_name] = param_norm
                            total_norm_sq += param_norm ** 2
                    total_group_norm = np.sqrt(total_norm_sq)
                    norm_metrics[f'norm/{group_name}_task{task_idx}/_total_norm'] = total_group_norm
            else:
                # Handle single module
                total_norm_sq = 0.0
                for param_name, param in group_module.named_parameters():
                    if param.requires_grad:
                        param_norm = param.data.norm(2).item()
                        log_name = f'norm/{group_name}/{param_name.replace(".", "/")}'
                        norm_metrics[log_name] = param_norm
                        total_norm_sq += param_norm ** 2
                total_group_norm = np.sqrt(total_norm_sq)
                norm_metrics[f'norm/{group_name}/_total_norm'] = total_group_norm

        return norm_metrics

    def _monitor_gradient_norms(self) -> Dict[str, float]:
        """
        Overview:
            Calculate and return gradient norms for key model components.
            This function should be called after gradient computation and before parameter updates.
        Returns:
            - grad_metrics (:obj:`Dict[str, float]`): Dictionary containing all gradient norm metrics for logging.
        """
        world_model = self._learn_model.world_model
        grad_metrics = {}

        # Define module groups to monitor
        module_groups = {
            'encoder': world_model.tokenizer.encoder,
            'transformer': world_model.transformer,
            'head_value': world_model.head_value_multi_task,
            'head_reward': world_model.head_rewards_multi_task,
            'head_policy': world_model.head_policy_multi_task,
        }

        for group_name, group_module in module_groups.items():
            # Handle ModuleList (for multi-task heads)
            if isinstance(group_module, torch.nn.ModuleList):
                for task_idx, task_module in enumerate(group_module):
                    total_grad_norm_sq = 0.0
                    num_params_with_grad = 0
                    for param_name, param in task_module.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_norm = param.grad.data.norm(2).item()
                            log_name = f'grad/{group_name}_task{task_idx}/{param_name.replace(".", "/")}'
                            grad_metrics[log_name] = grad_norm
                            total_grad_norm_sq += grad_norm ** 2
                            num_params_with_grad += 1
                    if num_params_with_grad > 0:
                        total_group_grad_norm = np.sqrt(total_grad_norm_sq)
                        grad_metrics[f'grad/{group_name}_task{task_idx}/_total_norm'] = total_group_grad_norm
                    else:
                        grad_metrics[f'grad/{group_name}_task{task_idx}/_total_norm'] = 0.0
            else:
                # Handle single module
                total_grad_norm_sq = 0.0
                num_params_with_grad = 0
                for param_name, param in group_module.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        log_name = f'grad/{group_name}/{param_name.replace(".", "/")}'
                        grad_metrics[log_name] = grad_norm
                        total_grad_norm_sq += grad_norm ** 2
                        num_params_with_grad += 1
                if num_params_with_grad > 0:
                    total_group_grad_norm = np.sqrt(total_grad_norm_sq)
                    grad_metrics[f'grad/{group_name}/_total_norm'] = total_group_grad_norm
                else:
                    grad_metrics[f'grad/{group_name}/_total_norm'] = 0.0

        return grad_metrics
    # =================================================================

    def _init_learn(self) -> None:
        """
        Overview:
            Initializes the learn mode. This method is called by ``self.__init__``.
            It sets up the learn model, optimizer, target model, and other utilities required for training.
        """
        if self._cfg.optim_type == 'SGD':
            # Configure SGD optimizer
            self._optimizer_world_model = torch.optim.SGD(
                self._model.world_model.parameters(),
                lr=self._cfg.learning_rate,
                momentum=self._cfg.momentum,
                weight_decay=self._cfg.weight_decay
            )
        elif self._cfg.optim_type == 'AdamW':
            # NOTE: nanoGPT optimizer
            self._optimizer_world_model = configure_optimizers_nanogpt(
                model=self._model.world_model,
                learning_rate=self._cfg.learning_rate,
                weight_decay=self._cfg.weight_decay,
                device_type=self._cfg.device,
                betas=(0.9, 0.95),
            )
        elif self._cfg.optim_type == 'AdamW_mix_lr_wdecay':
            self._optimizer_world_model = configure_optimizer_unizero(
                model=self._model.world_model,
                learning_rate=self._cfg.learning_rate,
                weight_decay=self._cfg.weight_decay,
                device_type=self._cfg.device,
                betas=(0.9, 0.95),
            )

        if self._cfg.cos_lr_scheduler:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            total_iters = self._cfg.get('total_iterations', 500000) # 500k iter
            final_lr = self._cfg.get('final_learning_rate', 1e-6)

            self.lr_scheduler = CosineAnnealingLR(
                self._optimizer_world_model,
                T_max=total_iters,
                eta_min=final_lr
            )
            print(f"CosineAnnealingLR enabled: T_max={total_iters}, eta_min={final_lr}")


        if self._cfg.piecewise_decay_lr_scheduler:
            from torch.optim.lr_scheduler import LambdaLR
            max_step = self._cfg.threshold_training_steps_for_final_lr
            # NOTE: the 1, 0.1, 0.01 is the decay rate, not the lr.
            lr_lambda = lambda step: 1 if step < max_step * 0.5 else (0.1 if step < max_step else 0.01)  # noqa
            self.lr_scheduler = LambdaLR(self._optimizer_world_model, lr_lambda=lr_lambda)


        # Use a deep copy for the target model.
        self._target_model = copy.deepcopy(self._model)
        # Ensure that the installed torch version is >= 2.0 for torch.compile.
        assert int(''.join(filter(str.isdigit, torch.__version__))) >= 200, "We need torch version >= 2.0"
        self._model = torch.compile(self._model)
        self._target_model = torch.compile(self._target_model)

        # Wrap the target model for soft updates (momentum-based).
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.target_update_theta}
        )
        self._learn_model = self._model

        if self._cfg.use_augmentation:
            self.image_transforms = ImageTransforms(
                self._cfg.augmentation,
                image_shape=(self._cfg.model.observation_shape[1], self._cfg.model.observation_shape[2])
            )

        self.value_support = DiscreteSupport(*self._cfg.model.value_support_range, self._cfg.device)
        self.reward_support = DiscreteSupport(*self._cfg.model.reward_support_range, self._cfg.device)
        self.value_inverse_scalar_transform_handle = InverseScalarTransform(self.value_support, self._cfg.model.categorical_distribution)
        self.reward_inverse_scalar_transform_handle = InverseScalarTransform(self.reward_support, self._cfg.model.categorical_distribution)

        self.intermediate_losses = defaultdict(float)
        self.l2_norm_before = 0.
        self.l2_norm_after = 0.
        self.grad_norm_before = 0.
        self.grad_norm_after = 0.



        self.task_id = self._cfg.task_id
        self.task_num_for_current_rank = self._cfg.task_num

        print(f'self._cfg.only_use_moco_stats:{self._cfg.only_use_moco_stats}')
        if self._cfg.use_moco or self._cfg.only_use_moco_stats:
            # The prediction heads' gradients are not corrected.
            self.wrapped_model = WrappedModelV2(
                # TODO: This assumes the tokenizer has an encoder attribute which is a list. This might need to be more robust.
                self._learn_model.world_model.tokenizer.encoder[0],
                self._learn_model.world_model.transformer,
                self._learn_model.world_model.pos_emb,
                self._learn_model.world_model.task_emb,
                self._learn_model.world_model.act_embedding_table,
            )

            # Pass the wrapped_model as `shared_module` to the gradient correction method.
            # ========= Initialize MoCo/CAGrad parameters =========
            if self._cfg.moco_version=="v0":
                # This version is only compatible with single-GPU training.
                self.grad_correct = GradCorrect(self.wrapped_model, self._cfg.total_task_num, self._cfg.device, self._cfg.multi_gpu)
                self.grad_correct.init_param()
                self.grad_correct.rep_grad = False
            elif self._cfg.moco_version=="v1":
                cfg_moco = MoCoCfg(
                    beta0=0.9,  beta_sigma=0.95,
                    gamma0=0.1, gamma_sigma=0.95,
                    rho=0.01,   stat_interval=10000)
                self.grad_correct = FastMoCo(
                    shared_module=self.wrapped_model,
                    world_task_num=self._cfg.total_task_num,   # Total number of tasks globally
                    device=self._cfg.device,
                    multi_gpu=self._cfg.multi_gpu,
                    cfg=cfg_moco,
                )

        # Cache for plasticity-related metrics from the previous frame.
        self._prev_plasticity_metrics = dict(
            dormant_ratio_encoder      = 0.0,
            dormant_ratio_transformer  = 0.0,
            dormant_ratio_head         = 0.0,
            avg_weight_mag_encoder     = 0.0,
            avg_weight_mag_transformer = 0.0,
            avg_weight_mag_head        = 0.0,
            e_rank_last_linear         = 0.0,
            e_rank_sim_norm            = 0.0,
        )

        # ==================== START: Target Entropy Regularization Initialization ====================
        # Read whether to enable adaptive alpha from config, and provide a default value
        self.use_adaptive_entropy_weight = self._cfg.get('use_adaptive_entropy_weight', True)

        # Add configuration in _init_learn
        self.target_entropy_start_ratio = self._cfg.get('target_entropy_start_ratio', 0.98)
        self.target_entropy_end_ratio = self._cfg.get('target_entropy_end_ratio', 0.7)
        self.target_entropy_decay_steps = self._cfg.get('target_entropy_decay_steps', 200000) # e.g., complete annealing within 200k steps (2M envsteps)

        if self.use_adaptive_entropy_weight:
            # 1. Set target entropy. For discrete action spaces, a common heuristic is the negative logarithm
            #    of action space dimension multiplied by a coefficient.
            #    This coefficient (e.g., 0.98) can be used as a hyperparameter.
            action_space_size = self._cfg.model.action_space_size
            self.target_entropy = -np.log(1.0 / action_space_size) * 0.98

            # 2. Initialize a learnable log_alpha parameter.
            #    Initialized to 0, meaning initial alpha = exp(0) = 1.0.
            self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=self._cfg.device), requires_grad=True)

            # 3. Create a dedicated optimizer for log_alpha.
            #    Using a smaller learning rate (e.g., 1e-4) different from the main optimizer is usually more stable.
            alpha_lr = self._cfg.get('adaptive_entropy_alpha_lr', 1e-4)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

            print("="*20)
            print(">>> Target Entropy Regularization (Adaptive Alpha) Enabled <<<")
            print(f"    Target Entropy: {self.target_entropy:.4f}")
            print(f"    Alpha Optimizer Learning Rate: {alpha_lr:.2e}")
            print("="*20)
        # ===================== END: Target Entropy Regularization Initialization =====================

        self.latent_norm_clip_threshold = self._cfg.get('latent_norm_clip_threshold', 30.0)
        # ==================== START: Initialize Encoder-Clip Annealing Parameters ====================
        self.use_encoder_clip_annealing = self._cfg.get('use_encoder_clip_annealing', False)
        if self.use_encoder_clip_annealing:
            self.encoder_clip_anneal_type = self._cfg.get('encoder_clip_anneal_type', 'cosine')
            self.encoder_clip_start = self._cfg.get('encoder_clip_start_value', 30.0)
            self.encoder_clip_end = self._cfg.get('encoder_clip_end_value', 10.0)
            self.encoder_clip_anneal_steps = self._cfg.get('encoder_clip_anneal_steps', 200000)

            print("="*20)
            print(">>> Encoder-Clip Annealing Enabled <<<")
            print(f"    Type: {self.encoder_clip_anneal_type}")
            print(f"    Range: {self.encoder_clip_start} -> {self.encoder_clip_end}")
            print(f"    Steps: {self.encoder_clip_anneal_steps}")
            print("="*20)
        else:
            # If annealing is not enabled, use a fixed clip threshold
            self.latent_norm_clip_threshold = self._cfg.get('latent_norm_clip_threshold', 30.0)
        # ===================== END: Initialize Encoder-Clip Annealing Parameters =====================

        # Policy Label Smoothing Parameters
        self.policy_ls_eps_start = self._cfg.get('policy_ls_eps_start', 0.05) # TODO policy_label_smoothing_eps_start: larger action space requires larger eps
        self.policy_ls_eps_end = self._cfg.get('policy_label_smoothing_eps_end ', 0.01) # TODO policy_label_smoothing_eps_start
        self.policy_ls_eps_decay_steps = self._cfg.get('policy_ls_eps_decay_steps ', 50000) # TODO 50k
        print(f"self.policy_ls_eps_start:{self.policy_ls_eps_start}")

    @staticmethod
    def _is_zero(x: Union[float, torch.Tensor], eps: float = 1e-8) -> bool:
        """
        Overview:
            Checks if a scalar or a 0-D tensor can be considered zero within a small tolerance.
        Arguments:
            - x (:obj:`Union[float, torch.Tensor]`): The input value to check.
            - eps (:obj:`float`): The tolerance for checking against zero.
        Returns:
            - (:obj:`bool`): True if the value is close to zero, False otherwise.
        """
        if isinstance(x, torch.Tensor):
            return torch.all(torch.abs(x) < eps).item()
        return abs(x) < eps

    def _retain_prev_if_zero(self, name: str,
                             value: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        Overview:
            If the current `value` is close to zero, returns the cached value from the previous frame.
            Otherwise, it updates the cache with the current value and returns it. This is useful for
            metrics that are computed intermittently.
        Arguments:
            - name (:obj:`str`): The name of the metric to cache.
            - value (:obj:`Union[float, torch.Tensor]`): The current value of the metric.
        Returns:
            - (:obj:`Union[float, torch.Tensor]`): The retained or current value.
        """
        if self._is_zero(value):
            # Directly return the previous value (can be float or tensor).
            return self._prev_plasticity_metrics[name]
        else:
            # Update the cache and return the current value.
            self._prev_plasticity_metrics[name] = value
            return value


    def _forward_learn(self, data: Tuple[torch.Tensor], task_weights=None, train_iter=None, ignore_grad=False) -> Dict[str, Union[float, int]]:
        """
        Overview:
            The forward function for learning in the policy. This is the core of the training process.
            Data is sampled from the replay buffer, losses are calculated, and the model is updated via backpropagation.
        Arguments:
            - data (:obj:`Tuple[torch.Tensor]`): A tuple of data batches, where each element corresponds to a different task.
            - task_weights (:obj:`Any`, optional): Optional weights for each task's loss. Not currently used.
            - ignore_grad (:obj:`bool`): If True, gradients are zeroed out after computation, effectively skipping the update.
        Returns:
            - info_dict (:obj:`Dict[str, Union[float, int]]`): A dictionary containing current learning losses and statistics for logging.
        """
        self._learn_model.train()
        self._target_model.train()

        # Lists to store metrics for each task within the batch.
        obs_loss_multi_task = []
        reward_loss_multi_task = []
        policy_loss_multi_task = []
        value_loss_multi_task = []
        latent_recon_loss_multi_task = []
        perceptual_loss_multi_task = []
        orig_policy_loss_multi_task = []
        policy_entropy_multi_task = []
        weighted_total_loss = 0.0  # Initialize to 0.0 to avoid in-place operations.
        total_alpha_loss = 0.0

        latent_state_l2_norms_multi_task = []
        average_target_policy_entropy_multi_task = []
        value_priority_multi_task = []
        value_priority_mean_multi_task = []

        # Metrics for network plasticity analysis.
        dormant_ratio_encoder_multi_task = []
        dormant_ratio_transformer_multi_task = []
        dormant_ratio_head_multi_task = []
        avg_weight_mag_encoder_multi_task = []
        avg_weight_mag_transformer_multi_task = []
        avg_weight_mag_head_multi_task = []
        e_rank_last_linear_multi_task = []
        e_rank_sim_norm_multi_task = []

        current_policy_label_eps = 0.01

        # Add a list to collect real global IDs of all tasks in the current batch
        global_task_ids_in_batch = []
        alpha_loss = None


        # New lists for Alpha logging
        alpha_loss_multi_task = []
        target_entropy_multi_task = []

        # Pre-fetch current alpha value only when adaptive alpha is enabled, ensuring consistency across all tasks in a single iteration
        current_alpha = self._cfg.model.world_model_cfg.policy_entropy_weight
        if self.use_adaptive_entropy_weight:
            current_alpha = self.log_alpha.exp().detach()

        losses_list = []  # Used to store the loss tensor for each task, required by gradient correction methods.
        for task_id, data_one_task in enumerate(data):
            current_batch, target_batch, task_id = data_one_task # task_id is the real global ID

            # Add the real global ID to the list
            global_task_ids_in_batch.append(task_id)

            # TODO: Adapt RoPE for multitask settings (using timestep_batch).
            obs_batch_ori, action_batch, target_action_batch, mask_batch, indices, weights, make_time, timestep_batch  = current_batch
            target_reward, target_value, target_policy = target_batch

            # Prepare observations based on frame stack number.
            if self._cfg.model.frame_stack_num == 4:
                obs_batch, obs_target_batch = prepare_obs_stack_for_unizero(obs_batch_ori, self._cfg)
            else:
                obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, self._cfg)

            # Apply augmentations if needed.
            if self._cfg.use_augmentation:
                obs_batch = self.image_transforms.transform(obs_batch)
                if self._cfg.model.self_supervised_learning_loss:
                    obs_target_batch = self.image_transforms.transform(obs_target_batch)

            # Prepare action batch and convert to a torch tensor.
            action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(
                -1).long()  # For discrete action space.
            data_list = [mask_batch, target_reward.astype('float32'), target_value.astype('float32'), target_policy,
                         weights]
            mask_batch, target_reward, target_value, target_policy, weights = to_torch_float_tensor(data_list,
                                                                                                    self._cfg.device)

            cur_batch_size = target_reward.size(0)          # Run-time batch size.

            target_reward = target_reward.view(cur_batch_size, -1)
            target_value = target_value.view(cur_batch_size, -1)

            # Transform scalar rewards and values to their scaled representations.
            transformed_target_reward = scalar_transform(target_reward)
            transformed_target_value = scalar_transform(target_value)

            # Convert scaled representations to categorical distributions.
            target_reward_categorical = phi_transform(self.reward_support, transformed_target_reward, label_smoothing_eps= self._cfg.label_smoothing_eps)
            target_value_categorical = phi_transform(self.value_support, transformed_target_value, label_smoothing_eps=self._cfg.label_smoothing_eps)


            # Prepare the batch for the transformer-based world model.
            batch_for_gpt = {}
            if isinstance(self._cfg.model.observation_shape, int) or len(self._cfg.model.observation_shape) == 1:
                batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
                    cur_batch_size, -1, self._cfg.model.observation_shape)
            elif len(self._cfg.model.observation_shape) == 3:
                batch_for_gpt['observations'] = torch.cat((obs_batch, obs_target_batch), dim=1).reshape(
                    cur_batch_size, -1, *self._cfg.model.observation_shape)

            batch_for_gpt['actions'] = action_batch.squeeze(-1)
            batch_for_gpt['rewards'] = target_reward_categorical[:, :-1]
            batch_for_gpt['mask_padding'] = mask_batch == 1.0  # 0 means invalid padding data.
            batch_for_gpt['mask_padding'] = batch_for_gpt['mask_padding'][:, :-1]
            batch_for_gpt['observations'] = batch_for_gpt['observations'][:, :-1]
            batch_for_gpt['ends'] = torch.zeros(batch_for_gpt['mask_padding'].shape, dtype=torch.long,
                                                device=self._cfg.device)
            batch_for_gpt['target_value'] = target_value_categorical[:, :-1]
            batch_for_gpt['target_policy'] = target_policy[:, :-1]
            batch_for_gpt['scalar_target_value'] = target_value

            # Extract valid target policy data and compute its entropy.
            valid_target_policy = batch_for_gpt['target_policy'][batch_for_gpt['mask_padding']]
            target_policy_entropy = -torch.sum(valid_target_policy * torch.log(valid_target_policy + 1e-9), dim=-1)
            average_target_policy_entropy = target_policy_entropy.mean().item()

            # Update world model and compute losses.
            intermediate_losses = defaultdict(float)

            losses = self._learn_model.world_model.compute_loss(
                batch_for_gpt, self._target_model.world_model.tokenizer, self.value_inverse_scalar_transform_handle, current_policy_label_eps=current_policy_label_eps, task_id=task_id
            )

            # Extract the calculated value_priority from the returned losses.
            value_priority_tensor = losses.intermediate_losses['value_priority']
            # Convert to numpy array for the replay buffer, adding a small epsilon.
            value_priority_np = value_priority_tensor.detach().cpu().numpy() + 1e-6


            # TODO: Accumulate the weighted total loss. This assumes the loss from `compute_loss` is already weighted.
            weighted_total_loss += losses.loss_total # NOTE:+=

            # TODO: Append the total loss for this task, used by MoCo.
            losses_list.append(losses.loss_total)

            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"{loss_name}"] = loss_value



            obs_loss = intermediate_losses['loss_obs']
            reward_loss = intermediate_losses['loss_rewards']
            policy_loss = intermediate_losses['loss_policy']
            orig_policy_loss = intermediate_losses['orig_policy_loss']
            policy_entropy = intermediate_losses['policy_entropy']
            value_loss = intermediate_losses['loss_value']
            latent_recon_loss = intermediate_losses['latent_recon_loss']
            perceptual_loss = intermediate_losses['perceptual_loss']
            latent_state_l2_norms = intermediate_losses['latent_state_l2_norms']

            # ==================== START: Target Entropy Regularization Update Logic ====================
            current_alpha = self._cfg.model.world_model_cfg.policy_entropy_weight # Default to fixed value
            if self.use_adaptive_entropy_weight:

                # Dynamically calculate target entropy (this logic is correct and preserved)
                progress = min(1.0, train_iter / self.target_entropy_decay_steps)
                current_ratio = self.target_entropy_start_ratio * (1 - progress) + self.target_entropy_end_ratio * progress
                action_space_size = self._cfg.model.action_space_size
                # Note: We define target_entropy as a positive number, which is more intuitive
                current_target_entropy = -np.log(1.0 / action_space_size) * current_ratio

                # Calculate alpha_loss (corrected sign)
                # This is the core correction: removed the negative sign at the front
                # detach() is still critical to ensure alpha_loss gradient only flows to log_alpha
                alpha_loss_task = (self.log_alpha * (policy_entropy.detach() - current_target_entropy)).mean() # NOTE:=


                # Accumulate alpha_loss
                total_alpha_loss += alpha_loss_task
                # Collect each task's alpha_loss and target_entropy for logging
                alpha_loss_multi_task.append(alpha_loss_task)
                target_entropy_multi_task.append(current_target_entropy)

                # [Optimization suggestion] Add log_alpha clipping as a safety measure
                with torch.no_grad():
                    # Limit alpha to a range, e.g., [1e-4, 10.0]
                    self.log_alpha.clamp_(np.log(5e-3), np.log(10.0))


                # Use current updated alpha (with gradient flow truncated)
                current_alpha = self.log_alpha.exp().detach()

                # Recalculate weighted policy loss and total loss
                # Note: policy_entropy here is already an average value of a batch
                weighted_policy_loss = orig_policy_loss - current_alpha * policy_entropy
                # Rebuild total loss (not using losses.loss_total)
                # Ensure the weights here are consistent with the calculation in LossWithIntermediateLosses class
                self.obs_loss_weight = 10
                self.value_loss_weight = 0.5
                self.reward_loss_weight = 1.
                self.policy_loss_weight = 1.
                self.ends_loss_weight = 0.
                total_loss = (
                    self.reward_loss_weight * reward_loss +
                    self.value_loss_weight * value_loss +
                    self.policy_loss_weight * weighted_policy_loss +
                    self.obs_loss_weight  * obs_loss
                )
                weighted_total_loss += (weights * total_loss).mean() # NOTE:+=
            # ===================== END: Target Entropy Regularization Update Logic =====================

            # Metrics related to network plasticity.
            # Use the helper function to retain the previous value if the current one is zero.
            dormant_ratio_encoder  = self._retain_prev_if_zero(
                                'dormant_ratio_encoder',
                                            intermediate_losses['dormant_ratio_encoder'])
            dormant_ratio_transformer  = self._retain_prev_if_zero(
                                            'dormant_ratio_transformer',
                                            intermediate_losses['dormant_ratio_transformer'])
            dormant_ratio_head         = self._retain_prev_if_zero(
                                            'dormant_ratio_head',
                                            intermediate_losses['dormant_ratio_head'])
            avg_weight_mag_encoder     = self._retain_prev_if_zero(
                                            'avg_weight_mag_encoder',
                                            intermediate_losses['avg_weight_mag_encoder'])
            avg_weight_mag_transformer = self._retain_prev_if_zero(
                                            'avg_weight_mag_transformer',
                                            intermediate_losses['avg_weight_mag_transformer'])
            avg_weight_mag_head        = self._retain_prev_if_zero(
                                            'avg_weight_mag_head',
                                            intermediate_losses['avg_weight_mag_head'])
            e_rank_last_linear         = self._retain_prev_if_zero(
                                            'e_rank_last_linear',
                                            intermediate_losses['e_rank_last_linear'])
            e_rank_sim_norm            = self._retain_prev_if_zero(
                                            'e_rank_sim_norm',
                                            intermediate_losses['e_rank_sim_norm'])

            # Append all metrics for this task to their respective lists.
            obs_loss_multi_task.append(obs_loss)
            reward_loss_multi_task.append(reward_loss)
            policy_loss_multi_task.append(policy_loss)
            orig_policy_loss_multi_task.append(orig_policy_loss)
            policy_entropy_multi_task.append(policy_entropy)
            value_loss_multi_task.append(value_loss)
            latent_recon_loss_multi_task.append(latent_recon_loss)
            perceptual_loss_multi_task.append(perceptual_loss)
            latent_state_l2_norms_multi_task.append(latent_state_l2_norms)
            value_priority_multi_task.append(value_priority_tensor)
            value_priority_mean_multi_task.append(value_priority_tensor.mean().item())

            # Append plasticity metrics.
            dormant_ratio_encoder_multi_task.append(dormant_ratio_encoder)
            dormant_ratio_transformer_multi_task.append(dormant_ratio_transformer)
            dormant_ratio_head_multi_task.append(dormant_ratio_head)
            avg_weight_mag_encoder_multi_task.append(avg_weight_mag_encoder)
            avg_weight_mag_transformer_multi_task.append(avg_weight_mag_transformer)
            avg_weight_mag_head_multi_task.append(avg_weight_mag_head)
            e_rank_last_linear_multi_task.append(e_rank_last_linear)
            e_rank_sim_norm_multi_task.append(e_rank_sim_norm)


        # ==================== Integrate norm monitoring logic ====================
        norm_log_dict = {}
        # Check if monitoring frequency is reached
        if self._cfg.monitor_norm_freq > 0 and (train_iter == 0 or (train_iter % self._cfg.monitor_norm_freq == 0)):
            with torch.no_grad():
                # 1. Monitor model parameter norms
                param_norm_metrics = self._monitor_model_norms()
                norm_log_dict.update(param_norm_metrics)

                # 2. Monitor intermediate tensor x (Transformer output)
                intermediate_x = losses.intermediate_losses.get('intermediate_tensor_x')
                if intermediate_x is not None:
                    # x shape is (B, T, E)
                    # Calculate L2 norm for each token
                    token_norms = intermediate_x.norm(p=2, dim=-1)

                    # Record statistics of these norms
                    norm_log_dict['norm/x_token/mean'] = token_norms.mean().item()
                    norm_log_dict['norm/x_token/std'] = token_norms.std().item()
                    norm_log_dict['norm/x_token/max'] = token_norms.max().item()
                    norm_log_dict['norm/x_token/min'] = token_norms.min().item()

                # 3. Monitor detailed statistics of logits (Value, Policy, Reward)
                logits_value = losses.intermediate_losses.get('logits_value')
                if logits_value is not None:
                    norm_log_dict['logits/value/mean'] = logits_value.mean().item()
                    norm_log_dict['logits/value/std'] = logits_value.std().item()
                    norm_log_dict['logits/value/max'] = logits_value.max().item()
                    norm_log_dict['logits/value/min'] = logits_value.min().item()
                    norm_log_dict['logits/value/abs_max'] = logits_value.abs().max().item()

                logits_policy = losses.intermediate_losses.get('logits_policy')
                if logits_policy is not None:
                    norm_log_dict['logits/policy/mean'] = logits_policy.mean().item()
                    norm_log_dict['logits/policy/std'] = logits_policy.std().item()
                    norm_log_dict['logits/policy/max'] = logits_policy.max().item()
                    norm_log_dict['logits/policy/min'] = logits_policy.min().item()
                    norm_log_dict['logits/policy/abs_max'] = logits_policy.abs().max().item()

                logits_reward = losses.intermediate_losses.get('logits_reward')
                if logits_reward is not None:
                    norm_log_dict['logits/reward/mean'] = logits_reward.mean().item()
                    norm_log_dict['logits/reward/std'] = logits_reward.std().item()
                    norm_log_dict['logits/reward/max'] = logits_reward.max().item()
                    norm_log_dict['logits/reward/min'] = logits_reward.min().item()
                    norm_log_dict['logits/reward/abs_max'] = logits_reward.abs().max().item()

                # 4. Monitor obs_embeddings (Encoder output) statistics
                obs_embeddings = losses.intermediate_losses.get('obs_embeddings')
                if obs_embeddings is not None:
                    # Calculate L2 norm for each embedding
                    emb_norms = obs_embeddings.norm(p=2, dim=-1)
                    norm_log_dict['embeddings/obs/norm_mean'] = emb_norms.mean().item()
                    norm_log_dict['embeddings/obs/norm_std'] = emb_norms.std().item()
                    norm_log_dict['embeddings/obs/norm_max'] = emb_norms.max().item()
                    norm_log_dict['embeddings/obs/norm_min'] = emb_norms.min().item()
        # =================================================================

        # Core learn model update step.
        self._optimizer_world_model.zero_grad()

        if self.use_adaptive_entropy_weight:
            self.alpha_optimizer.zero_grad()
        # 2. Calculate final alpha loss (average after accumulation)
        final_alpha_loss = None
        if self.use_adaptive_entropy_weight:
            if len(data) > 0:
                final_alpha_loss = total_alpha_loss / len(data)
            else: # Defensive programming to avoid division by zero
                final_alpha_loss = torch.tensor(0.0, device=self._cfg.device)

        # Assuming losses_list is a list of tensors with gradients, e.g., [loss1, loss2, ...].
        if self._cfg.use_moco:
            # Call MoCo's backward method, which handles gradient correction internally.
            if self._cfg.moco_version=="v0":
                lambd, stats = self.grad_correct.backward(losses=losses_list, **self._cfg.grad_correct_params)
            elif self._cfg.moco_version=="v1":
                lambd, stats = self.grad_correct.backward(losses_list)

            # Separate backward pass for alpha loss
            if self.use_adaptive_entropy_weight:
                final_alpha_loss.backward()

        elif self._cfg.only_use_moco_stats:
            # Only compute MoCo stats without applying gradient correction.
            lambd, stats = self.grad_correct.backward(losses=losses_list, **self._cfg.grad_correct_params)

            # If adaptive alpha is enabled, add alpha loss to main loss for joint backward pass
            if self.use_adaptive_entropy_weight:
                (weighted_total_loss + final_alpha_loss).backward()
            elif weighted_total_loss != 0.0: # Ensure there is loss to backpropagate
                weighted_total_loss.backward()

        else:
            # If not using gradient correction, each rank performs standard backpropagation.
            lambd = torch.tensor([0. for _ in range(self.task_num_for_current_rank)], device=self._cfg.device)

            # If adaptive alpha is enabled, add alpha loss to main loss for joint backward pass
            if self.use_adaptive_entropy_weight:
                (weighted_total_loss + final_alpha_loss).backward()
            elif weighted_total_loss != 0.0: # Ensure there is loss to backpropagate
                weighted_total_loss.backward()

        # Still executed within torch.no_grad() context
        # =================================================================
        with torch.no_grad():
            # 1. Encoder-Clip
            # ==================== START: Dynamically calculate current Clip threshold ====================
            current_clip_value = self.latent_norm_clip_threshold  # Default to fixed value
            if self.use_encoder_clip_annealing:
                progress = min(1.0, train_iter / self.encoder_clip_anneal_steps)

                if self.encoder_clip_anneal_type == 'cosine':
                    # Cosine schedule: smoothly transition from 1 to 0
                    cosine_progress = 0.5 * (1.0 + np.cos(np.pi * progress))
                    current_clip_value = self.encoder_clip_end + \
                                         (self.encoder_clip_start - self.encoder_clip_end) * cosine_progress
                else:  # Default to linear schedule
                    current_clip_value = self.encoder_clip_start * (1 - progress) + \
                                         self.encoder_clip_end * progress
            # ===================== END: Dynamically calculate current Clip threshold =====================

            # 1. Encoder-Clip (using dynamically calculated current_clip_value)
            if current_clip_value > 0 and 'obs_embeddings' in losses.intermediate_losses:
                obs_embeddings = losses.intermediate_losses['obs_embeddings']
                if obs_embeddings is not None:
                    max_latent_norm = obs_embeddings.norm(p=2, dim=-1).max()
                    if max_latent_norm > current_clip_value:
                        scale_factor = current_clip_value / max_latent_norm.item()
                        # No longer print frequently, or can be changed to print every N steps
                        if train_iter % 1000 == 0:
                            print(f"[Encoder-Clip Annealing] Iter {train_iter}: Max latent norm {max_latent_norm.item():.2f} > {current_clip_value:.2f}. Scaling by {scale_factor:.4f}.")
                        scale_module_weights_vectorized(self._model.world_model.tokenizer.encoder, scale_factor)


        # ==================== Monitor gradient norms ====================
        # Monitor gradient norms before gradient clipping to diagnose gradient explosion/vanishing issues
        if self._cfg.monitor_norm_freq > 0 and (train_iter == 0 or (train_iter % self._cfg.monitor_norm_freq == 0)):
            grad_norm_metrics = self._monitor_gradient_norms()
            norm_log_dict.update(grad_norm_metrics)
        # =================================================================

        if self._cfg.analysis_sim_norm:
            del self.l2_norm_before, self.l2_norm_after, self.grad_norm_before, self.grad_norm_after
            self.l2_norm_before, self.l2_norm_after, self.grad_norm_before, self.grad_norm_after = self._learn_model.encoder_hook.analyze()
            self._target_model.encoder_hook.clear_data()

        total_grad_norm_before_clip_wm = torch.nn.utils.clip_grad_norm_(self._learn_model.world_model.parameters(),
                                                                        self._cfg.grad_clip_value)

        if ignore_grad:
            # NOTE: For cases where all tasks on a GPU are solved, `train` is still called for DDP synchronization,
            # but gradients should be zeroed out to prevent updates.
            self._optimizer_world_model.zero_grad()

        if self._cfg.multi_gpu:
            # If not using a gradient correction method that handles it, sync gradients manually.
            if not self._cfg.use_moco:
                self.sync_gradients(self._learn_model)

        self._optimizer_world_model.step()

        # 4. Update Alpha optimizer
        if self.use_adaptive_entropy_weight:
            self.alpha_optimizer.step()
            # Clip log_alpha to ensure stability
            with torch.no_grad():
                self.log_alpha.clamp_(np.log(1e-4), np.log(10.0))

        if self._cfg.cos_lr_scheduler or self._cfg.piecewise_decay_lr_scheduler:
            self.lr_scheduler.step()

        # Core target model update step.
        self._target_model.update(self._learn_model.state_dict())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_memory_allocated = torch.cuda.memory_allocated()
            max_memory_allocated = torch.cuda.max_memory_allocated()
            current_memory_allocated_gb = current_memory_allocated / (1024 ** 3)
            max_memory_allocated_gb = max_memory_allocated / (1024 ** 3)
        else:
            current_memory_allocated_gb = 0.
            max_memory_allocated_gb = 0.

        # Build the dictionary of return values for logging.
        return_log_dict = {
            'Current_GPU': current_memory_allocated_gb,
            'Max_GPU': max_memory_allocated_gb,
            'collect_mcts_temperature': self._collect_mcts_temperature,
            'collect_epsilon': self._collect_epsilon,
            'cur_lr_world_model': self._optimizer_world_model.param_groups[0]['lr'],
            'weighted_total_loss': weighted_total_loss.item(),
            'total_grad_norm_before_clip_wm': total_grad_norm_before_clip_wm.item(),
        }

        # ==================== START: Add new log items ====================
        if self.use_adaptive_entropy_weight:
            return_log_dict['adaptive_alpha'] = current_alpha.item()
            return_log_dict['adaptive_target_entropy_ratio'] = current_ratio
            return_log_dict['final_alpha_loss'] = final_alpha_loss.item()
        # ===================== END: Add new log items =====================

        # Generate task-related loss dictionaries and prefix each task-related loss with "noreduce_".
        multi_task_loss_dicts = {
            **generate_task_loss_dict(obs_loss_multi_task, 'noreduce_obs_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(latent_recon_loss_multi_task, 'noreduce_latent_recon_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(perceptual_loss_multi_task, 'noreduce_perceptual_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(latent_state_l2_norms_multi_task, 'noreduce_latent_state_l2_norms_task{}', task_id=self.task_id),
            **generate_task_loss_dict(dormant_ratio_head_multi_task, 'noreduce_dormant_ratio_head_task{}', task_id=self.task_id),

            **generate_task_loss_dict(policy_loss_multi_task, 'noreduce_policy_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(orig_policy_loss_multi_task, 'noreduce_orig_policy_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(policy_entropy_multi_task, 'noreduce_policy_entropy_task{}', task_id=self.task_id),
            **generate_task_loss_dict(reward_loss_multi_task, 'noreduce_reward_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(value_loss_multi_task, 'noreduce_value_loss_task{}', task_id=self.task_id),
            **generate_task_loss_dict(average_target_policy_entropy_multi_task, 'noreduce_target_policy_entropy_task{}', task_id=self.task_id),
            **generate_task_loss_dict(lambd, 'noreduce_lambd_task{}', task_id=self.task_id),
            **generate_task_loss_dict(value_priority_multi_task, 'noreduce_value_priority_task{}', task_id=self.task_id),
            **generate_task_loss_dict(value_priority_mean_multi_task, 'noreduce_value_priority_mean_task{}', task_id=self.task_id),

            # Add alpha related logs
            **generate_task_loss_dict(alpha_loss_multi_task, 'noreduce_alpha_loss_task{}', self.task_id),
            **generate_task_loss_dict(target_entropy_multi_task, 'noreduce_target_entropy_task{}', self.task_id),
        }
        return_log_dict.update(multi_task_loss_dicts)


        if self._learn_model.world_model.do_analysis:
            # Include plasticity metrics if analysis is enabled.
            plasticity_loss_dicts = {
                **generate_task_loss_dict(dormant_ratio_encoder_multi_task, 'noreduce_dormant_ratio_encoder_task{}', task_id=self.task_id),
                **generate_task_loss_dict(dormant_ratio_transformer_multi_task, 'noreduce_dormant_ratio_transformer_task{}', task_id=self.task_id),
                **generate_task_loss_dict(dormant_ratio_head_multi_task, 'noreduce_dormant_ratio_head_task{}', task_id=self.task_id),
                **generate_task_loss_dict(avg_weight_mag_encoder_multi_task, 'noreduce_avg_weight_mag_encoder_task{}', task_id=self.task_id),
                **generate_task_loss_dict(avg_weight_mag_transformer_multi_task, 'noreduce_avg_weight_mag_transformer_task{}', task_id=self.task_id),
                **generate_task_loss_dict(avg_weight_mag_head_multi_task, 'noreduce_avg_weight_mag_head_task{}', task_id=self.task_id),
                **generate_task_loss_dict(e_rank_last_linear_multi_task, 'noreduce_e_rank_last_linear_task{}', task_id=self.task_id),
                **generate_task_loss_dict(e_rank_sim_norm_multi_task, 'noreduce_e_rank_sim_norm_task{}', task_id=self.task_id),
            }
            # Merge the dictionaries.
            return_log_dict.update(plasticity_loss_dicts)

        # Merge norm monitoring results into the log
        if norm_log_dict:
            return_log_dict.update(norm_log_dict)

        # Return the final loss dictionary.
        return return_log_dict

    def monitor_weights_and_grads(self, model: torch.nn.Module) -> None:
        """
        Overview:
            A utility function to print the mean and standard deviation of weights and their gradients for each layer in a model.
            Useful for debugging training issues like exploding or vanishing gradients.
        Arguments:
            - model (:obj:`torch.nn.Module`): The model to monitor.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | "
                      f"Weight mean: {param.data.mean():.4f} | "
                      f"Weight std: {param.data.std():.4f} | "
                      f"Grad mean: {param.grad.mean():.4f} | "
                      f"Grad std: {param.grad.std():.4f}")

    def _init_collect(self) -> None:
        """
        Overview:
            Initializes the collect mode. This method is called by ``self.__init__``.
            It sets up the collect model and MCTS utilities for data collection.
        """
        self._collect_model = self._model

        # Create a copy of the configuration for collect MCTS and set a specific number of simulations.
        mcts_collect_cfg = copy.deepcopy(self._cfg)
        mcts_collect_cfg.num_simulations = self._cfg.collect_num_simulations

        if self._cfg.mcts_ctree:
            self._mcts_collect = MCTSCtree(mcts_collect_cfg)
        else:
            self._mcts_collect = MCTSPtree(mcts_collect_cfg)

        self._collect_mcts_temperature = 1.
        self._collect_epsilon = 0.0
        self.collector_env_num = self._cfg.collector_env_num
        if self._cfg.model.model_type == 'conv':
            self.last_batch_obs = torch.zeros([self.collector_env_num, self._cfg.model.observation_shape[0], 64, 64]).to(self._cfg.device)
            self.last_batch_action = [-1 for i in range(self.collector_env_num)]
        elif self._cfg.model.model_type == 'mlp':
            self.last_batch_obs = torch.zeros([self.collector_env_num, self._cfg.model.observation_shape]).to(self._cfg.device)
            self.last_batch_action = [-1 for i in range(self.collector_env_num)]

    # TODO: The num_tasks parameter is hardcoded. It should ideally be derived from the config.
    def _monitor_vars_learn(self, num_tasks: int = 2) -> List[str]:
        """
        Overview:
            Registers variables to be monitored during training. These variables will be logged in TensorBoard.
            It dynamically creates variable names for each task if `num_tasks` is provided.
        Arguments:
            - num_tasks (:obj:`int`): The number of tasks being trained on the current rank.
        Returns:
            - monitored_vars (:obj:`List[str]`): A list of strings, where each string is the name of a variable to be logged.
        """
        # Basic monitored variables that do not depend on the number of tasks.
        monitored_vars = [
            'Current_GPU',
            'Max_GPU',
            'collect_epsilon',
            'collect_mcts_temperature',
            'cur_lr_world_model',
            'weighted_total_loss',
            'total_grad_norm_before_clip_wm',

            'adaptive_alpha',
            "adaptive_target_entropy_ratio",
            'final_alpha_loss',
        ]

        # ==================== Norm and Intermediate Tensor Monitoring Variables ====================
        # These variables are shared across all tasks (not per-task)
        norm_vars = [
            # Module total norms (parameter norms) - shared modules
            'norm/encoder/_total_norm',
            'norm/transformer/_total_norm',

            # Module total norms (gradient norms) - shared modules
            'grad/encoder/_total_norm',
            'grad/transformer/_total_norm',

            # Intermediate tensor x (Transformer output) statistics
            'norm/x_token/mean',
            'norm/x_token/std',
            'norm/x_token/max',
            'norm/x_token/min',

            # Detailed logits statistics (Value)
            'logits/value/mean',
            'logits/value/std',
            'logits/value/max',
            'logits/value/min',
            'logits/value/abs_max',

            # Detailed logits statistics (Policy)
            'logits/policy/mean',
            'logits/policy/std',
            'logits/policy/max',
            'logits/policy/min',
            'logits/policy/abs_max',

            # Detailed logits statistics (Reward)
            'logits/reward/mean',
            'logits/reward/std',
            'logits/reward/max',
            'logits/reward/min',
            'logits/reward/abs_max',

            # Embeddings statistics
            'embeddings/obs/norm_mean',
            'embeddings/obs/norm_std',
            'embeddings/obs/norm_max',
            'embeddings/obs/norm_min',
        ]
        monitored_vars.extend(norm_vars)
        # ========================================================================



        # Task-specific variables to be monitored.
        task_specific_vars = [
            'noreduce_obs_loss',
            'noreduce_orig_policy_loss',
            'noreduce_policy_loss',
            'noreduce_latent_recon_loss',
            'noreduce_policy_entropy',
            'noreduce_target_policy_entropy',
            'noreduce_reward_loss',
            'noreduce_value_loss',
            'noreduce_perceptual_loss',
            'noreduce_latent_state_l2_norms',
            'noreduce_lambd',
            'noreduce_value_priority_mean',
            # Metrics related to network plasticity.
            'noreduce_dormant_ratio_encoder',
            'noreduce_dormant_ratio_transformer',
            'noreduce_dormant_ratio_head',
            'noreduce_avg_weight_mag_encoder',
            'noreduce_avg_weight_mag_transformer',
            'noreduce_avg_weight_mag_head',
            'noreduce_e_rank_last_linear',
            'noreduce_e_rank_sim_norm',
            "noreduce_alpha_loss",
            "noreduce_target_entropy",

        ]

        # Use self.task_num_for_current_rank as the number of tasks for the current rank.
        num_tasks = self.task_num_for_current_rank
        # If the number of tasks is provided, extend the monitored variables list with task-specific variable names.
        if num_tasks is not None:
            for var in task_specific_vars:
                for task_idx in range(num_tasks):
                    monitored_vars.append(f'{var}_task{self.task_id+task_idx}')
        else:
            # If num_tasks is not provided, assume a single task and use the original variable names.
            monitored_vars.extend(task_specific_vars)

        return monitored_vars

    def _forward_collect(
            self,
            data: torch.Tensor,
            action_mask: list = None,
            temperature: float = 1,
            to_play: List = [-1],
            epsilon: float = 0.25,
            ready_env_id: np.array = None,
            timestep: List = [0],
            task_id: int = None,
    ) -> Dict:
        """
        Overview:
            The forward function for collecting data. It uses the model to perform MCTS search and
            selects actions via sampling to encourage exploration.
        Arguments:
            - data (:obj:`torch.Tensor`): The input data, i.e., the current observation.
            - action_mask (:obj:`list`, optional): A list of action masks for each environment.
            - temperature (:obj:`float`, optional): The temperature for MCTS action selection.
            - to_play (:obj:`List`, optional): A list of player IDs for each environment.
            - epsilon (:obj:`float`, optional): The probability for epsilon-greedy exploration.
            - ready_env_id (:obj:`np.array`, optional): An array of IDs for environments that are ready for a new action.
            - timestep (:obj:`List`, optional): The current timestep in each environment.
            - task_id (:obj:`int`, optional): The ID of the task for the current environments.
        Returns:
            - output (:obj:`Dict`): A dictionary where keys are environment IDs and values are dictionaries
              containing the selected action and other MCTS statistics.
        """
        self._collect_model.eval()

        self._collect_mcts_temperature = temperature
        self._collect_epsilon = epsilon
        active_collect_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_collect_env_num)
        output = {i: None for i in ready_env_id}

        with torch.no_grad():
            network_output = self._collect_model.initial_inference(self.last_batch_obs, self.last_batch_action, data, task_id=task_id)
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            pred_values = self.value_inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()

            # Core fix: C++ binding requires a list, even though it represents rewards in MuZero.
            reward_roots = reward_roots.detach().cpu().numpy().tolist()

            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)]
            # The main difference between collect and eval is the addition of Dirichlet noise at the root.
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                    ).astype(np.float32).tolist() for j in range(active_collect_env_num)
            ]
            if self._cfg.mcts_ctree:
                # C++ MCTS tree implementation.
                roots = MCTSCtree.roots(active_collect_env_num, legal_actions)
            else:
                # Python MCTS tree implementation.
                roots = MCTSPtree.roots(active_collect_env_num, legal_actions)


            roots.prepare(self._cfg.root_noise_weight, noises, reward_roots, policy_logits, to_play)
            self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play,  timestep= timestep, task_id=task_id)

            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()

            batch_action = []
            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]

                if self._cfg.eps.eps_greedy_exploration_in_collect:
                    # Epsilon-greedy collection strategy.
                    action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                        distributions, temperature=self._collect_mcts_temperature, deterministic=True
                    )
                    action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                    if np.random.rand() < self._collect_epsilon:
                        action = np.random.choice(legal_actions[i])
                else:
                    # Standard collection strategy (sampling from MCTS policy).
                    # NOTE: `action_index_in_legal_action_set` is the index within the set of legal actions.
                    action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                        distributions, temperature=self._collect_mcts_temperature, deterministic=False
                    )
                    # Convert the index back to the action in the full action space.
                    action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]

                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'predicted_value': pred_values[i],
                    'predicted_policy_logits': policy_logits[i],
                }
                batch_action.append(action)

            self.last_batch_obs = data
            self.last_batch_action = batch_action

            # TODO: This logic is currently for the `muzero_segment_collector`.
            if active_collect_env_num < self.collector_env_num:
                # When one environment in `collect_env` finishes early, the length of `self.last_batch_obs` is reduced.
                # The transformer needs the `env_id` to retrieve from the KV cache, which is complex to manage with a dynamic batch size.
                # Therefore, we reset `self.last_batch_action` for all environments to -1, forcing the transformer
                # to start from scratch and avoid retrieval errors.
                print('==========collect_forward============')
                print(f'len(self.last_batch_obs) < self.collector_env_num, {active_collect_env_num}<{self.collector_env_num}')
                self._reset_collect(reset_init_data=True, task_id=task_id)
                if getattr(self._cfg, 'sample_type', '') == 'episode':
                    print('BUG: sample_type is episode, but len(self.last_batch_obs) < self.collector_env_num')

        return output

    def _init_eval(self) -> None:
        """
        Overview:
            Initializes the eval mode. This method is called by ``self.__init__``.
            It sets up the eval model and MCTS utilities for evaluation.
        """
        self._eval_model = self._model

        # Create a copy of the configuration for eval MCTS and set a specific number of simulations.
        mcts_eval_cfg = copy.deepcopy(self._cfg)
        mcts_eval_cfg.num_simulations = self._cfg.eval_num_simulations

        if self._cfg.mcts_ctree:
            self._mcts_eval = MCTSCtree(mcts_eval_cfg)
        else:
            self._mcts_eval = MCTSPtree(mcts_eval_cfg)

        self.evaluator_env_num = self._cfg.evaluator_env_num

        if self._cfg.model.model_type == 'conv':
            self.last_batch_obs = torch.zeros([self.evaluator_env_num, self._cfg.model.observation_shape[0], 64, 64]).to(self._cfg.device)
            self.last_batch_action = [-1 for _ in range(self.evaluator_env_num)]
        elif self._cfg.model.model_type == 'mlp':
            self.last_batch_obs = torch.zeros([self.evaluator_env_num, self._cfg.model.observation_shape]).to(self._cfg.device)
            self.last_batch_action = [-1 for _ in range(self.evaluator_env_num)]

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: int = -1,
                      ready_env_id: np.array = None, timestep: List = [0], task_id: int = None) -> Dict:
        """
        Overview:
            The forward function for evaluating the policy. It uses the model to perform MCTS search and
            selects actions deterministically (choosing the one with the highest visit count).
        Arguments:
            - data (:obj:`torch.Tensor`): The input data, i.e., the current observation.
            - action_mask (:obj:`list`): A list of action masks for each environment.
            - to_play (:obj:`int`, optional): The player ID for the current turn.
            - ready_env_id (:obj:`np.array`, optional): An array of IDs for environments that are ready for a new action.
            - timestep (:obj:`List`, optional): The current timestep in each environment.
            - task_id (:obj:`int`, optional): The ID of the task for the current environments.
        Returns:
            - output (:obj:`Dict`): A dictionary where keys are environment IDs and values are dictionaries
              containing the selected action and other MCTS statistics.
        """
        self._eval_model.eval()
        active_eval_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_eval_env_num)
        output = {i: None for i in ready_env_id}
        with torch.no_grad():
            network_output = self._eval_model.initial_inference(self.last_batch_obs_eval, self.last_batch_action, data, task_id=task_id)
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            pred_values = self.value_inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            # Core fix: C++ binding requires a list, even though it represents rewards in MuZero.
            reward_roots = reward_roots.detach().cpu().numpy().tolist() # TODO=============================


            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_eval_env_num)]
            if self._cfg.mcts_ctree:
                # C++ MCTS tree implementation.
                roots = MCTSCtree.roots(active_eval_env_num, legal_actions)
            else:
                # Python MCTS tree implementation.
                roots = MCTSPtree.roots(active_eval_env_num, legal_actions)

            # During evaluation, no noise is added to the root policy.
            roots.prepare_no_noise(reward_roots, policy_logits, to_play)
            self._mcts_eval.search(roots, self._eval_model, latent_state_roots, to_play,  timestep= timestep, task_id=task_id)

            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()

            batch_action = []

            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]

                # NOTE: `deterministic=True` means we select the action with the highest visit count (argmax)
                # rather than sampling, which is standard for evaluation.
                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                    distributions, temperature=1, deterministic=True
                )
                # Convert the index back to the action in the full action space.
                action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]

                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'predicted_value': pred_values[i],
                    'predicted_policy_logits': policy_logits[i],
                }
                batch_action.append(action)

            self.last_batch_obs_eval = data
            self.last_batch_action = batch_action

        return output

    def _reset_collect(self, env_id: int = None, current_steps: int = 0, reset_init_data: bool = True, task_id: int = None) -> None:
        """
        Overview:
            Resets the collection process for a specific environment or all environments.
            It can clear caches and reset initial data to ensure optimal performance and prevent state leakage.
        Arguments:
            - env_id (:obj:`int`, optional): The ID of the environment to reset. If None, the reset applies more broadly. Defaults to None.
            - current_steps (:obj:`int`, optional): The current step count in the environment, used to trigger periodic cache clearing. Defaults to 0.
            - reset_init_data (:obj:`bool`, optional): If True, resets the initial observation and action buffers. Defaults to True.
            - task_id (:obj:`int`, optional): The task ID, currently unused in this method. Defaults to None.
        """
        if reset_init_data:
            self.last_batch_obs = initialize_zeros_batch(
                self._cfg.model.observation_shape,
                self._cfg.collector_env_num,
                self._cfg.device
            )
            self.last_batch_action = [-1 for _ in range(self._cfg.collector_env_num)]

        # We must handle both single int and list of ints for env_id.
        if env_id is not None:
            if isinstance(env_id, int):
                env_ids_to_reset = [env_id]
            else: # Assumes it's a list
                env_ids_to_reset = env_id

            # The key condition: `current_steps` is None only on the end-of-episode reset call from the collector.
            if current_steps is None:
                world_model = self._collect_model.world_model
                for eid in env_ids_to_reset:
                    # Clear the specific environment's initial inference cache.
                    if eid < len(world_model.past_kv_cache_init_infer_envs):
                        world_model.past_kv_cache_init_infer_envs[eid].clear()

                    print(f'>>> [Collector] Cleared KV cache for env_id: {eid} at episode end.')


        # Determine the clear interval based on the environment's sample type.
        clear_interval = 2000 if getattr(self._cfg, 'sample_type', '') == 'episode' else self._cfg.game_segment_length

        # Clear caches periodically to manage memory.
        if current_steps is not None and current_steps % clear_interval == 0:

            print(f'clear_interval: {clear_interval}')

            # Clear various KV caches in the collect model's world model.
            world_model = self._collect_model.world_model
            for kv_cache_dict_env in world_model.past_kv_cache_init_infer_envs:
                kv_cache_dict_env.clear()
            world_model.past_kv_cache_recurrent_infer.clear()
            world_model.keys_values_wm_list.clear()

            # Free up unused GPU memory.
            torch.cuda.empty_cache()

            print(f'Collector: Caches cleared for collect_model at step {current_steps} for env {env_id}.')

            # TODO: Check if resetting the target model here is correct and necessary.
            self._reset_target_model()

    def _reset_target_model(self) -> None:
        """
        Overview:
            Resets the target model by clearing its internal caches. This is crucial for managing memory,
            especially when using transformer-based models with KV caching.
        """
        # Clear various KV caches in the target model's world model.
        world_model = self._target_model.world_model
        for kv_cache_dict_env in world_model.past_kv_cache_init_infer_envs:
            kv_cache_dict_env.clear()
        world_model.past_kv_cache_recurrent_infer.clear()
        world_model.keys_values_wm_list.clear()

        # Free up unused GPU memory.
        torch.cuda.empty_cache()
        print('Collector: Target model past_kv_cache cleared.')

    def _reset_eval(self, env_id: int = None, current_steps: int = 0, reset_init_data: bool = True, task_id: int = None) -> None:
        """
        Overview:
            Resets the evaluation process for a specific environment or all environments.
            Clears caches and resets initial data to ensure clean evaluation runs.
        Arguments:
            - env_id (:obj:`int`, optional): The ID of the environment to reset. Defaults to None.
            - current_steps (:obj:`int`, optional): The current step count, used for periodic cache clearing. Defaults to 0.
            - reset_init_data (:obj:`bool`, optional): If True, resets the initial observation and action buffers. Defaults to True.
            - task_id (:obj:`int`, optional): The task ID. Can be used to handle different observation shapes per task. Defaults to None.
        """
        if reset_init_data:
            self.last_batch_obs_eval = initialize_zeros_batch(
                self._cfg.model.observation_shape,
                self._cfg.evaluator_env_num,
                self._cfg.device
            )

            self.last_batch_action = [-1 for _ in range(self._cfg.evaluator_env_num)]


        # This logic handles the crucial end-of-episode cache clearing for evaluation.
        # The evaluator calls `_policy.reset([env_id])` when an episode is done.
        if env_id is not None:
            if isinstance(env_id, int):
                env_ids_to_reset = [env_id]
            else: # Assumes it's a list
                env_ids_to_reset = env_id

            # The key condition: `current_steps` is None only on the end-of-episode reset call from the evaluator.
            if current_steps is None:
                world_model = self._eval_model.world_model
                for eid in env_ids_to_reset:
                    # Clear the specific environment's initial inference cache.
                    if eid < len(world_model.past_kv_cache_init_infer_envs):
                        world_model.past_kv_cache_init_infer_envs[eid].clear()

                    print(f'>>> [Evaluator] Cleared KV cache for env_id: {eid} at episode end.')

                # The recurrent cache is global.
                world_model.past_kv_cache_recurrent_infer.clear()

                if hasattr(world_model, 'keys_values_wm_list'):
                    world_model.keys_values_wm_list.clear()

                torch.cuda.empty_cache()
                return

        # Determine the clear interval.
        clear_interval = 2000 if getattr(self._cfg, 'sample_type', '') == 'episode' else self._cfg.game_segment_length

        # Clear caches periodically.
        if current_steps is not None and current_steps % clear_interval == 0:

            print(f'clear_interval: {clear_interval}')

            # Clear various KV caches in the eval model's world model.
            world_model = self._eval_model.world_model
            for kv_cache_dict_env in world_model.past_kv_cache_init_infer_envs:
                kv_cache_dict_env.clear()
            world_model.past_kv_cache_recurrent_infer.clear()
            world_model.keys_values_wm_list.clear()

            # Free up unused GPU memory.
            torch.cuda.empty_cache()

            print(f'Evaluator: Caches cleared for eval_model at step {current_steps} for env {env_id}.')


    def recompute_pos_emb_diff_and_clear_cache(self) -> None:
        """
        Overview:
            Clears all KV caches and precomputes positional embedding matrices in the model.
            This is typically called when the maximum sequence length changes.
        """
        # NOTE: This must be done for both the collect and target models.
        for model in [self._collect_model, self._target_model]:
            model.world_model.precompute_pos_emb_diff_kv()
            model.world_model.clear_caches()
        torch.cuda.empty_cache()

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Returns the state dictionary of the learn mode.
            This typically includes the model, target model, and optimizer states,
            which are necessary for saving and resuming training.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The state dictionary for the current learning progress.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_world_model': self._optimizer_world_model.state_dict(),
        }

    # ========== NOTE: This is a pretrain-finetune version that selectively loads parameters and freezes layers. ==========
    def _load_state_dict_learn(self, state_dict: Dict[str, Any], finetune_components: List[str] = []) -> None:
        """
        Overview:
            Loads a state_dict for fine-tuning. It excludes multi-task specific parameters
            and can freeze parts of the model (e.g., encoder, transformer) based on `finetune_components`.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The state dictionary from a pre-trained model.
            - finetune_components (:obj:`List[str]`, optional): A list of component names (e.g., "encoder", "transformer")
              that will remain trainable. Components not in this list will have their parameters frozen.
        """
        # Example configurations for fine-tuning:
        # finetune_components = []  # Loads encoder & transformer, fine-tunes only heads.
        # finetune_components = ['transformer'] # Loads encoder & transformer, fine-tunes transformer & heads.
        finetune_components = ["representation_network", "encoder"] # Loads encoder & transformer, fine-tunes encoder & heads.

        # Define prefixes of parameters to be excluded from loading (typically multi-task heads).
        exclude_prefixes = [
            '_orig_mod.world_model.head_policy_multi_task.',
            '_orig_mod.world_model.head_value_multi_task.',
            '_orig_mod.world_model.head_rewards_multi_task.',
            '_orig_mod.world_model.head_observations_multi_task.',
            '_orig_mod.world_model.task_emb.'
        ]

        # Define specific parameter keys to be excluded (for special cases like task embeddings).
        exclude_keys = [
            '_orig_mod.world_model.task_emb.weight',
            '_orig_mod.world_model.task_emb.bias',
        ]

        def filter_state_dict(state_dict_loader: Dict[str, Any], exclude_prefixes: list, exclude_keys: list = []) -> Dict[str, Any]:
            """
            Filters out parameters from a state_dict based on prefixes and specific keys.
            """
            filtered = {}
            for k, v in state_dict_loader.items():
                if any(k.startswith(prefix) for prefix in exclude_prefixes):
                    print(f"Excluding parameter: {k}")  # For debugging
                    continue
                if k in exclude_keys:
                    print(f"Excluding specific parameter: {k}")  # For debugging
                    continue
                filtered[k] = v
            return filtered

        # Filter and load the 'model' state_dict.
        if 'model' in state_dict:
            model_state_dict = state_dict['model']
            filtered_model_state_dict = filter_state_dict(model_state_dict, exclude_prefixes, exclude_keys)
            missing_keys, unexpected_keys = self._learn_model.load_state_dict(filtered_model_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys when loading _learn_model: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading _learn_model: {unexpected_keys}")
        else:
            print("No 'model' key found in the state_dict.")

        # Filter and load the 'target_model' state_dict.
        if 'target_model' in state_dict:
            target_model_state_dict = state_dict['target_model']
            filtered_target_model_state_dict = filter_state_dict(target_model_state_dict, exclude_prefixes, exclude_keys)
            missing_keys, unexpected_keys = self._target_model.load_state_dict(filtered_target_model_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys when loading _target_model: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading _target_model: {unexpected_keys}")
        else:
            print("No 'target_model' key found in the state_dict.")

        # Handle freezing/unfreezing of parameters in _learn_model based on finetune_components.
        # This assumes a naming convention where component names are present in parameter names.
        for name, param in self._learn_model.named_parameters():
            # Freeze the encoder if "encoder" is not in finetune_components.
            if "encoder" in name and "encoder" not in finetune_components:
                param.requires_grad = False
                print(f"Freezing parameter: {name}")
            # Freeze the representation network if "representation_network" is not in finetune_components.
            elif "representation_network" in name and "representation_network" not in finetune_components:
                param.requires_grad = False
                print(f"Freezing parameter: {name}")
            # Freeze the transformer if "transformer" is not in finetune_components.
            elif "transformer" in name and "transformer" not in finetune_components:
                param.requires_grad = False
                print(f"Freezing parameter: {name}")
            else:
                # Other parameters remain trainable by default.
                print(f"Parameter remains trainable: {name}")
