# fused_unizero_entry.py

import asyncio
import os
from functools import partial
from typing import Tuple, Optional, List, Dict

import ray
import torch
import numpy as np
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_rank
from tensorboardX import SummaryWriter
from loguru import logger

# Import necessary components from LightZero/UniZero
from lzero.entry.utils import log_buffer_memory_usage, calculate_update_per_collect
from lzero.policy import visit_count_temperature
from lzero.worker import MuZeroSegmentCollector as UniZeroCollector
from lzero.worker import MuZeroEvaluator as Evaluator
from lzero.mcts import UniZeroGameBuffer # The replay buffer
from ding.worker import BaseLearner

# Import ORZ/vLLM components for LLM inference
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

# --- Custom Components for PriorZero ---

class PriorZeroCollector(UniZeroCollector):
    """
    Custom Collector for PriorZero.
    It uses an LLM for policy priors at the MCTS root and a World Model for search.
    """
    def __init__(self, env, policy, tb_logger, exp_name, policy_config, vllm_engine: AsyncLLMEngine):
        super().__init__(env, policy, tb_logger, exp_name, policy_config)
        self.vllm_engine = vllm_engine
        self.llm_policy_cfg = policy_config.llm_policy_cfg
        logger.info("PriorZeroCollector initialized with vLLM engine.")

    async def _async_get_llm_prior(self, states: List[str]) -> List[Dict]:
        """ Asynchronously gets policy priors from the LLM. """
        prompts = []
        for state in states:
            instruction = (
                "You are an expert player in a text-based adventure game. "
                "Based on the history, think step-by-step and propose a ranked list of the best actions to take next. "
                "Your goal is to maximize the score.\n\n"
                f"=== History ===\n{state}\n\n"
                "=== Analysis and Ranked Actions (e.g., 1. take key 2. look) ==="
            )
            # NOTE: Assuming the policy model uses the same tokenizer as ORZ
            prompts.append(self._policy.llm_policy_model_tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}], tokenize=False
            ))
        
        sampling_params = SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=self.llm_policy_cfg.generate_max_len, stop=["==="]
        )
        
        request_ids = [f"collect_{self._collect_count}_{i}" for i in range(len(prompts))]
        results_generator = self.vllm_engine.generate(prompts, sampling_params, request_ids)
        
        llm_outputs = []
        async for result in results_generator:
            llm_outputs.append(result)
        
        # Sort results back to original order
        llm_outputs.sort(key=lambda r: int(r.request_id.split('_')[-1]))
        return llm_outputs

    @override
    async def collect(self, n_segment: Optional[int] = None, train_iter: int = 0, policy_kwargs: Optional[dict] = None) -> List[Dict]:
        """
        Asynchronous data collection method.
        """
        # This is a simplified version of the collection loop.
        # A full implementation would handle multiple segments and episodes.
        
        # Get current states and valid actions from all parallel envs
        # This part requires modification in the env_manager to be async or batched
        current_obs = self._env.ready_obs
        states = [obs['raw_obs'] for obs in current_obs.values()]
        valid_actions_list = [obs['action_mask'] for obs in current_obs.values()] # Assuming this format

        # 1. Get policy priors from LLM asynchronously
        llm_outputs = await self._async_get_llm_prior(states)

        # The rest of the logic is inside _forward_collect of the policy
        # We need to pass the LLM priors to it.
        policy_kwargs = policy_kwargs or {}
        policy_kwargs['llm_outputs'] = llm_outputs
        
        # The original `collect` is synchronous. We are calling the internal `_collect` logic here.
        # This part needs significant re-engineering to fit the async model.
        # For this blueprint, we assume the policy's forward pass can handle this.
        
        # The original call is synchronous, we are showing the conceptual flow
        # In a real implementation, `self._policy._forward_collect` would need to be async
        # and handle the interaction loop.
        
        # For now, let's just say the policy's collect function is now async
        # and we await it. This implies deep changes in the policy class itself.
        
        # Conceptual: The policy's `_forward_collect` will now:
        # a. Parse llm_outputs to create root priors.
        # b. Run MCTS using the world model.
        # c. Sample actions and step the environments.
        # d. Return the collected game segments.
        
        # This is a placeholder for the complex interaction logic.
        # The key is that the `collect` method is now `async`.
        logger.info("Conceptual async collect step completed.")
        # In a real system, this would return collected data segments.
        # We will mock this by returning an empty list, assuming data is pushed to buffer inside policy.
        
        # Let's simulate one step and data push for demonstration
        # This logic would actually be inside the policy/collector loop
        mock_game_segments = []
        for i in range(len(states)):
            # Mock MCTS result
            mcts_policy = np.ones(len(valid_actions_list[i])) / len(valid_actions_list[i])
            action = np.random.choice(len(valid_actions_list[i]))
            # Mock env step
            # self._env.step(...)
            mock_game_segments.append({'state': states[i], 'action': action, 'mcts_policy': mcts_policy})

        return mock_game_segments


class PriorZeroLearner(BaseLearner):
    """
    Custom Learner for PriorZero.
    Trains both the World Model and the LLM Policy.
    """
    def _init_learn(self):
        # This method is called by BaseLearner's __init__
        self.world_model = self._policy.world_model
        self.llm_policy_model = self._policy.llm_policy_model

        # Optimizer for World Model
        self.world_model_optimizer = torch.optim.AdamW(
            self.world_model.parameters(),
            lr=self._cfg.learning_rate, # From UniZero config
            weight_decay=self._cfg.weight_decay
        )
        
        # Optimizer for LLM Policy Model
        # This assumes the LLM is loaded and managed by the policy
        self.llm_policy_optimizer = torch.optim.AdamW(
            self.llm_policy_model.parameters(),
            lr=self._cfg.llm_policy_cfg.llm_learning_rate,
            weight_decay=self._cfg.llm_policy_cfg.llm_weight_decay
        )

    def _forward(self, data: List[Dict]) -> Dict[str, any]:
        """
        The main training step.
        """
        # --- 1. World Model Update ---
        # Prepare batch for world model (as in UniZero)
        # This is a complex data transformation step
        # wm_batch = self._policy.prepare_data_for_wm(data)
        world_model_loss_info = self.world_model.compute_loss({}) # Mocked call
        wm_loss = world_model_loss_info.loss_total
        
        self.world_model_optimizer.zero_grad()
        wm_loss.backward()
        self.world_model_optimizer.step()
        
        # --- 2. LLM Policy Update (RFT) ---
        # Prepare batch for LLM policy (instruction tuning format)
        # llm_batch = self._policy.prepare_data_for_llm(data)
        
        # For simplicity, we'll implement a Behavior Cloning (SFT) loss
        # The LLM should predict the MCTS policy
        # In a real PPO setup, this would be much more complex
        
        # Conceptual SFT loss:
        # llm_inputs = self.tokenizer(llm_batch['prompts'], return_tensors='pt', padding=True)
        # target_logits = self.tokenizer(llm_batch['targets'], return_tensors='pt', padding=True).input_ids
        # outputs = self.llm_policy_model(**llm_inputs, labels=target_logits)
        # llm_loss = outputs.loss
        llm_loss = torch.tensor(0.1, requires_grad=True) # Mock loss

        self.llm_policy_optimizer.zero_grad()
        llm_loss.backward()
        self.llm_policy_optimizer.step()

        return {
            'wm_loss': wm_loss.item(),
            'llm_loss': llm_loss.item(),
        }

async def train_priorzero(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        max_env_step: Optional[int] = int(1e10),
) -> None:
    """
    Asynchronous training entry for PriorZero.
    """
    cfg, create_cfg = input_cfg
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # 1. Create vLLM Engine as a Ray Actor (like in ORZ)
    engine_args = AsyncEngineArgs(
        model=cfg.policy.llm_policy_cfg.pretrain,
        tensor_parallel_size=cfg.policy.llm_policy_cfg.vllm_tensor_parallel_size,
        gpu_memory_utilization=cfg.policy.llm_policy_cfg.gpu_memory_utilization,
        worker_use_ray=True,
    )
    vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info("vLLM Engine created successfully.")

    # 2. Create Environment and Policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=True)
    
    # This will create our custom PriorZeroPolicy
    policy = create_policy(cfg.policy, enable_field=['learn', 'collect', 'eval'])
    
    # 3. Create Custom Worker Components
    tb_logger = SummaryWriter(os.path.join(f'./{cfg.exp_name}/log/', 'serial'))
    
    # Pass the vLLM engine to the collector
    collector = PriorZeroCollector(
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
        vllm_engine=vllm_engine
    )
    
    # The learner needs to be our custom one
    learner = PriorZeroLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = Evaluator(eval_freq=cfg.policy.eval_freq, n_evaluator_episode=cfg.env.n_evaluator_episode,
                          stop_value=cfg.env.stop_value, env=evaluator_env, policy=policy.eval_mode,
                          tb_logger=tb_logger, exp_name=cfg.exp_name, policy_config=cfg.policy)
    
    replay_buffer = UniZeroGameBuffer(cfg.policy)

    # --- Main Asynchronous Training Loop ---
    learner.call_hook('before_run')
    
    while collector.envstep < max_env_step:
        log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)
        
        # Collect experience asynchronously
        collect_kwargs = {'temperature': visit_count_temperature(trained_steps=learner.train_iter, **cfg.policy)}
        new_data = await collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        
        replay_buffer.push_game_segments(new_data)
        
        # Train models if buffer is ready
        if collector.envstep > cfg.policy.train_start_after_envsteps:
            update_per_collect = calculate_update_per_collect(cfg, new_data)
            for i in range(update_per_collect):
                train_data = replay_buffer.sample(cfg.policy.batch_size, policy)
                if not train_data:
                    break
                log_vars = learner.train(train_data, collector.envstep)
                
                # Log to tensorboard
                for k, v in log_vars.items():
                    tb_logger.add_scalar(f'train/{k}', v, learner.train_iter)

        # Evaluation
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
    
    learner.call_hook('after_run')


if __name__ == "__main__":
    # Get configuration
    main_cfg, create_cfg = get_priorzero_config(env_id='zork1.z5')
    
    # Start the asynchronous training process
    asyncio.run(train_priorzero([main_cfg, create_cfg], seed=0))