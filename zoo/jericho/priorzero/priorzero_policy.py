# priorzero_policy.py
import copy
import re
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
from ding.utils import POLICY_REGISTRY
from ding.model import model_wrap
from transformers import AutoTokenizer, AutoModelForCausalLM

# 从您提供的原始 unizero_policy.py 中导入大部分内容
from lzero.policy.unizero import UniZeroPolicy as OriginalUniZeroPolicy
from lzero.policy import phi_transform, InverseScalarTransform, to_torch_float_tensor
from lzero.mcts import UniZeroMCTSCtree as MCTSCtree

# 辅助函数
def parse_llm_action_ranking(text: str, action_map: Dict[str, int], action_space_size: int) -> np.ndarray:
    """
    [PRIORZERO-NEW]
    解析LLM生成的有序动作列表，并将其转换为策略向量。
    例如，解析 "1. take key\n2. go north"
    """
    # 这是一个简化的实现，实际需要更鲁棒的正则表达式
    ranked_actions = re.findall(r'\d+\.\s*(.+)', text)
    
    policy = np.zeros(action_space_size, dtype=np.float32)
    found_actions = 0
    
    for action_text in ranked_actions:
        action_text = action_text.strip().lower()
        if action_text in action_map:
            action_idx = action_map[action_text]
            # 分配一个递减的权重，这里使用简单的逆序排名
            policy[action_idx] = len(ranked_actions) - found_actions
            found_actions += 1
            
    if policy.sum() > 0:
        policy /= policy.sum() # 归一化为概率分布
    else:
        # 如果LLM没有生成任何有效动作，返回均匀分布
        policy = np.ones(action_space_size, dtype=np.float32) / action_space_size
        
    return policy

def format_mcts_policy_to_text(mcts_policy: np.ndarray, action_inv_map: Dict[int, str]) -> str:
    """
    [PRIORZERO-NEW]
    将MCTS策略向量转换为有序的文本列表，作为SFT的训练目标。
    """
    sorted_indices = np.argsort(mcts_policy)[::-1] # 降序排序
    output_lines = []
    rank = 1
    for idx in sorted_indices:
        if mcts_policy[idx] > 0: # 只包括有概率的动作
            action_text = action_inv_map.get(idx, f"action_{idx}")
            output_lines.append(f"{rank}. {action_text}")
            rank += 1
            if rank > 5: # 最多只取 top 5
                break
    return "\n".join(output_lines)


@POLICY_REGISTRY.register('priorzero')
class PriorZeroPolicy(OriginalUniZeroPolicy):
    """
    [PRIORZERO-MODIFIED]
    融合了LLM先验和UniZero世界模型的策略。
    """

    def __init__(self, cfg: Dict, model: torch.nn.Module = None, enable_field: List[str] = None):
        super().__init__(cfg, model, enable_field)
        # [PRIORZERO-NEW] LLM相关组件将在 _init_learn 中初始化
        self.llm_policy_model = None
        self.llm_tokenizer = None
        self._optimizer_llm = None
        self.llm_policy_cfg = self._cfg.llm_policy_cfg
        
    def _init_learn(self) -> None:
        """
        [PRIORZERO-MODIFIED]
        初始化两个模型和两个优化器。
        """
        # 1. 初始化 UniZero 世界模型和其优化器 (复用原始逻辑)
        super()._init_learn()
        self._logger.info("UniZero World Model and its optimizer initialized.")

        # 2. [PRIORZERO-NEW] 初始化 LLM 策略模型和其优化器
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_policy_cfg.pretrain_llm_path, trust_remote_code=True)
        self.llm_policy_model = AutoModelForCausalLM.from_pretrained(
            self.llm_policy_cfg.pretrain_llm_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 # 使用 bfloat16 以节省显存
        ).to(self._cfg.device)
        self.llm_policy_model.train() # 设置为训练模式
        
        self._optimizer_llm = torch.optim.AdamW(
            self.llm_policy_model.parameters(),
            lr=self.llm_policy_cfg.llm_learning_rate,
            weight_decay=self.llm_policy_cfg.llm_weight_decay
        )
        self._logger.info(f"LLM Policy Model ({self.llm_policy_cfg.pretrain_llm_path}) and its optimizer initialized.")

    def _forward_learn(self, data: Tuple[torch.Tensor]) -> Dict[str, Union[float, int]]:
        """
        [PRIORZERO-MODIFIED]
        实现双模型训练的核心逻辑。
        """
        self._learn_model.train()
        self.llm_policy_model.train()

        current_batch, target_batch, train_iter, game_segments = data # 假设 data 包含 game_segments

        # ==============================================================
        # 1. UniZero 世界模型训练 (大部分复用)
        # ==============================================================
        obs_batch_ori, action_batch, _, mask_batch, _, weights, _, timestep_batch = current_batch
        target_reward, target_value, target_policy = target_batch
        
        # ... (此处省略了大量与原始 _forward_learn 相同的预处理代码)
        # ... (包括 prepare_obs, to_torch_float_tensor, scalar_transform, phi_transform 等)
        # ... (我们直接跳到 batch_for_gpt 的准备和损失计算)

        # 假设预处理已完成
        # ...
        data_list = [mask_batch, target_reward, target_value, target_policy, weights]
        mask_batch, target_reward, target_value, target_policy, weights = to_torch_float_tensor(data_list, self._cfg.device)
        target_reward = target_reward.view(self._cfg.batch_size, -1)
        target_value = target_value.view(self._cfg.batch_size, -1)
        transformed_target_reward = self.scalar_transform(target_reward)
        transformed_target_value = self.scalar_transform(target_value)
        target_reward_categorical = phi_transform(self.reward_support, transformed_target_reward)
        target_value_categorical = phi_transform(self.value_support, transformed_target_value)
        
        batch_for_gpt = {
            'observations': obs_batch_ori, # 假设已处理好
            'actions': action_batch.squeeze(-1),
            'timestep': timestep_batch.squeeze(-1),
            'rewards': target_reward_categorical[:, :-1],
            'mask_padding': (mask_batch == 1.0)[:, :-1],
            'target_value': target_value_categorical[:, :-1],
            'target_policy': target_policy[:, :-1],
        }
        # ... 更多 batch_for_gpt 的构建 ...

        wm_losses = self._learn_model.world_model.compute_loss(batch_for_gpt, self._target_model.world_model.tokenizer, self.value_inverse_scalar_transform_handle)
        wm_total_loss = (weights * wm_losses.loss_total).mean()

        # ==============================================================
        # 2. [PRIORZERO-NEW] LLM 策略模型训练 (RFT/SFT)
        # ==============================================================
        # 假设 game_segments 是一个包含我们自定义 GameSegment 对象的列表
        sft_prompts = []
        sft_targets = []
        
        # NOTE: 此处需要环境提供 action -> text 的映射
        # 我们假设在 self._cfg 中有这个映射
        action_inv_map = self._cfg.action_inv_map 

        for segment in game_segments:
            for i in range(len(segment.obs_segment)):
                if segment.mcts_policy_segment[i] is not None:
                    raw_obs_text = segment.obs_segment[i] # 假设 obs 就是原始文本
                    mcts_policy_vec = segment.mcts_policy_segment[i]
                    
                    # 构建 Prompt
                    instruction = (
                        "You are an expert player in a text-based adventure game. "
                        "Based on the history, think step-by-step and propose a ranked list of the best actions to take next. "
                        "Your goal is to maximize the score.\n\n"
                        f"=== History ===\n{raw_obs_text}\n\n"
                        "=== Analysis and Ranked Actions ==="
                    )
                    prompt = self.llm_tokenizer.apply_chat_template(
                        [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
                    )
                    
                    # 构建 Target
                    target_text = format_mcts_policy_to_text(mcts_policy_vec, action_inv_map)
                    
                    sft_prompts.append(prompt)
                    sft_targets.append(target_text)

        if len(sft_prompts) > 0:
            # Tokenize and train
            full_texts = [p + t + self.llm_tokenizer.eos_token for p, t in zip(sft_prompts, sft_targets)]
            inputs = self.llm_tokenizer(full_texts, padding=True, truncation=True, max_length=self.llm_policy_cfg.prompt_max_len, return_tensors="pt").to(self._cfg.device)
            
            # 创建 labels，将 prompt 部分的 token 设置为 -100 以忽略损失
            prompt_tokens = self.llm_tokenizer(sft_prompts, padding=True, truncation=True, max_length=self.llm_policy_cfg.prompt_max_len, return_tensors="pt")
            labels = inputs.input_ids.clone()
            labels[labels == self.llm_tokenizer.pad_token_id] = -100
            for i in range(len(sft_prompts)):
                prompt_len = prompt_tokens.attention_mask[i].sum()
                labels[i, :prompt_len] = -100

            llm_outputs = self.llm_policy_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
            llm_loss = llm_outputs.loss
        else:
            llm_loss = torch.tensor(0.0, device=self._cfg.device)

        # ==============================================================
        # 3. 联合优化
        # ==============================================================
        total_loss = wm_total_loss + self.llm_policy_cfg.llm_loss_weight * llm_loss
        
        # 优化世界模型
        self._optimizer_world_model.zero_grad()
        # 优化LLM
        self._optimizer_llm.zero_grad()
        
        total_loss.backward()
        
        # 裁剪梯度
        torch.nn.utils.clip_grad_norm_(self._learn_model.world_model.parameters(), self._cfg.grad_clip_value)
        torch.nn.utils.clip_grad_norm_(self.llm_policy_model.parameters(), self._cfg.grad_clip_value)

        self._optimizer_world_model.step()
        self._optimizer_llm.step()

        # 更新 target model
        self._target_model.update(self._learn_model.state_dict())

        # 日志记录
        log_dict = super()._forward_learn(data) # 调用父类获取基础日志
        log_dict['llm_loss'] = llm_loss.item()
        log_dict['total_loss_fused'] = total_loss.item()
        log_dict['wm_loss'] = wm_total_loss.item()

        return log_dict

    def _forward_collect(self, data: torch.Tensor, **kwargs) -> Dict:
        """
        [PRIORZERO-MODIFIED]
        使用 LLM 先验来指导 MCTS。
        """
        self._collect_model.eval()
        
        # [PRIORZERO-NEW] 从 kwargs 中获取 LLM 的输出
        llm_prior_outputs = kwargs.pop('llm_prior_outputs', None)
        action_mask = kwargs.get('action_mask')
        
        if llm_prior_outputs is None:
            # 如果没有LLM先验，退化为原始UniZero行为
            return super()._forward_collect(data, **kwargs)

        # NOTE: 此处需要环境提供 action -> text 的映射
        action_map = self._cfg.action_map

        # 解析 LLM 输出以获得策略先验
        policy_logits = []
        for output in llm_prior_outputs:
            generated_text = output.outputs[0].text
            prior_policy = parse_llm_action_ranking(generated_text, action_map, self._cfg.model.action_space_size)
            # 将概率转换为 logits
            policy_logits.append(torch.log(torch.from_numpy(prior_policy) + 1e-9))
        
        policy_logits = torch.stack(policy_logits).to(self._cfg.device)

        # 使用世界模型进行初始推理，但我们将覆盖其策略头输出
        with torch.no_grad():
            network_output = self._collect_model.initial_inference(data, **kwargs)
            latent_state_roots, reward_roots, pred_values, _ = self.mz_network_output_unpack(network_output)
            
            # [PRIORZERO-NEW] 使用 LLM 的输出覆盖原始策略 logits
            network_output.policy_logits = policy_logits

            # ... MCTS 搜索逻辑 (与原始 _forward_collect 基本相同) ...
            # 只是现在 MCTS 的根节点将使用来自 LLM 的先验
            # ...
            # (以下代码简化并改编自原始 _forward_collect)
            
            # MCTS搜索... (这部分逻辑与原始代码相同，使用被覆盖的 policy_logits)
            # ...
        
        # 这是一个简化的返回，实际实现应完整复用父类的 MCTS 逻辑
        # 这里仅为展示概念
        mock_output = {}
        for i, env_id in enumerate(kwargs['ready_env_id']):
             mock_output[env_id] = {
                'action': np.random.choice(np.where(action_mask[i] == 1.0)[0]),
                'visit_count_distributions': np.ones(self._cfg.model.action_space_size) / self._cfg.model.action_space_size,
                'visit_count_distribution_entropy': 1.0,
                'searched_value': 0.0,
                'predicted_value': 0.0,
             }
        return mock_output

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        [PRIORZERO-MODIFIED] 保存两个模型和优化器的状态。
        """
        state_dict = super()._state_dict_learn()
        state_dict['llm_model'] = self.llm_policy_model.state_dict()
        state_dict['optimizer_llm'] = self._optimizer_llm.state_dict()
        return state_dict

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        [PRIORZERO-MODIFIED] 加载两个模型和优化器的状态。
        """
        super()._load_state_dict_learn(state_dict)
        if 'llm_model' in state_dict:
            self.llm_policy_model.load_state_dict(state_dict['llm_model'])
        if 'optimizer_llm' in state_dict:
            self._optimizer_llm.load_state_dict(state_dict['optimizer_llm'])