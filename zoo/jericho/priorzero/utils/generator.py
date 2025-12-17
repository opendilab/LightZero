from typing import List, Dict, Any, Optional, Tuple
import ray
import torch

class SamplesGenerator:
    def __init__(self, vllm_engines, strategy, tokenizer, prompt_max_len, temperature, top_p):
        self.strategy = strategy
        self.args = strategy.args
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.temperature = temperature
        self.top_p = top_p
    
    @torch.no_grad()
    def _build_cot_prefix_texts(self, all_prompts: List[str]) -> List[str]:
        """
        use_cot=True 时：
        1) 用原 prompt（chat_template 后的 context）让 vLLM 生成一次完整输出（包含推理 + action: <action>）
        2) 把“action: ”之前（包含 action: 和其后的空格）作为前缀拼回 prompt
        3) 返回新的 all_prompts（作为 user_prompt 传回 _generate_vllm，保持原流程不变）
        """
        from vllm import SamplingParams
        import re

        llms = self.vllm_engines

        cot_sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=self.prompt_max_len,
            include_stop_str_in_output=True,
            logprobs=None,
            prompt_logprobs=None,
        )

        all_context_texts = []
        for user_prompt in all_prompts:
            context_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            all_context_texts.append(context_text)

        context_token_ids = self.tokenizer(
            all_context_texts,
            add_special_tokens=False,
            max_length=self.prompt_max_len,
            padding=False,
            truncation=True,
        )["input_ids"]

        refs = []
        batch_size = (len(context_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            chunk = context_token_ids[i * batch_size: (i + 1) * batch_size]
            if len(chunk) > 0:
                refs.append(llm.add_requests.remote(sampling_params=cot_sampling_params, prompt_token_ids=chunk))
        ray.get(refs)

        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote())
        cot_outputs = sum(ray.get(all_output_refs), [])

        prefix_cot_list = []
        for user_prompt, output in zip(all_prompts, cot_outputs):
            gen_text = output.outputs[0].text

            matches = list(re.finditer(r"(?mi)^\s*Action\s*:\s*", gen_text))
            if not matches:
                matches = list(re.finditer(r"action\s*:\s*", gen_text, flags=re.IGNORECASE))

            if not matches:
                prefix_cot_list.append("")
                continue

            m = matches[-1]
            # prefix_piece = “推理 + action: ”（动作值之前）
            prefix_piece = gen_text[: m.end()].strip()

            prefix_cot_list.append(prefix_piece)

        return prefix_cot_list
    
    @torch.no_grad()
    def _generate_vllm(self, all_prompts: List[str], all_labels: List[str], reduction: str = "mean"):
        """Generate samples using vLLM engine.

        Args:
            all_prompts: List of prompts to generate from
            all_labels: List of labels corresponding to prompts
            **kwargs: Additional arguments for generation

        Returns:
            List of Experience objects containing generated samples
        """
        from vllm import SamplingParams
        assert reduction in ("mean", "sum")
        assert len(all_prompts) == len(all_labels)
        
        if self.args.use_cot:
            all_prefix_cot = self._build_cot_prefix_texts(all_prompts)
            
        llms = self.vllm_engines
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=1,
            include_stop_str_in_output=True,
            logprobs=None,
            prompt_logprobs=1
        )
        
        all_context_texts = []
        for user_prompt in all_prompts:
            context_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            all_context_texts.append(context_text)
        
        if self.args.use_cot:
            all_context_texts = [context + cot + " " for context, cot in zip(all_context_texts, all_prefix_cot)]

        context_token_ids = self.tokenizer(all_context_texts, add_special_tokens=False, max_length=self.prompt_max_len - 20, padding=False, truncation=True)["input_ids"]
        
        label_texts = [l + self.tokenizer.eos_token for l in all_labels]
        label_token_ids = self.tokenizer(label_texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        
        full_prompt_token_ids = [c + l for c, l in zip(context_token_ids, label_token_ids)]
        
        prompt_lens = [len(x) for x in context_token_ids]
        label_lens = [len(x) for x in label_token_ids]
    

        refs = []
        batch_size = (len(full_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            full_prompt_token = full_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            refs.append(llm.add_requests.remote(sampling_params=sampling_params, prompt_token_ids=full_prompt_token))
        ray.get(refs)

        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote())
        all_outputs = sum(ray.get(all_output_refs), [])

        scores = []
        for output, full_ids, p_len, l_len in zip(all_outputs, full_prompt_token_ids, prompt_lens, label_lens):
            prompt_logprobs = getattr(output, "prompt_logprobs", None)
            if prompt_logprobs is None:
                scores.append(float("-inf"))
                continue
            
            token_lps = []
            for idx in range(p_len, p_len + l_len):
                label_token_id = full_ids[idx]
                logprob_dict = prompt_logprobs[idx]

                token_lps.append(logprob_dict[label_token_id].logprob)

            if len(token_lps) == 0:
                scores.append(float("-inf"))
                continue
            if reduction == "sum":
                scores.append(sum(token_lps))
            else:
                scores.append(sum(token_lps) / len(token_lps))

        return scores
