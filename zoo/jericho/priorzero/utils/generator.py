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
        all_full_texts = [c + l + self.tokenizer.eos_token for c, l in zip(all_context_texts, all_labels)]
        
        full_prompt_token_ids = self.tokenizer(all_full_texts, add_special_tokens=False, max_length=self.prompt_max_len + 1, padding=False, truncation=True)["input_ids"]
        context_token_ids = self.tokenizer(all_context_texts, add_special_tokens=False, max_length=self.prompt_max_len, padding=False, truncation=True)["input_ids"]
        
        prompt_lens = [len(x) for x in context_token_ids]
        label_lens = [len(full_ids) - p_len for full_ids, p_len in zip(full_prompt_token_ids, prompt_lens)]
    

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
