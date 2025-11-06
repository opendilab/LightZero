import multiprocessing


class PromptDataset:
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    # eot_text = EOT
    # bot_text = BOT

    # currently hack for llama 3.1
    start_header_text = "<|start_header_id|>"
    end_header_text = "<|end_header_id|>"
    eot_text = "<|eot_id|>"

    def __init__(
        self,
        dialogues,
        tokenizer: callable,
        max_length: int,
        strategy,
        pretrain_mode: bool = False,
        num_processors: int = 8,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length

        # Preprocess dialogues
        if num_processors < 2:
            self.dialogues = [self.process_dialogue(x) for x in dialogues]
        else:
            pool = multiprocessing.Pool(processes=num_processors)
            self.dialogues = pool.map(self.process_dialogue, dialogues)
            pool.close()
            pool.join()

    def process_dialogue(self, dialogue: dict):
        prompt_template = ""
        if self.tokenizer.bos_token_id is not None:
            prompt_template += f"{self.tokenizer.decode([self.tokenizer.bos_token_id])}"

        prompts = dialogue["prompt"]
        if prompts[-1]["role"] == "assistant":
            prompts = prompts[:-1]
        for message in prompts:
            prompt_template += f"{self.start_header_text}{message['role']}{self.end_header_text}\n{message['content']}{self.eot_text}\n"
        # append bot token
        prompt_template += f"{self.start_header_text}assistant{self.end_header_text}\n"

        extra = {key: value for key, value in dialogue.items() if key != "prompt"}

        return prompt_template, extra

    def collate_fn(self, item_list):
        all_inputs = []
        for prompt, extra in item_list:
            all_inputs.append((prompt, extra))
        return all_inputs

    def __getitem__(self, idx):
        inputs = self.dialogues[idx]
        return inputs

    def __len__(self):
        return len(self.dialogues)
