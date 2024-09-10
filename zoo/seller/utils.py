import copy
import warnings
import time
import re
import json
from functools import lru_cache

import openai


class APIClient:

    def __init__(self, api_key, agent='gpt3.5'):
        self.seed = 42
        self.agent = agent
        self.api_key_pool = api_key
        self.cur_api_idx = 0

        self.default_generate_cfg = dict(temperature=0.9, top_p=0.7, frequency_penalty=0, presence_penalty=0, stop=None)
        if self.agent == 'gpt3.5':
            self.client = openai.AzureOpenAI(
                azure_endpoint="https://normanhus-canadaeast.openai.azure.com/",
                api_key=api_key[0],
                api_version="2024-02-15-preview"
            )
        elif self.agent == 'gpt4':
            self.client = openai.AzureOpenAI(
                azure_endpoint="https://normanhus-uksouth.openai.azure.com/",
                api_key=api_key[0],
                api_version="2024-02-15-preview",
            )
        elif self.agent == 'deepseek':
            self.client = openai.OpenAI(api_key=api_key[0], base_url="https://api.deepseek.com")
        else:
            raise ValueError()

    @lru_cache(50000)
    def generate(self, history, max_retry=10, **kwargs):
        history = eval(history)
        generate_cfg = copy.deepcopy(self.default_generate_cfg)
        generate_cfg.update(kwargs)

        for _ in range(len(self.api_key_pool)):
            for _ in range(max_retry):
                cur_api_key = self.api_key_pool[self.cur_api_idx]
                self.client.api_key = cur_api_key
                try:
                    if self.agent == 'gpt3.5':
                        response_i = self.client.chat.completions.create(
                            model="gpt-35-turbo-1106", messages=history, **generate_cfg
                        )
                    elif self.agent == 'gpt4':
                        response_i = self.client.chat.completions.create(
                            model="gpt-4", messages=history, **generate_cfg
                        )
                    elif self.agent == 'deepseek':
                        response_i = self.client.chat.completions.create(
                            model='deepseek-chat', messages=history, **generate_cfg
                        )
                    else:
                        raise ValueError

                    reply_i = response_i.choices[0].message.content.strip()
                    self.cur_api_idx = (self.cur_api_idx + 1) % len(self.api_key_pool)
                    return reply_i
                except Exception as e:
                    warnings.warn(e)
                    time.sleep(3)
                    continue
            self.cur_api_idx = (self.cur_api_idx + 1) % len(self.api_key_pool)
            warnings.warn(f'API-key at index {self.cur_api_idx} seems to be wrong, skip ...')

        assert False, 'All api-keys fail, something wrong ...'

        return None


def extract_json(text):
    match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if match:
        json_str = match.group(1)
        return json.loads(json_str)
    else:
        return None
