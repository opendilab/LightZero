import os.path

import numpy as np
import gym
from easydict import EasyDict
import jsonlines
import copy, csv

from ding.utils import ENV_REGISTRY
from dizoo.seller.utils import APIClient, extract_json
from ding.envs import BaseEnv, BaseEnvTimestep

path_prefix = '/mnt/miaohua/niuyazhe/code/RolePlay/'

class BaseRole:

    def __init__(self, agent, api_key, template_path):
        self.model = APIClient(api_key=api_key, agent=agent)
        with open(path_prefix+template_path, encoding='utf-8') as f:
            self.template = f.read().strip()


class Buyer(BaseRole):

    def call(self, history, info):
        query = self.template.replace('{{history}}', str(history))
        query = query.replace('{{persona}}', str(info))
        return self.model.generate(str([{'role': 'user', 'content': query}]))


class Judge(BaseRole):

    def call(self, history):
        return self.model.generate(
            str([{
                'role': 'user',
                'content': self.template.replace('{{history}}', str(history))
            }])
        )


class Executor(BaseRole):

    def call(self, command, history, info):
        query = self.template.replace('{{history}}', str(history))
        query = query.replace('{{command}}', str(command))
        query = query.replace('{{good_info}}', info)
        return self.model.generate(str([{'role': 'user', 'content': query}]))


class Commander:

    def call(self):
        response = input('Please type in the command: ')
        return int(response)


@ENV_REGISTRY.register('seller')
class SellerEnv(BaseEnv):
    executor, judge, buyer = None, None, None
    personas, goods = [], []

    def __init__(self, cfg):
        self.cfg = cfg
        self.lang = cfg.get('lang', 'zh')
        assert self.lang in ['zh', 'en']
        self._seed = 0

        self.persona_info = None
        self.persona_num = cfg.get('persona_num', 6)
        self.good_info = None
        self.good_num = cfg.get('good_num', 6)

        if not (SellerEnv.executor and SellerEnv.judge and SellerEnv.buyer):
            self._init_roles()
        if not (SellerEnv.personas and SellerEnv.goods):
            self._init_settings()
        self.history = []
        self.commands = cfg.commands
        self.max_round = cfg.max_round
        self.round_cnt = 0

        self.finished = False
        self._init_flag = False

        self.observation_space = gym.spaces.Dict()
        self.action_space = gym.spaces.Discrete(len(self.commands))
        self.reward_space = gym.spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.int32)

        self._replay = ''
        self._suffix = cfg.get('log_suffix', '')


    def _init_settings(self):
        # Init the personas.
        with open(path_prefix+"dizoo/seller/data/persona.jsonl", "r+", encoding="utf8") as f:
            cnt = 0
            for item in jsonlines.Reader(f):
                SellerEnv.personas.append(item['persona'])
                cnt += 1
                if cnt >= self.persona_num:
                    break

        # Init the descriptions to goods.
        with open(path_prefix+"dizoo/seller/data/good.jsonl", "r+", encoding="utf8") as f:
            cnt = 0
            for item in jsonlines.Reader(f):
                new_item = {'title': item['title'], 'description': item['description']}
                SellerEnv.goods.append(str(new_item))
                cnt += 1
                if cnt >= self.good_num:
                    break

    def reset(self):
        self.history = []
        self.round_cnt = 0
        self.finished = False
        self._init_flag = True
        self._replay = ''
        self._replay_csv = []
        obs = {'train_sample': str(self.history), 'candidate_samples': str(self.commands)}

        self.persona_info = SellerEnv.personas[self._seed % self.persona_num]
        self.good_info = SellerEnv.goods[self._seed % self.good_num]

        self.eval_episode_return = None

        return obs
    
    def reset_from_history(self, history, round_cnt, replay, replay_csv):
        self.history = copy.deepcopy(history)  
        self.round_cnt = copy.deepcopy(round_cnt)
        self.finished = False
        # self.replay = replay
        # self.replay_csv = replay_csv

    def seed(self, seed: int, dynamic_seed: bool = False) -> None:
        self._seed = seed

    def _init_roles(self):
        SellerEnv.executor = Executor(
            agent=self.cfg.agent,
            api_key=self.cfg.api_key,
            template_path=f'dizoo/seller/prompt_templates/executor_template_{self.lang}.txt'
        )
        SellerEnv.judge = Judge(
            agent=self.cfg.agent,
            api_key=self.cfg.api_key,
            template_path=f'dizoo/seller/prompt_templates/judge_template_{self.lang}.txt'
        )
        SellerEnv.buyer = Buyer(
            agent=self.cfg.agent,
            api_key=self.cfg.api_key,
            template_path=f'dizoo/seller/prompt_templates/buyer_template_{self.lang}.txt'
        )

    def close(self) -> None:
        self._init_flag = False

    def __repr__(self) -> str:
        return "DI-engine SELLER Env"

    def step(self, action):
        command = self.commands[action[0]]
        self.round_cnt += 1
        executor_resp = SellerEnv.executor.call(command=command, history=self.history, info=self.good_info)
        role = '卖家' if self.lang == 'zh' else 'seller'
        self.history.append({role: executor_resp})

        role = '买家' if self.lang == 'zh' else 'customer'
        buyer_resp = SellerEnv.buyer.call(self.history, info=self.persona_info)
        self.history.append({role: buyer_resp})

        judge_resp = SellerEnv.judge.call(self.history)

        rew = 0
        success_flag = '愿意购买' if self.lang == 'zh' else 'Purchase'
        fail_flag = '拒绝购买' if self.lang == 'zh' else 'Refuse'
        try:
            extracted_judge = extract_json(judge_resp)
            if extracted_judge['decision'] == success_flag:
                rew = 1
                self.finished = True
            elif extracted_judge['decision'] == fail_flag:
                rew = -1
                self.finished = True
        except:
            pass

        if self.round_cnt >= self.max_round:
            self.finished = True
            if rew == 0:
                rew = -1

        if self.finished:
            self.eval_episode_return = rew

        obs = {'train_sample': str(self.history), 'candidate_samples': self.commands}
        env_step = BaseEnvTimestep(
            obs, rew, self.finished, {
                'command': command,
                'executor': executor_resp,
                'buyer': buyer_resp,
                'judge': judge_resp
            }
        )
        # print(f'self.history: {self.history}')

        if self.round_cnt == 1:
            self._replay += f'【产品信息】： {self.good_info}\n'
            self._replay += f'【个性信息】： {self.persona_info}\n'
            self._replay_csv.append([f'【卖家产品信息】： {self.good_info}'])
            self._replay_csv.append([f'【买家个性信息】： {self.persona_info}'])

        self._replay += f'########## Round {self.round_cnt} ##########\n'
        self._replay += f'【动作序号】 {action}\n'
        # self._replay_csv.append([f'【动作序号】 {action}'])

        for kk in env_step.info:
            self._replay += f'【{kk} 的回复】\n'
            self._replay += env_step.info[kk] + '\n'
            self._replay_csv.append([f'【{kk} 的回复】', env_step.info[kk]])

        self._replay += f'【reward: {rew}, done: {self.finished}】\n'
        self._replay_csv.append([f'【reward: {rew}, done: {self.finished}】'])

        if rew != 0:
            if not os.path.exists('./logs'):
                os.mkdir('./logs')
            env_step.info['eval_episode_return'] = rew
            with open(f'./logs/evaluate_log_{self._seed}_{self._suffix}.txt', 'w', encoding='utf-8') as f:
                f.write(self._replay + '\n')
            with open(f'./logs/evaluate_log_{self._seed}_{self._suffix}.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(self._replay_csv)

        return env_step


    def copy(self):
        env_copy = SellerEnv(self.cfg)
        env_copy._seed = self._seed
        env_copy.persona_info = self.persona_info
        env_copy.good_info = self.good_info
        env_copy.history = copy.deepcopy(self.history)
        env_copy.round_cnt = self.round_cnt
        env_copy.finished = self.finished
        env_copy.eval_episode_return = self.eval_episode_return
        env_copy._init_flag = self._init_flag
        env_copy._replay = self._replay
        return env_copy

if __name__ == '__main__':
    env_cfg = EasyDict(
        dict(
            agent='deepseek',
            api_key='sk-7866ab6ea8ca408a91971ef18eed4b75',
            commands=[
                '向用户问好', '介绍产品的简要情况', '根据用户的疑虑进一步解答', '询问用户最关心的产品要求', '和用户共情，从用户的角度解释选择的原因', '威胁用户，如果不买就打他',
                '询问用户的具体使用情景', '向用户表示不耐烦，让他尽快做出决定', '询问用户当前还有哪些疑虑'
            ],
            # max_round=5,
            max_round=2,
            seed=0,
            lang='zh',
            log_suffix='',
        )
    )

    env = SellerEnv(cfg=env_cfg)
    history = env.reset()

    commander = Commander()

    while not env.finished:
        print(f'Legal actions: {" ".join([str(i) + ": " + env.commands[i] for i in range(len(env.commands))])}')
        command = commander.call()
        env_step = env.step([command])
        print(f'########## Round {env.round_cnt} ##########')
        for k in env_step.info:
            print(f'【{k} 的回复】')
            print(env_step.info[k])
        print(f'【reward: {env_step.reward}, done: {env_step.done}】')
