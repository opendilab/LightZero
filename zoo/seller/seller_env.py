import os.path
from typing import List

import time
import numpy as np
import gym
from easydict import EasyDict
import jsonlines
import copy, csv
import ast
from ding.utils import ENV_REGISTRY
from zoo.seller.utils import APIClient, extract_json
from ding.envs import BaseEnv, BaseEnvTimestep

path_prefix = '/mnt/afs/niuyazhe/code/LightZero/'

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


class Commander(BaseRole):
    def call(self, history, retry=10):
        """
        Calls the model to generate commands and retries if parsing fails.
        
        Parameters:
        - history: The history data used to generate the query.
        - retry: Number of retry attempts in case of failure. Default is 3.
        
        Returns:
        - A list of parsed commands. If parsing fails after all retries, returns an empty list.
        """
        query = self.template.replace('{{history}}', str(history))
        response = self.model.generate(str([{'role': 'user', 'content': query}]), temperature=0.5)  # TODO: temperature

        for attempt in range(retry):
            try:
                # Try to parse the response as a Python list
                self.commands = ast.literal_eval(response.strip())
                # If parsing succeeds, return the commands
                return self.commands
            except (SyntaxError, ValueError):
                # Print error log and retry
                print(f"Attempt {attempt + 1}/{retry}: Failed to parse response as Python list: {response}")
                if attempt < retry - 1:
                    # If there are remaining retries, regenerate the response
                    response = self.model.generate(str([{'role': 'user', 'content': query}]), temperature=0.5)
                else:
                    # If max retries are reached, return an empty list
                    print(f"Max retry attempts reached. Returning empty command list.")
                    self.commands = []
                    return self.commands


class InputCommand:

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

        self.persona_info = None
        self.total_persona_num = cfg.get('total_persona_num', 10)

        self.good_info = None
        self.total_good_num = cfg.get('total_good_num', 20)
        self.train_good_num = cfg.get('train_good_num', 10)
        self.eval_good_num = cfg.get('eval_good_num', 10)

        # self.is_eval = cfg.get('is_eval', False)
        # print(f'is_eval: {self.is_eval}')

        if not (SellerEnv.executor and SellerEnv.judge and SellerEnv.buyer):
            self._init_roles()
        if not (SellerEnv.personas and SellerEnv.goods):
            self._init_settings()
        self.history = []  # TODO: for default_collate
        self.commands = cfg.commands
        self.dynamic_action_space = cfg.get('dynamic_action_space', False)

        self.max_round = cfg.max_round
        self.round_cnt = 0

        self.finished = False
        self._init_flag = False

        self.observation_space = gym.spaces.Dict()
        if self.dynamic_action_space:
            self.action_space = gym.spaces.Discrete(5)  # TODO 
        else:
            self.action_space = gym.spaces.Discrete(len(self.commands))
        self.reward_space = gym.spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.int32)

        self._replay = ''
        self._suffix = cfg.get('log_suffix', '')
        self.save_replay = cfg.get('save_replay', False)

    def _init_settings(self):
        # Init the personas.
        with open(path_prefix+"zoo/seller/data/persona.jsonl", "r+", encoding="utf8") as f:
            cnt = 0
            for item in jsonlines.Reader(f):
                SellerEnv.personas.append(item['persona'])
                cnt += 1
                if cnt >= self.total_persona_num:
                    break

        # Init the descriptions to goods.
        with open(path_prefix+"zoo/seller/data/good.jsonl", "r+", encoding="utf8") as f:
            cnt = 0
            for item in jsonlines.Reader(f):
                new_item = {'title': item['title'], 'description': item['description']}
                SellerEnv.goods.append(str(new_item))
                cnt += 1
                if cnt >= self.total_good_num:
                    break

    def reset(self, history=[], round_cnt = 0, eval_episode_return=0, is_eval=False, seed=None):
        # for collect and eval env, not for mcts simulate_env!
        if round_cnt > 0:
            self.history = copy.deepcopy(history)  
            self.round_cnt = copy.deepcopy(round_cnt)
        else:
            self.round_cnt = 0
            self.history = []

        self.is_eval = is_eval
        if seed is not None:
            self.seed_for_persona = seed
            self.seed_for_goods = seed
        else:
            if not self.is_eval:
                self.seed_for_persona = np.random.randint(0, self.total_persona_num)
                # TODO: train on N goods, eval on 2N goods
                self.seed_for_goods = np.random.randint(0, self.train_good_num)
                # self.seed_for_goods = np.random.randint(0, self.train_good_num + self.eval_good_num)

        print(f'======= reset, is_eval: {self.is_eval} ======= ')
        print(f'current seed for goods: {self.seed_for_goods}, ')
        print(f'current seed for persona: {self.seed_for_persona}')

        self.eval_episode_return = copy.deepcopy(eval_episode_return)
        self.finished = self.round_cnt >= self.max_round

        self._init_flag = True
        self._replay = ''
        self._replay_csv = []

        if self.dynamic_action_space:
            self.commands = SellerEnv.commander.call(history=self.history)

        self.action_mask = np.ones(len(self.commands), 'int8')
        self.legal_actions = np.arange(len(self.commands))

        obs = {'observation': self.history, 'action_mask': self.action_mask, 'round_cnt': self.round_cnt, 'eval_episode_return': self.eval_episode_return, 'seed_for_goods':self.seed_for_goods, 'seed_for_persona':self.seed_for_persona}

        self.persona_info = SellerEnv.personas[self.seed_for_persona % self.total_persona_num]
        self.good_info = SellerEnv.goods[self.seed_for_goods % self.total_good_num]


        return obs
    
    def reset_from_history(self, history, round_cnt, eval_episode_return=0, seed_for_goods=0, seed_for_persona=0, replay='', replay_csv=[]):
        # for MCTS and alphazero: simulation_env
        # NOTE
        self.save_replay = False
        
        self.seed_for_goods = seed_for_goods
        self.seed_for_persona = seed_for_persona
        self.persona_info = SellerEnv.personas[self.seed_for_persona % self.total_persona_num]
        self.good_info = SellerEnv.goods[self.seed_for_goods % self.total_good_num]

        self.history = copy.deepcopy(history)  
        self.round_cnt = copy.deepcopy(round_cnt)
        self.eval_episode_return = copy.deepcopy(eval_episode_return)
        self.finished = self.round_cnt >= self.max_round

        if self.dynamic_action_space:
            self.commands = SellerEnv.commander.call(history=self.history)

        # print(f'======= reset_from_history: is_eval: {self.is_eval}, is_simulation_env: True =======')
        # print(f' simulation_env reset, current seed for goods: {self.seed_for_goods}, ')
        # print(f' simulation_env reset, current seed for persona: {self.seed_for_persona}, ')
        
        self._replay = replay
        self._replay_csv = replay_csv

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the seed for the environment.
        """
        self._seed = seed
        self.seed_for_persona = self._seed
        self.seed_for_goods = self._seed
        if not dynamic_seed:
            np.random.seed(self._seed)
        else:
            np.random.seed(100 * np.random.randint(1, 1000))

    def _init_roles(self):
        SellerEnv.executor = Executor(
            agent=self.cfg.agent,
            api_key=self.cfg.api_key,
            template_path=f'zoo/seller/prompt_templates/executor_template_{self.lang}.txt'
        )
        SellerEnv.judge = Judge(
            agent=self.cfg.agent,
            api_key=self.cfg.api_key,
            template_path=f'zoo/seller/prompt_templates/judge_template_{self.lang}.txt'
        )
        SellerEnv.buyer = Buyer(
            agent=self.cfg.agent,
            api_key=self.cfg.api_key,
            template_path=f'zoo/seller/prompt_templates/buyer_template_{self.lang}.txt'
        )
        SellerEnv.commander = Commander(
            agent=self.cfg.agent,
            api_key=self.cfg.api_key,
            template_path=f'zoo/seller/prompt_templates/commander_template_{self.lang}.txt'
        )

    def close(self) -> None:
        self._init_flag = False

    def __repr__(self) -> str:
        return "LightZero SELLER Env"

    def step(self, action):
        if isinstance(action, int):
            command = self.commands[action]
        elif isinstance(action, list):
            command = self.commands[action[0]]
        else:
            command = self.commands[int(action.item())]
        self.round_cnt += 1
        
        executor_resp = SellerEnv.executor.call(command=command, history=self.history, info=self.good_info)
        role = '卖家' if self.lang == 'zh' else 'seller'
        # self.history.append({role: executor_resp})
        self.history.append({"role": role, "round": self.round_cnt, "content": executor_resp})

        role = '买家' if self.lang == 'zh' else 'customer'
        buyer_resp = SellerEnv.buyer.call(self.history, info=self.persona_info)
        # self.history.append({role: buyer_resp})
        self.history.append({"role": role, "round": self.round_cnt, "content": buyer_resp})

        judge_resp = SellerEnv.judge.call(self.history)

        rew = 0
        success_flag = '决定购买' if self.lang == 'zh' else 'Purchase'
        fail_flag = '拒绝购买' if self.lang == 'zh' else 'Refuse'
        try:
            extracted_judge = extract_json(judge_resp)
            if extracted_judge['评估结论'] == success_flag:
                rew = 1
                self.finished = True
            elif extracted_judge['评估结论'] == fail_flag:
                rew = -1
                self.finished = True
        except:
            pass

        if self.round_cnt >= self.max_round:
            self.finished = True
            if rew == 0:
                rew = -1

        self.eval_episode_return += rew

        # obs = {'observation': str(self.history), 'candidate_samples': self.commands}
        action_mask = np.ones(len(self.commands), 'int8')
        # obs = {'observation': str(self.history), 'action_mask': action_mask}
        obs = {'observation': self.history, 'action_mask': action_mask, 'round_cnt': self.round_cnt, 'eval_episode_return': self.eval_episode_return, 'seed_for_goods':self.seed_for_goods, 'seed_for_persona':self.seed_for_persona}

        env_step = BaseEnvTimestep(
            obs, rew, self.finished, {
                'command': command,
                'executor': executor_resp,
                'buyer': buyer_resp,
                'judge': judge_resp
            }
        )
        if self.finished:
            env_step.info['eval_episode_return'] = self.eval_episode_return
        # print(f'self.history: {self.history}')

        if self.save_replay:
            if self.round_cnt == 1:
                self._replay += f'【产品信息】： {self.good_info}\n'
                self._replay += f'【个性信息】： {self.persona_info}\n'
                self._replay_csv.append([f'【卖家产品信息】', f'{self.good_info}'])
                self._replay_csv.append([f'【买家个性信息】', f'{self.persona_info}'])

            self._replay += f'########## Round {self.round_cnt} ##########\n'
            self._replay += f'【动作序号】 {action}\n'
            # self._replay_csv.append([f'【动作序号】 {action}'])

            for kk in env_step.info:
                if kk == 'eval_episode_return':
                    continue
                self._replay += f'【{kk} 的回复】\n'
                self._replay += env_step.info[kk] + '\n'
                self._replay_csv.append([f'【{kk} 的回复】', env_step.info[kk]])

            self._replay += f'【Round {self.round_cnt}: reward: {rew}, done: {self.finished}】\n'
            self._replay_csv.append([f'【Round {self.round_cnt}】', f'【reward: {rew}, done: {self.finished}】'])

            if self.finished:
                log_dir = f'./logs_{self._suffix}'
                
                # 创建目录（如果不存在）
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)
                
                # 生成基本的日志文件名
                base_filename = f'log_goods-{self.seed_for_goods}_persona-{self.seed_for_persona}_{self._suffix}'
                
                # 处理txt文件
                txt_filename = f'{log_dir}/{base_filename}.txt'
                if os.path.exists(txt_filename):
                    timestamp = time.strftime('%Y%m%d_%H%M%S')  # 获取当前时间戳
                    txt_filename = f'{log_dir}/{base_filename}_{timestamp}.txt'
                
                with open(txt_filename, 'w', encoding='utf-8') as f:
                    f.write(self._replay + '\n')
                
                # 处理csv文件
                csv_filename = f'{log_dir}/{base_filename}.csv'
                if os.path.exists(csv_filename):
                    # timestamp = time.strftime('%Y%m%d_%H%M%S')  # 再次获取当前时间戳
                    csv_filename = f'{log_dir}/{base_filename}_{timestamp}.csv'
                
                with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(self._replay_csv)


        if self.dynamic_action_space:
            self.commands = SellerEnv.commander.call(history=self.history)

        return env_step


    def get_done_winner(self):
        """
        Overview:
             Check if the game is over and who the winner is. Return 'done' and 'winner'.
        Returns:
            - outputs (:obj:`Tuple`): Tuple containing 'done' and 'winner',
                - if player 1 win,     'done' = True, 'winner' = 1
                - if player 2 win,     'done' = True, 'winner' = 2
                - if draw,             'done' = True, 'winner' = -1
                - if game is not over, 'done' = False, 'winner' = -1
        """
        # Convert NumPy arrays to nested tuples to make them hashable.
        return self.finished, -1

    def clone(self):
        env_clone = SellerEnv(self.cfg)
        env_clone.reset(history=copy.deepcopy(self.history), round_cnt = copy.deepcopy(self.round_cnt), eval_episode_return=self.eval_episode_return, is_simulation_env=True)
        return env_clone
    
    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_eval = False
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_eval = True
        return [cfg for _ in range(evaluator_env_num)]

if __name__ == '__main__':
    env_cfg = EasyDict(
        dict(
            # agent='deepseek',
            agent='lmdeploy',
            api_key=[
            'sk-f50d634a123f4c84bc08fa880387ff76', 'sk-f8e6d25f99e5434c9ebda6e447fa8a7a',
            'sk-d020afbebe1e4d1ba1db7d32700c068c', 'sk-514a633560104439a4324dc30deab907',
            # 'sk-c4a8fe52693a4aaab64e648c42f40be6', 'sk-7866ab6ea8ca408a91971ef18eed4b75',
        ],
            # commands=[
            #     '向用户问好', '介绍产品的简要情况', '根据用户的疑虑进一步解答', '询问用户最关心的产品要求', '和用户共情，从用户的角度解释选择的原因', '威胁用户，如果不买就打他',
            #     '询问用户的具体使用情景', '向用户表示不耐烦，让他尽快做出决定', '询问用户当前还有哪些疑虑'
            # ],
            commands=[
                '将你的产品推销给用户'
            ],
            max_round=5,
            seed=0,
            lang='zh',
            # log_suffix='direct_0911_3eps_qwen2', # TODO
            # log_suffix='eval_direct_0911_20eps_interlm', # TODO
            log_suffix='eval_direct_0911_20eps_qwen2', # TODO
            # log_suffix='random_0910_20eps', # TODO
            save_replay=True,  # TODO
            # save_replay=False,  # TODO
            # dynamic_action_space=True,
            dynamic_action_space=False,

        )
    )

    input_command = InputCommand()

    env = SellerEnv(cfg=env_cfg)


    eval_episodes = 20
    # eval_episodes = 2

    for seed in range(0, eval_episodes):
        env.seed(seed=seed, dynamic_seed=False)
        env.reset(is_eval=True) # NOTE
        while not env.finished:
            # ===== for human input command =====
            # print(f'commands: {env.commands}')
            # command = input_command.call()

            # === direct policy =====
            command = 0

            # === random policy =====
            # command = int(np.random.randint(0,9,1))

            env_step = env.step([command])
            print(f'########## Round {env.round_cnt} ##########')
            for k in env_step.info:
                print(f'【{k} 的回复】')
                print(env_step.info[k])
            print(f'【reward: {env_step.reward}, done: {env_step.done}】')

