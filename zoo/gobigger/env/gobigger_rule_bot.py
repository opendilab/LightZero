import copy
from ding.policy.base_policy import Policy
from ding.utils import POLICY_REGISTRY
import torch
import math
import queue
import random
from pygame.math import Vector2
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import namedtuple
from collections import defaultdict


@POLICY_REGISTRY.register('gobigger_bot')
class GoBiggerBot(Policy):

    def __init__(self, env_num, agent_id: List[int]):
        self.env_num = env_num
        self.agent_id = agent_id
        self.bot = [[BotAgent(i) for i in self.agent_id] for _ in range(self.env_num)]

    def forward(self, raw_obs):
        action = defaultdict(dict)
        for env_id in range(self.env_num):
            obs = raw_obs[env_id]
            for agent in self.bot[env_id]:
                action[env_id].update(agent.step(obs))
        return action

    def reset(self, env_id_lst=None):
        if env_id_lst is None:
            env_id_lst = range(self.env_num)
        for env_id in env_id_lst:
            for agent in self.bot[env_id]:
                agent.reset()

    # The following ensures compatibility with the DI-engine Policy class.
    def _init_learn(self) -> None:
        pass

    def _init_collect(self) -> None:
        pass

    def _init_eval(self) -> None:
        pass

    def _forward_learn(self, data: dict) -> dict:
        pass

    def _forward_collect(self, envs: Dict, obs: Dict, temperature: float = 1) -> Dict[str, torch.Tensor]:
        pass

    def _forward_eval(self, data: dict) -> dict:
        pass

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        pass

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        pass

    def default_model(self) -> Tuple[str, List[str]]:
        return 'bot_model', ['lzero.model.bot_model']

    def _monitor_vars_learn(self) -> List[str]:
        pass


class BotAgent():

    def __init__(self, game_player_id):
        self.game_player_id = game_player_id  # start from 0
        self.actions_queue = queue.Queue()

    def step(self, obs):
        obs = obs[1][self.game_player_id]
        if self.actions_queue.qsize() > 0:
            return {self.game_player_id: self.actions_queue.get()}
        overlap = obs['overlap']
        overlap = self.preprocess(overlap)
        food_balls = overlap['food']
        thorns_balls = overlap['thorns']
        spore_balls = overlap['spore']
        clone_balls = overlap['clone']

        my_clone_balls, others_clone_balls = self.process_clone_balls(clone_balls)

        if len(my_clone_balls) >= 9 and my_clone_balls[4]['radius'] > 4:
            self.actions_queue.put([0, 0, 0])
            self.actions_queue.put([0, 0, 0])
            self.actions_queue.put([0, 0, 0])
            self.actions_queue.put([0, 0, 0])
            self.actions_queue.put([0, 0, 0])
            self.actions_queue.put([0, 0, 0])
            self.actions_queue.put([0, 0, 0])
            self.actions_queue.put([None, None, 1])
            self.actions_queue.put([None, None, 1])
            self.actions_queue.put([None, None, 1])
            self.actions_queue.put([None, None, 1])
            self.actions_queue.put([None, None, 1])
            self.actions_queue.put([None, None, 1])
            self.actions_queue.put([None, None, 1])
            self.actions_queue.put([None, None, 1])
            action_ret = self.actions_queue.get()
            return {self.game_player_id: action_ret}

        if len(others_clone_balls) > 0 and self.can_eat(others_clone_balls[0]['radius'], my_clone_balls[0]['radius']):
            direction = (my_clone_balls[0]['position'] - others_clone_balls[0]['position'])
            action_type = 0
        else:
            min_distance, min_thorns_ball = self.process_thorns_balls(thorns_balls, my_clone_balls[0])
            if min_thorns_ball is not None:
                direction = (min_thorns_ball['position'] - my_clone_balls[0]['position'])
            else:
                min_distance, min_food_ball = self.process_food_balls(food_balls, my_clone_balls[0])
                if min_food_ball is not None:
                    direction = (min_food_ball['position'] - my_clone_balls[0]['position'])
                else:
                    direction = (Vector2(0, 0) - my_clone_balls[0]['position'])
            action_random = random.random()
            if action_random < 0.02:
                action_type = 1
            if action_random < 0.04 and action_random > 0.02:
                action_type = 2
            else:
                action_type = 0
        if direction.length() > 0:
            direction = direction.normalize()
        else:
            direction = Vector2(1, 1).normalize()
        direction = self.add_noise_to_direction(direction).normalize()
        self.actions_queue.put([direction.x, direction.y, action_type])
        action_ret = self.actions_queue.get()
        return {self.game_player_id: action_ret}

    def process_clone_balls(self, clone_balls):
        my_clone_balls = []
        others_clone_balls = []
        for clone_ball in clone_balls:
            if clone_ball['player'] == self.game_player_id:
                my_clone_balls.append(copy.deepcopy(clone_ball))
        my_clone_balls.sort(key=lambda a: a['radius'], reverse=True)

        for clone_ball in clone_balls:
            if clone_ball['player'] != self.game_player_id:
                others_clone_balls.append(copy.deepcopy(clone_ball))
        others_clone_balls.sort(key=lambda a: a['radius'], reverse=True)
        return my_clone_balls, others_clone_balls

    def process_thorns_balls(self, thorns_balls, my_max_clone_ball):
        min_distance = 10000
        min_thorns_ball = None
        for thorns_ball in thorns_balls:
            if self.can_eat(my_max_clone_ball['radius'], thorns_ball['radius']):
                distance = (thorns_ball['position'] - my_max_clone_ball['position']).length()
                if distance < min_distance:
                    min_distance = distance
                    min_thorns_ball = copy.deepcopy(thorns_ball)
        return min_distance, min_thorns_ball

    def process_food_balls(self, food_balls, my_max_clone_ball):
        min_distance = 10000
        min_food_ball = None
        for food_ball in food_balls:
            distance = (food_ball['position'] - my_max_clone_ball['position']).length()
            if distance < min_distance:
                min_distance = distance
                min_food_ball = copy.deepcopy(food_ball)
        return min_distance, min_food_ball

    def preprocess(self, overlap):
        new_overlap = {}
        for k, v in overlap.items():
            if k == 'clone':
                new_overlap[k] = []
                for index, vv in enumerate(v):
                    tmp = {}
                    tmp['position'] = Vector2(vv[0], vv[1])
                    tmp['radius'] = vv[2]
                    tmp['player'] = int(vv[-2])
                    tmp['team'] = int(vv[-1])
                    new_overlap[k].append(tmp)
            else:
                new_overlap[k] = []
                for index, vv in enumerate(v):
                    tmp = {}
                    tmp['position'] = Vector2(vv[0], vv[1])
                    tmp['radius'] = vv[2]
                    new_overlap[k].append(tmp)
        return new_overlap

    def preprocess_tuple2vector(self, overlap):
        new_overlap = {}
        for k, v in overlap.items():
            new_overlap[k] = []
            for index, vv in enumerate(v):
                new_overlap[k].append(vv)
                new_overlap[k][index]['position'] = Vector2(*vv['position'])
        return new_overlap

    def add_noise_to_direction(self, direction, noise_ratio=0.1):
        direction = direction + Vector2(
            ((random.random() * 2 - 1) * noise_ratio) * direction.x,
            ((random.random() * 2 - 1) * noise_ratio) * direction.y
        )
        return direction

    def radius_to_score(self, radius):
        return (math.pow(radius, 2) - 0.15) / 0.042 * 100

    def can_eat(self, radius1, radius2):
        return self.radius_to_score(radius1) > 1.3 * self.radius_to_score(radius2)

    def reset(self, ):
        self.actions_queue.queue.clear()
