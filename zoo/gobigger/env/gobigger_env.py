import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from gobigger.envs import GoBiggerEnv
import math


@ENV_REGISTRY.register('gobigger_lightzero')
class GoBiggerLightZeroEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        # ding env info
        self._init_flag = False
        self._observation_space = None
        self._action_space = None
        self._reward_space = None
        # gobigger env info
        self.team_num = self._cfg.team_num
        self.player_num_per_team = self._cfg.player_num_per_team
        self.direction_num = self._cfg.direction_num
        self.use_action_mask = self._cfg.use_action_mask
        self.action_space_size = self._cfg.action_space_size  # discrete action space size
        self.step_mul = self._cfg.get('step_mul', 8)
        self.setup_action()
        self.setup_feature()
        self.contain_raw_obs = self._cfg.contain_raw_obs  # for save memory

    def setup_feature(self):
        self.second_per_frame = 0.05
        self.spatial_x = 64
        self.spatial_y = 64
        self.max_ball_num = 80
        self.max_food_num = 256
        self.max_spore_num = 64
        self.max_player_num = self.player_num_per_team
        self.reward_div_value = self._cfg.reward_div_value
        self.reward_type = self._cfg.reward_type
        self.player_init_score = self._cfg.manager_settings.player_manager.ball_settings.score_init
        self.start_spirit_progress = self._cfg.start_spirit_progress
        self.end_spirit_progress = self._cfg.end_spirit_progress

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = GoBiggerEnv(self._cfg, step_mul=self.step_mul)
            self._init_flag = True
        self.last_action_types = {
            player_id: self.direction_num * 2
            for player_id in range(self.player_num_per_team * self.team_num)
        }
        raw_obs = self._env.reset()
        obs = self.observation(raw_obs)
        self.last_action_types = {
            player_id: self.direction_num * 2
            for player_id in range(self.player_num_per_team * self.team_num)
        }
        self.last_leaderboard = {
            team_idx: self.player_init_score * self.player_num_per_team
            for team_idx in range(self.team_num)
        }
        self.last_player_scores = {
            player_id: self.player_init_score
            for player_id in range(self.player_num_per_team * self.team_num)
        }
        return obs

    def observation(self, raw_obs):
        obs = self.preprocess_obs(raw_obs)
        # for alignment with other environments, reverse the action mask
        action_mask = [np.logical_not(o['action_mask']) for o in obs]
        to_play = [-1 for _ in range(len(obs))]  # Moot, for alignment with other environments
        if self.contain_raw_obs:
            obs = {'observation': obs, 'action_mask': action_mask, 'to_play': to_play, 'raw_obs': raw_obs}
        else:
            obs = {'observation': obs, 'action_mask': action_mask, 'to_play': to_play}
        return obs

    def postproecess(self, action_dict):
        for k, v in action_dict.items():
            if np.isscalar(v):
                self.last_action_types[k] = v
            else:
                self.last_action_types[k] = self.direction_num * 2

    def step(self, action_dict: dict) -> BaseEnvTimestep:
        action = {k: self.transform_action(v) if np.isscalar(v) else v for k, v in action_dict.items()}
        raw_obs, raw_rew, done, info = self._env.step(action)
        # print('current_frame={}'.format(raw_obs[0]['last_time']))
        # print('action={}'.format(action))
        # print('raw_rew={}, done={}'.format(raw_rew, done))
        rew = self.transform_reward(raw_obs)
        obs = self.observation(raw_obs)
        # postprocess
        self.postproecess(action_dict)
        if done:
            info['eval_episode_return'] = [raw_obs[0]['leaderboard'][i] for i in range(self.team_num)]
        return BaseEnvTimestep(obs, rew, done, info)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    @property
    def observation_space(self) -> gym.spaces.Space:
        # The following ensures compatibility with the DI-engine Env class.
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        # The following ensures compatibility with the DI-engine Env class.
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        # The following ensures compatibility with the DI-engine Env class.
        return self._reward_space

    def __repr__(self) -> str:
        return "LightZero Env({})".format(self.cfg.env_name)

    def transform_obs(
        self,
        obs,
        own_player_id=1,
        padding=True,
        last_action_type=None,
    ):
        global_state, player_observations = obs
        player2team = self.get_player2team()
        leaderboard = global_state['leaderboard']
        team2rank = {key: rank for rank, key in enumerate(sorted(leaderboard, key=leaderboard.get, reverse=True), )}

        own_player_obs = player_observations[own_player_id]
        own_team_id = player2team[own_player_id]

        # ===========
        # scalar info
        # ===========
        scene_size = global_state['border'][0]
        own_left_top_x, own_left_top_y, own_right_bottom_x, own_right_bottom_y = own_player_obs['rectangle']
        own_view_center = [
            (own_left_top_x + own_right_bottom_x - scene_size) / 2,
            (own_left_top_y + own_right_bottom_y - scene_size) / 2
        ]
        # own_view_width == own_view_height
        own_view_width = float(own_right_bottom_x - own_left_top_x)

        own_score = own_player_obs['score'] / 100
        own_team_score = global_state['leaderboard'][own_team_id] / 100
        own_rank = team2rank[own_team_id]

        scalar_info = {
            'view_x': np.round(np.array(own_view_center[0])).astype(np.int64),
            'view_y': np.round(np.array(own_view_center[1])).astype(np.int64),
            'view_width': np.round(np.array(own_view_width)).astype(np.int64),
            'score': np.clip(np.round(np.log(np.array(own_score) / 10)).astype(np.int64), a_min=None, a_max=9),
            'team_score': np.clip(
                np.round(np.log(np.array(own_team_score / 10))).astype(np.int64), a_min=None, a_max=9
            ),
            'time': np.array(global_state['last_time'] // 20, dtype=np.int64),
            'rank': np.array(own_rank, dtype=np.int64),
            'last_action_type': np.array(last_action_type, dtype=np.int64)
        }

        # ===========
        # team_info
        # ===========

        all_players = []
        scene_size = global_state['border'][0]

        for game_player_id in player_observations.keys():
            game_team_id = player2team[game_player_id]
            game_player_left_top_x, game_player_left_top_y, game_player_right_bottom_x, game_player_right_bottom_y = \
                player_observations[game_player_id]['rectangle']
            if game_player_id == own_player_id:
                alliance = 0
            elif game_team_id == own_team_id:
                alliance = 1
            else:
                alliance = 2
            if alliance != 2:
                game_player_view_x = (game_player_right_bottom_x + game_player_left_top_x - scene_size) / 2
                game_player_view_y = (game_player_right_bottom_y + game_player_left_top_y - scene_size) / 2

                all_players.append([
                    alliance,
                    game_player_view_x,
                    game_player_view_y,
                ])

        all_players = np.array(all_players)
        player_padding_num = self.max_player_num - len(all_players)
        player_num = len(all_players)
        if player_padding_num < 0:
            all_players = all_players[:self.max_player_num, :]
        else:
            all_players = np.pad(all_players, pad_width=((0, player_padding_num), (0, 0)), mode='constant')
        team_info = {
            'alliance': all_players[:, 0].astype(np.int64),
            'view_x': np.round(all_players[:, 1]).astype(np.int64),
            'view_y': np.round(all_players[:, 2]).astype(np.int64),
            'player_num': np.array(player_num, dtype=np.int64),
        }

        # ===========
        # ball info
        # ===========
        ball_type_map = {'clone': 1, 'food': 2, 'thorns': 3, 'spore': 4}
        clone = own_player_obs['overlap']['clone']
        thorns = own_player_obs['overlap']['thorns']
        food = own_player_obs['overlap']['food']
        spore = own_player_obs['overlap']['spore']

        neutral_team_id = self.team_num
        neutral_player_id = self.team_num * self.player_num_per_team
        neutral_team_rank = self.team_num

        # clone = [type, score, player_id, team_id, team_rank, x, y, next_x, next_y]
        clone = [
            [
                ball_type_map['clone'], bl[3], bl[-2], bl[-1], team2rank[bl[-1]], bl[0], bl[1],
                *self.next_position(bl[0], bl[1], bl[4], bl[5])
            ] for bl in clone
        ]

        # thorn = [type, score, player_id, team_id, team_rank, x, y, next_x, next_y]
        thorns = [
            [
                ball_type_map['thorns'], bl[3], neutral_player_id, neutral_team_id, neutral_team_rank, bl[0], bl[1],
                *self.next_position(bl[0], bl[1], bl[4], bl[5])
            ] for bl in thorns
        ]

        # thorn = [type, score, player_id, team_id, team_rank, x, y, next_x, next_y]
        food = [
            [
                ball_type_map['food'], bl[3], neutral_player_id, neutral_team_id, neutral_team_rank, bl[0], bl[1],
                bl[0], bl[1]
            ] for bl in food
        ]

        # spore = [type, score, player_id, team_id, team_rank, x, y, next_x, next_y]
        spore = [
            [
                ball_type_map['spore'], bl[3], bl[-1], player2team[bl[-1]], team2rank[player2team[bl[-1]]], bl[0],
                bl[1], *self.next_position(bl[0], bl[1], bl[4], bl[5])
            ] for bl in spore
        ]

        all_balls = clone + thorns + food + spore

        # Particularly handle balls outside the field of view
        for b in all_balls:
            if b[2] == own_player_id and b[0] == 1:
                if b[5] < own_left_top_x or b[5] > own_right_bottom_x or \
                        b[6] < own_left_top_y or b[6] > own_right_bottom_y:
                    b[5] = int((own_left_top_x + own_right_bottom_x) / 2)
                    b[6] = int((own_left_top_y + own_right_bottom_y) / 2)
                    b[7], b[8] = b[5], b[6]
        all_balls = np.array(all_balls)

        origin_x = own_left_top_x
        origin_y = own_left_top_y

        all_balls[:, -4] = ((all_balls[:, -4] - origin_x) / own_view_width * self.spatial_x)
        all_balls[:, -3] = ((all_balls[:, -3] - origin_y) / own_view_width * self.spatial_y)
        all_balls[:, -2] = ((all_balls[:, -2] - origin_x) / own_view_width * self.spatial_x)
        all_balls[:, -1] = ((all_balls[:, -1] - origin_y) / own_view_width * self.spatial_y)

        # ball
        ball_indices = np.logical_and(
            all_balls[:, 0] != 2, all_balls[:, 0] != 4
        )  # include player balls and thorn balls
        balls = all_balls[ball_indices]

        balls_num = len(balls)

        # consider position of thorns ball
        if balls_num > self.max_ball_num:  # filter small balls
            own_indices = balls[:, 3] == own_player_id
            teammate_indices = (balls[:, 4] == own_team_id) & ~own_indices
            enemy_indices = balls[:, 4] != own_team_id

            own_balls = balls[own_indices]
            teammate_balls = balls[teammate_indices]
            enemy_balls = balls[enemy_indices]

            if own_balls.shape[0] + teammate_balls.shape[0] >= self.max_ball_num:
                remain_ball_num = self.max_ball_num - own_balls.shape[0]
                teammate_ball_score = teammate_balls[:, 1]
                teammate_high_score_indices = teammate_ball_score.sort(descending=True)[1][:remain_ball_num]
                teammate_remain_balls = teammate_balls[teammate_high_score_indices]
                balls = np.concatenate([own_balls, teammate_remain_balls], axis=0)
            else:
                remain_ball_num = self.max_ball_num - own_balls.shape[0] - teammate_balls.shape[0]
                enemy_ball_score = enemy_balls[:, 1]
                enemy_high_score_ball_indices = enemy_ball_score.sort(descending=True)[1][:remain_ball_num]
                remain_enemy_balls = enemy_balls[enemy_high_score_ball_indices]

                balls = np.concatenate([own_balls, teammate_balls, remain_enemy_balls], axis=0)
        balls_num = len(balls)
        ball_padding_num = self.max_ball_num - len(balls)
        if ball_padding_num < 0:
            balls = balls[:self.max_ball_num, :]
            alliance = np.zeros(self.max_ball_num)
            balls_num = self.max_ball_num
        elif padding:
            balls = np.pad(balls, ((0, ball_padding_num), (0, 0)), 'constant', constant_values=0)
            alliance = np.zeros(self.max_ball_num)
            balls_num = min(self.max_ball_num, balls_num)
        else:
            alliance = np.zeros(balls_num)
        alliance[balls[:, 3] == own_team_id] = 2
        alliance[balls[:, 2] == own_player_id] = 1
        alliance[balls[:, 3] != own_team_id] = 3
        alliance[balls[:, 0] == 3] = 0

        ## score&radius
        scale_score = balls[:, 1] / 100
        radius = np.clip(np.sqrt(scale_score * 0.042 + 0.15) / own_view_width, a_max=1, a_min=None)
        score = np.clip(
            np.round(np.clip(np.sqrt(scale_score * 0.042 + 0.15) / own_view_width, a_max=1, a_min=None) * 50
                     ).astype(int),
            a_max=49,
            a_min=None
        )
        ## rank:
        ball_rank = balls[:, 4]

        ## coordinates relative to the center of [spatial_x, spatial_y]
        x = balls[:, -4] - self.spatial_x // 2
        y = balls[:, -3] - self.spatial_y // 2
        next_x = balls[:, -2] - self.spatial_x // 2
        next_y = balls[:, -1] - self.spatial_y // 2

        ball_info = {
            'alliance': alliance.astype(np.int64),
            'score': score.astype(np.int64),
            'radius': radius,
            'rank': ball_rank.astype(np.int64),
            'x': np.round(x).astype(np.int64),
            'y': np.round(y).astype(np.int64),
            'next_x': np.round(next_x).astype(np.int64),
            'next_y': np.round(next_y).astype(np.int64),
            'ball_num': np.array(balls_num).astype(np.int64),
        }

        # ============
        # spatial info
        # ============
        # ball coordinate for scatter connection
        # coordinates relative to the upper left corner of [spatial_x, spatial_y]
        ball_x = balls[:, -4]
        ball_y = balls[:, -3]

        food_indices = all_balls[:, 0] == 2
        food_x = all_balls[food_indices, -4]
        food_y = all_balls[food_indices, -3]
        food_num = len(food_x)
        food_padding_num = self.max_food_num - len(food_x)
        if food_padding_num < 0:
            food_x = food_x[:self.max_food_num]
            food_y = food_y[:self.max_food_num]
        elif padding:
            food_x = np.pad(food_x, (0, food_padding_num), 'constant', constant_values=0)
            food_y = np.pad(food_y, (0, food_padding_num), 'constant', constant_values=0)
        food_num = min(food_num, self.max_food_num)

        spore_indices = all_balls[:, 0] == 4
        spore_x = all_balls[spore_indices, -4]
        spore_y = all_balls[spore_indices, -3]
        spore_num = len(spore_x)
        spore_padding_num = self.max_spore_num - len(spore_x)
        if spore_padding_num < 0:
            spore_x = spore_x[:self.max_spore_num]
            spore_y = spore_y[:self.max_spore_num]
        elif padding:
            spore_x = np.pad(spore_x, (0, spore_padding_num), 'constant', constant_values=0)
            spore_y = np.pad(spore_y, (0, spore_padding_num), 'constant', constant_values=0)
        spore_num = min(spore_num, self.max_spore_num)

        spatial_info = {
            'food_x': np.clip(np.round(food_x), 0, self.spatial_x - 1).astype(np.int64),
            'food_y': np.clip(np.round(food_y), 0, self.spatial_y - 1).astype(np.int64),
            'spore_x': np.clip(np.round(spore_x), 0, self.spatial_x - 1).astype(np.int64),
            'spore_y': np.clip(np.round(spore_y), 0, self.spatial_y - 1).astype(np.int64),
            'ball_x': np.clip(np.round(ball_x), 0, self.spatial_x - 1).astype(np.int64),
            'ball_y': np.clip(np.round(ball_y), 0, self.spatial_y - 1).astype(np.int64),
            'food_num': np.array(food_num).astype(np.int64),
            'spore_num': np.array(spore_num).astype(np.int64),
        }

        output_obs = {
            'scalar_info': scalar_info,
            'team_info': team_info,
            'ball_info': ball_info,
            'spatial_info': spatial_info,
        }
        return output_obs

    def preprocess_obs(self, raw_obs):
        env_player_obs = []
        for game_player_id in range(self.player_num_per_team * self.team_num):
            last_action_type = self.last_action_types[game_player_id]
            if self.use_action_mask:
                can_eject = raw_obs[1][game_player_id]['can_eject']
                can_split = raw_obs[1][game_player_id]['can_split']
                action_mask = self.generate_action_mask(can_eject=can_eject, can_split=can_split)
            else:
                action_mask = self.generate_action_mask(can_eject=True, can_split=True)
            game_player_obs = self.transform_obs(
                raw_obs, own_player_id=game_player_id, padding=True, last_action_type=last_action_type
            )
            game_player_obs['action_mask'] = action_mask
            env_player_obs.append(game_player_obs)
        return env_player_obs

    def generate_action_mask(self, can_eject, can_split):
        # action mask
        # 1 represent can not do this action
        # 0 represent can do this action
        action_mask = np.zeros((self.action_space_size, ), dtype=np.bool_)
        if not can_eject:
            action_mask[self.direction_num * 2 + 1] = True
        if not can_split:
            action_mask[self.direction_num * 2 + 2] = True
        return action_mask

    def get_player2team(self, ):
        player2team = {}
        for player_id in range(self.player_num_per_team * self.team_num):
            player2team[player_id] = player_id // self.player_num_per_team
        return player2team

    def next_position(self, x, y, vel_x, vel_y):
        next_x = x + self.second_per_frame * vel_x * self.step_mul
        next_y = y + self.second_per_frame * vel_y * self.step_mul
        return next_x, next_y

    def transform_action(self, action_idx):
        return self.x_y_action_List[int(action_idx)]

    def setup_action(self):
        theta = math.pi * 2 / self.direction_num
        self.x_y_action_List = [[0.3 * math.cos(theta * i), 0.3 * math.sin(theta * i), 0] for i in range(self.direction_num)] + \
                               [[math.cos(theta * i), math.sin(theta * i), 0] for i in range(self.direction_num)] + \
                               [[0, 0, 0], [0, 0, 1], [0, 0, 2]]

    def get_spirit(self, progress):
        if progress < self.start_spirit_progress:
            return 0
        elif progress <= self.end_spirit_progress:
            spirit = (progress - self.start_spirit_progress) / (self.end_spirit_progress - self.start_spirit_progress)
            return spirit
        else:
            return 1

    def transform_reward(self, next_obs):
        last_time = next_obs[0]['last_time']
        total_frame = next_obs[0]['total_frame']
        progress = last_time / total_frame
        spirit = self.get_spirit(progress)
        score_rewards_list = []
        for game_player_id in range(self.player_num_per_team * self.team_num):
            game_team_id = game_player_id // self.player_num_per_team
            player_score = next_obs[1][game_player_id]['score']
            team_score = next_obs[0]['leaderboard'][game_team_id]
            if self.reward_type == 'log_reward':
                player_reward = math.log(player_score) - math.log(self.last_player_scores[game_player_id])
                team_reward = math.log(team_score) - math.log(self.last_leaderboard[game_team_id])
                score_reward = (1 - spirit) * player_reward + spirit * team_reward / self.player_num_per_team
                score_reward = score_reward / self.reward_div_value
                score_rewards_list.append(score_reward)
            elif self.reward_type == 'score':
                player_reward = player_score - self.last_player_scores[game_player_id]
                team_reward = team_score - self.last_leaderboard[game_team_id]
                score_reward = (1 - spirit) * player_reward + spirit * team_reward / self.player_num_per_team
                score_reward = score_reward / self.reward_div_value
                score_rewards_list.append(score_reward)
            elif self.reward_type == 'sqrt_player':
                player_reward = player_score - self.last_player_scores[game_player_id]
                reward_sign = (player_reward > 0) - (player_reward < 0)  # np.sign
                score_rewards_list.append(reward_sign * math.sqrt(abs(player_reward)) / 2)
            elif self.reward_type == 'sqrt_team':
                team_reward = team_score - self.last_leaderboard[game_team_id]
                reward_sign = (team_reward > 0) - (team_reward < 0)  # np.sign
                score_rewards_list.append(reward_sign * math.sqrt(abs(team_reward)) / 2)
            else:
                raise NotImplementedError
            self.last_player_scores[game_player_id] = player_score
        self.last_leaderboard = next_obs[0]['leaderboard']
        return score_rewards_list
