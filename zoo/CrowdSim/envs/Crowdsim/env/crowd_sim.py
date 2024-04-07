import pandas as pd

import logging
import random
import gym
from shapely.geometry import Point
import numpy as np
import folium
from folium.plugins import TimestampedGeoJson, AntPath

from zoo.CrowdSim.envs.Crowdsim.env.model.utils import *
from zoo.CrowdSim.envs.Crowdsim.env.model.mdp import HumanState, RobotState, JointState
from zoo.CrowdSim.envs.Crowdsim.env.base_env_config import get_selected_config



class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dataset, custom_config=None):
        # mcfg should include: 
        self.time_limit = None
        self.robots = None
        self.humans = None
        self.agent = None
        self.current_timestep = None
        self.phase = None

        self.config = get_selected_config(dataset)
        self.config.update(custom_config)

        self.human_num = self.config.human_num
        self.robot_num = self.config.robot_num
        self.num_timestep = self.config.num_timestep    # max timestep
        self.step_time = self.config.step_time  # second per step
        self.start_timestamp = self.config.start_timestamp  # fit timpestamp to datetime
        self.max_uav_energy = self.config.max_uav_energy
        # self.action_space = gym.spaces.Discrete(4**self.robot_num) # for each robot have 4 actions(up, down, left, right), then product
        self.action_space = gym.spaces.Discrete(len(self.config.one_uav_action_space))
        # human obs: [px, py, theta, aoi]
        # robot obs: [px, py, theta, energy]
        # self.observation_space = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(4), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(self.robot_num+self.human_num, 4), dtype=np.float32)

        # load_dataset
        self.nlon = self.config.nlon
        self.nlat = self.config.nlat
        self.lower_left = self.config.lower_left
        self.upper_right = self.config.upper_right
        self.human_df = pd.read_csv(self.config.dataset_dir)
        logging.info("Finished reading {} rows".format(len(self.human_df)))
        # # for temporarily processing data
        # sample_list=np.random.choice(self.human_num, size=[50,], replace=False)
        # sample_list=sample_list[np.argsort(sample_list)]
        # print(sample_list)
        # self.human_df= self.human_df[self.human_df["id"].isin(sample_list)]
        # for i,human_id in enumerate(sample_list):
        #     mask=(self.human_df["id"]==human_id)
        #     self.human_df.loc[mask,"id"]=i
        # self.human_df=self.human_df.sort_values(by=["id","timestamp"],ascending=[True,True])
        # print(self.human_df.head())
        # self.human_df.to_csv("50 users-5.csv",index=False)
        # exit(0)

        self.human_df['t'] = pd.to_datetime(self.human_df['timestamp'], unit='s')  # 's' stands for second
        self.human_df['aoi'] = -1  # 加入aoi记录aoi
        self.human_df['energy'] = -1  # 加入energy记录energy
        logging.info('human number: {}'.format(self.human_num))
        logging.info('Robot number: {}'.format(self.robot_num))

        # for debug
        self.current_human_aoi_list = np.ones([self.human_num, ])
        self.mean_aoi_timelist = np.ones([self.config.num_timestep + 1, ])
        self.robot_energy_timelist = np.zeros([self.config.num_timestep + 1, self.robot_num])
        self.robot_x_timelist = np.zeros([self.config.num_timestep + 1, self.robot_num])
        self.robot_y_timelist = np.zeros([self.config.num_timestep + 1, self.robot_num])
        self.update_human_timelist = np.zeros([self.config.num_timestep, ])
        self.data_collection = 0

    def set_agent(self, agent):
        self.agent = agent

    def generate_human(self, human_id, selected_data, selected_next_data):
        human = Human(human_id, self.config)
        px, py, theta = get_human_position_from_list(self.current_timestep, human_id, selected_data, selected_next_data, self.config)
        # human obs: [px, py, theta, aoi]
        human.set(px, py, theta, 1)  # initial aoi of human is 1
        return human

    def generate_robot(self, robot_id):
        robot = Robot(robot_id, self.config)
        # robot obs: [px, py, theta, energy]
        robot.set(self.nlon / 2, self.nlat / 2, 0, self.max_uav_energy)  # robot有energy
        return robot

    def sync_human_df(self, human_id, current_timestep, aoi):
        """
        Overview:
            Sync the human_df with the current timestep and aoi.
        Args:
            - human_id (:obj:`int`): The id of the human.
            - current_timestep (:obj:`int`): The current timestep.
            - aoi (:obj:`int`): The aoi of the human.
        """
        current_timestamp = self.start_timestamp + current_timestep * self.step_time
        current_index = self.human_df[
            (self.human_df.id == human_id) & (self.human_df.timestamp == current_timestamp)].index
        # self.human_df.loc[current_index, "aoi"] = aoi   # slower
        self.human_df.iat[current_index.values[0], 9] = aoi # faster

    def reset(self, phase='test', test_case=None):
        self.current_timestep = 0

        # generate human
        self.humans = []
        selected_data, selected_next_data = get_human_position_list(self.current_timestep, self.human_df, self.config)
        for human_id in range(self.human_num):
            self.humans.append(self.generate_human(human_id, selected_data, selected_next_data))
            self.sync_human_df(human_id, self.current_timestep, 1)

        # generate robot
        self.robots = []
        for robot_id in range(self.robot_num):
            self.robots.append(self.generate_robot(robot_id))

        self.current_human_aoi_list = np.ones([self.human_num, ])
        self.mean_aoi_timelist = np.ones([self.config.num_timestep + 1, ])
        self.mean_aoi_timelist[self.current_timestep] = np.mean(self.current_human_aoi_list)
        self.robot_energy_timelist = np.zeros([self.config.num_timestep + 1, self.robot_num])
        self.robot_energy_timelist[self.current_timestep, :] = self.max_uav_energy
        self.robot_x_timelist = np.zeros([self.config.num_timestep + 1, self.robot_num])
        self.robot_x_timelist[self.current_timestep, :] = self.nlon / 2
        self.robot_y_timelist = np.zeros([self.config.num_timestep + 1, self.robot_num])
        self.robot_y_timelist[self.current_timestep, :] = self.nlat / 2
        self.update_human_timelist = np.zeros([self.config.num_timestep, ])
        self.data_collection = 0

        # for visualization
        self.plot_states = []
        self.robot_actions = []
        self.rewards = []
        self.action_values = []
        self.plot_states.append([[robot.get_obs() for robot in self.robots],
                                 [human.get_obs() for human in self.humans]])
        
        state = JointState([robot.get_obs() for robot in self.robots], [human.get_obs() for human in self.humans])
        return state

    def step(self, action):
        new_robot_position = np.zeros([self.robot_num, 2])
        current_enenrgy_consume = np.zeros([self.robot_num, ])

        num_updated_human = 0   # number of humans whose AoI is updated

        for robot_id, robot in enumerate(self.robots):
            new_robot_px = robot.px + action[robot_id][0]
            new_robot_py = robot.py + action[robot_id][1]
            robot_theta = get_theta(0, 0, action[robot_id][0], action[robot_id][1])
            # print(action[robot_id], robot_theta)
            is_stopping = True if (action[robot_id][0] == 0 and action[robot_id][1] == 0) else False
            is_collide = True if judge_collision(new_robot_px, new_robot_py, robot.px, robot.py, self.config) else False

            if is_stopping is True:
                consume_energy = consume_uav_energy(0, self.step_time, self.config)
            else:
                consume_energy = consume_uav_energy(self.step_time, 0, self.config)
            current_enenrgy_consume[robot_id] = consume_energy / self.config.max_uav_energy
            new_energy = robot.energy - consume_energy
            self.robot_energy_timelist[self.current_timestep + 1][robot_id] = new_energy

            if is_collide is True:
                new_robot_position[robot_id][0] = robot.px
                new_robot_position[robot_id][1] = robot.py
                self.robot_x_timelist[self.current_timestep + 1][robot_id] = robot.px
                self.robot_y_timelist[self.current_timestep + 1][robot_id] = robot.py
                robot.set(robot.px, robot.py, robot_theta, energy=new_energy)
            else:
                new_robot_position[robot_id][0] = new_robot_px
                new_robot_position[robot_id][1] = new_robot_py
                self.robot_x_timelist[self.current_timestep + 1][robot_id] = new_robot_px
                self.robot_y_timelist[self.current_timestep + 1][robot_id] = new_robot_py
                robot.set(new_robot_px, new_robot_py, robot_theta, energy=new_energy)

        selected_data, selected_next_data = get_human_position_list(self.current_timestep + 1, self.human_df, self.config)
        delta_human_aoi_list = np.zeros_like(self.current_human_aoi_list)   # 0 means no update
        for human_id, human in enumerate(self.humans):
            next_px, next_py, next_theta = get_human_position_from_list(self.current_timestep + 1, human_id,
                                                                        selected_data, selected_next_data, self.config)
            should_reset = judge_aoi_update([next_px, next_py], new_robot_position, self.config)
            if should_reset:
                # if the human is in the range of the robot, then reset the aoi of the human
                if human.aoi > 1:
                    delta_human_aoi_list[human_id] = human.aoi
                else:
                    delta_human_aoi_list[human_id] = 1

                human.set(next_px, next_py, next_theta, aoi=1)
                num_updated_human += 1
            else:
                # if the human is not in the range of the robot, then update the aoi of the human
                delta_human_aoi_list[human_id] = 0
                new_aoi = human.aoi + 1
                human.set(next_px, next_py, next_theta, aoi=new_aoi)

            self.current_human_aoi_list[human_id] = human.aoi
            self.sync_human_df(human_id, self.current_timestep + 1, human.aoi)

        self.mean_aoi_timelist[self.current_timestep + 1] = np.mean(self.current_human_aoi_list)
        self.update_human_timelist[self.current_timestep] = num_updated_human
        delta_sum_aoi = np.sum(delta_human_aoi_list)
        self.data_collection += (delta_sum_aoi * 0.3)  # Mb, 0.02M/s per person

        # TODO: need to be well-defined
        reward = self.mean_aoi_timelist[self.current_timestep] - self.mean_aoi_timelist[self.current_timestep + 1] \
                 - self.config.energy_factor * np.sum(current_enenrgy_consume)

        # if hasattr(self.agent.policy, 'action_values'):
        #     self.action_values.append(self.agent.policy.action_values)
        self.robot_actions.append(action)
        self.rewards.append(reward)
        self.plot_states.append([[robot.get_obs() for robot in self.robots],
                                 [human.get_obs() for human in self.humans]])

        next_state = JointState([robot.get_obs() for robot in self.robots],
                                [human.get_obs() for human in self.humans])

        self.current_timestep += 1
        # print('This game is on',self.current_timestep,' step\n')
        if self.current_timestep >= self.num_timestep:
            done = True
        else:
            done = False
        info = {
            "performance_info": {
            "mean_aoi": self.mean_aoi_timelist[self.current_timestep],
            "mean_energy_consumption": 1.0 - (
                        np.mean(self.robot_energy_timelist[self.current_timestep]) / self.max_uav_energy),
            "collected_data_amount": self.data_collection/(self.num_timestep*self.human_num*0.3),
            "human_coverage": np.mean(self.update_human_timelist) / self.human_num
            },
        }

        return next_state, reward, done, info

    def render(self, mode='traj', output_file=None, plot_loop=False, moving_line=False):
        # -------------------------------------------------------------------
        if mode == 'html':
            pass
        elif mode == 'traj':
            pass
        else:
            raise NotImplementedError