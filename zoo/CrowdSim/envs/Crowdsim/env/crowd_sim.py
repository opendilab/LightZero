import pandas as pd

import logging
import random
import gym
# from shapely.geometry import Point
import numpy as np
from scipy.stats import entropy
# import folium
# from folium.plugins import TimestampedGeoJson, AntPath

from zoo.CrowdSim.envs.Crowdsim.env.model.utils import *
from zoo.CrowdSim.envs.Crowdsim.env.model.mdp import HumanState, RobotState, JointState
from LightZero.zoo.CrowdSim.envs.Crowdsim.env.crowd_sim_base_config import get_selected_config


class CrowdSim(gym.Env):
    """
    Overview:
        LightZero version of the CrowdSim environment. This class includes methods for resetting, closing, and \
        stepping through the environment, as well as seeding for reproducibility, saving replay videos, and generating \
        random actions. It also includes properties for accessing the observation space, action space, and reward space of the \
        environment. The environment is a grid world with humans and robots moving around. The robots are tasked with \
        minimizing the average age of information (AoI) of the humans by moving to their locations and collecting data from them. \
        The humans generate data at a constant rate, and the robots have a limited energy supply that is consumed by moving. \
        The environment is divided into two modes: 'easy' and 'hard'. In the 'easy' mode, the robots can only collect data from \
        humans when they are within a certain range, and the AoI of a human is reset to 0 when a robot collects data from them. \
        In the 'hard' mode, the robots can collect data from humans even when they are not within range, and the AoI of a human \
        is not reset when a robot collects data from them. The environment is initialized with a dataset of human locations and \
        timestamps, and the robots are tasked with collecting data from the humans to minimize the average AoI. The environment \
        is considered solved when the average AoI is minimized to a certain threshold or the time limit is reached.
    Interface:
        `__init__`, `reset`, `step`, `render`, `sync_human_df`, `generate_human`, `generate_robot`.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, dataset, custom_config=None):
        """
        Overview:
            Initialize the environment with a dataset and a custom configuration. The dataset contains the locations and \
            timestamps of the humans, and the custom configuration contains the environment mode, number of humans, number \
            of robots, maximum timestep, step time, start timestamp, and maximum UAV energy. The environment is divided into \
            two modes: 'easy' and 'hard'. In the 'easy' mode, the robots can only collect data from humans when they are within \
            a certain range, and the AoI of a human is reset to 0 when a robot collects data from them. In the 'hard' mode, the \
            robots can collect data from humans even when they are not within range, and the AoI of a human is not reset when a \
            robot collects data from them. The environment is initialized with a dataset of human locations and timestamps, and \
            the robots are tasked with collecting data from the humans to minimize the average AoI. The environment is considered \
            solved when the average AoI is minimized to a certain threshold or the time limit is reached.
        Args:
            - dataset (:obj:`str`): The path to the dataset file.
            - custom_config (:obj:`dict`): A dictionary containing the custom configuration for the environment. \
                The custom configuration should include the following keys:
                - env_mode (:obj:`str`): The environment mode ('easy' or 'hard').
                - human_num (:obj:`int`): The number of humans in the environment.
                - robot_num (:obj:`int`): The number of robots in the environment.
                - num_timestep (:obj:`int`): The maximum timestep for the environment.
                - step_time (:obj:`float`): The time per step in seconds.
                - start_timestamp (:obj:`int`): The start timestamp for the environment.
                - max_uav_energy (:obj:`float`): The maximum energy for the UAVs.
        """
        # mcfg should include:
        self.time_limit = None
        self.robots = None
        self.humans = None
        self.agent = None
        self.current_timestep = None
        self.phase = None

        self.config = get_selected_config(dataset)
        self.config.update(custom_config)

        self.env_mode = self.config.env_mode  # 'easy' or 'hard'
        self.human_num = self.config.human_num
        self.robot_num = self.config.robot_num
        self.num_timestep = self.config.num_timestep  # max timestep
        self.step_time = self.config.step_time  # second per step
        self.start_timestamp = self.config.start_timestamp  # fit timpestamp to datetime
        self.max_uav_energy = self.config.max_uav_energy
        # self.action_space = gym.spaces.Discrete(4**self.robot_num) # for each robot have 4 actions(up, down, left, right), then product
        self.action_space = gym.spaces.Discrete(len(self.config.one_uav_action_space))
        # human obs: [px, py, remaining_data_amount, aoi]
        # robot obs: [px, py, theta, energy]
        self.observation_space = gym.spaces.Box(
            low=float("-inf"), high=float("inf"), shape=(self.robot_num + self.human_num, 4), dtype=np.float32
        )

        # load_dataset
        self.transmit_v = self.config.transmit_v  # 5*0.3Mb/s
        self.nlon = self.config.nlon
        self.nlat = self.config.nlat
        self.lower_left = self.config.lower_left
        self.upper_right = self.config.upper_right
        self.human_df = pd.read_csv(self.config.dataset_dir)
        logging.info("Finished reading {} rows".format(len(self.human_df)))

        self.human_df['t'] = pd.to_datetime(self.human_df['timestamp'], unit='s')  # 's' stands for second
        self.human_df['aoi'] = -1  # 加入aoi记录aoi
        self.human_df['data_amount'] = -1  # record the remaining data amount of each human
        self.human_df['energy'] = -1  # 加入energy记录energy
        logging.info('Env mode: {}'.format(self.env_mode))
        logging.info('human number: {}'.format(self.human_num))
        logging.info('Robot number: {}'.format(self.robot_num))

        # for debug
        self.current_human_aoi_list = np.zeros([
            self.human_num,
        ])
        self.mean_aoi_timelist = np.zeros([
            self.config.num_timestep + 1,
        ])
        self.cur_data_amount_timelist = np.zeros([
            self.human_num,
        ])
        self.robot_energy_timelist = np.zeros([self.config.num_timestep + 1, self.robot_num])
        self.robot_x_timelist = np.zeros([self.config.num_timestep + 1, self.robot_num])
        self.robot_y_timelist = np.zeros([self.config.num_timestep + 1, self.robot_num])
        self.update_human_timelist = np.zeros([
            self.config.num_timestep,
        ])
        self.data_transmission = 0
        self.data_collection_distribution = np.zeros(self.human_num)
        self.data_transmission_distribution = np.zeros(self.human_num)

    def generate_human(self, human_id, selected_data, selected_next_data):
        """
        Overview:
            Generate a human with the given id, selected data, and selected next data. The human is initialized with \
            the given data and next data, and the remaining data amount is set to 0. The human is also initialized with \
            an AoI of 0.
        Argments:
            - human_id (:obj:`int`): The id of the human.
            - selected_data (:obj:`pd.DataFrame`): The selected data for the current timestep.
            - selected_next_data (:obj:`pd.DataFrame`): The selected data for the next timestep.
        Returns:
            - human (:obj:`Human`): The generated human.
        """
        human = Human(human_id, self.config)
        px, py, theta = get_human_position_from_list(
            self.current_timestep, human_id, selected_data, selected_next_data, self.config
        )
        # human obs: [px, py, data_amount, aoi]
        human.set(px, py, theta, 0, 0)  # initial aoi of human is 0
        return human

    def generate_robot(self, robot_id):
        """
        Overview:
            Generate a robot with the given id. The robot is initialized with the given id and the maximum UAV energy.
        Argments:
            - robot_id (:obj:`int`): The id of the robot.
        Returns:
            - robot (:obj:`Robot`): The generated robot.
        """
        robot = Robot(robot_id, self.config)
        # robot obs: [px, py, theta, energy]
        robot.set(self.nlon / 2, self.nlat / 2, 0, self.max_uav_energy)  # robot有energy
        return robot

    def sync_human_df(self, human_id, current_timestep, aoi, data_amount):
        """
        Overview:
            Sync the human_df with the current timestep and aoi.
        Args:
            - human_id (:obj:`int`): The id of the human.
            - current_timestep (:obj:`int`): The current timestep.
            - aoi (:obj:`int`): The aoi of the human.
        """
        current_timestamp = self.start_timestamp + current_timestep * self.step_time
        current_index = self.human_df[(self.human_df.id == human_id)
                                      & (self.human_df.timestamp == current_timestamp)].index
        # self.human_df.loc[current_index, "aoi"] = aoi   # slower
        self.human_df.iat[current_index.values[0], 9] = aoi  # faster
        self.human_df.iat[current_index.values[0], 10] = data_amount

    def reset(self, phase='test', test_case=None):
        """
        Overview:
            Reset the environment to the initial state. The environment is reset to the start timestamp, and the humans \
            and robots are generated with the given data. The humans are initialized with the selected data and next data, \
            and the robots are initialized with the given id. The environment is also initialized with the current timestep, \
            mean AoI, robot energy, robot x, robot y, and update human timelist. The environment is considered solved when \
            the average AoI is minimized to a certain threshold or the time limit is reached.
        Argments:
            - phase (:obj:`str`): The phase of the environment ('train' or 'test').
            - test_case (:obj:`int`): The test case for the environment.
        Returns:
            - state (:obj:`JointState`): The initial state of the environment.
        """
        self.current_timestep = 0

        # generate human
        self.humans = []
        selected_data, selected_next_data = get_human_position_list(self.current_timestep, self.human_df, self.config)
        self.generate_data_amount_per_step = 0
        self.total_generated_data_amount = 0
        for human_id in range(self.human_num):
            self.humans.append(self.generate_human(human_id, selected_data, selected_next_data))
            self.generate_data_amount_per_step += self.humans[human_id].collect_v
            self.sync_human_df(human_id, self.current_timestep, aoi=0, data_amount=0)

        # generate robot
        self.robots = []
        for robot_id in range(self.robot_num):
            self.robots.append(self.generate_robot(robot_id))

        self.cur_data_amount_timelist = np.zeros([
            self.human_num,
        ])
        self.current_human_aoi_list = np.zeros([
            self.human_num,
        ])
        self.mean_aoi_timelist = np.zeros([
            self.config.num_timestep + 1,
        ])
        self.mean_aoi_timelist[self.current_timestep] = np.mean(self.current_human_aoi_list)
        self.robot_energy_timelist = np.zeros([self.config.num_timestep + 1, self.robot_num])
        self.robot_energy_timelist[self.current_timestep, :] = self.max_uav_energy
        self.robot_x_timelist = np.zeros([self.config.num_timestep + 1, self.robot_num])
        self.robot_x_timelist[self.current_timestep, :] = self.nlon / 2
        self.robot_y_timelist = np.zeros([self.config.num_timestep + 1, self.robot_num])
        self.robot_y_timelist[self.current_timestep, :] = self.nlat / 2
        self.update_human_timelist = np.zeros([
            self.config.num_timestep,
        ])
        self.data_transmission = 0
        self.data_collection_distribution = np.zeros(self.human_num)
        self.data_transmission_distribution = np.zeros(self.human_num)

        # for visualization
        self.plot_states = []
        self.robot_actions = []
        self.rewards = []
        self.aoi_rewards = []
        self.energy_rewards = []
        self.action_values = []
        self.plot_states.append(
            [[robot.get_obs() for robot in self.robots], [human.get_obs() for human in self.humans]]
        )

        state = JointState([robot.get_obs() for robot in self.robots], [human.get_obs() for human in self.humans])
        return state

    def step(self, action):
        """
        Overview:
            Perform a step in the environment using the provided action, and return the next state of the environment. \
            The next state is encapsulated in a BaseEnvTimestep object, which includes the new observation, reward, done flag, \
            and info dictionary. The cumulative reward (`_eval_episode_return`) is updated with the reward obtained in this step. \
            If the episode ends (done is True), the total reward for the episode is stored in the info dictionary.
        Argments:
            - action (:obj:`Union[int, np.ndarray]`): The action to be performed in the environment. If the action is a 1-dimensional \
                numpy array, it is squeezed to a 0-dimension array.
        Returns:
            - next_state (:obj:`JointState`): The next state of the environment.
            - reward (:obj:`float`): The reward obtained in this step.
            - done (:obj:`bool`): A flag indicating whether the episode has ended.
            - info (:obj:`dict`): A dictionary containing additional information about the environment.
        """
        new_robot_position = np.zeros([self.robot_num, 2])
        current_enenrgy_consume = np.zeros([
            self.robot_num,
        ])

        num_updated_human = 0  # number of humans whose AoI is updated

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

            if is_collide or (new_robot_px < 0 or new_robot_px > self.nlon or new_robot_py < 0 or new_robot_py > self.nlat):
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

        selected_data, selected_next_data = get_human_position_list(
            self.current_timestep + 1, self.human_df, self.config
        )
        human_transmit_data_list = np.zeros_like(self.cur_data_amount_timelist)  # 0 means no update
        for human_id, human in enumerate(self.humans):
            next_px, next_py, next_theta = get_human_position_from_list(
                self.current_timestep + 1, human_id, selected_data, selected_next_data, self.config
            )
            should_reset = judge_aoi_update([next_px, next_py], new_robot_position, self.config)
            if self.env_mode == 'easy':
                if should_reset:
                    # if the human is in the range of the robot, then part of human's data will be transmitted
                    if human.aoi > 1:
                        human_transmit_data_list[human_id] = human.aoi
                    else:
                        human_transmit_data_list[human_id] = 1

                    human.set(next_px, next_py, next_theta, aoi=0, data_amount=0)
                    num_updated_human += 1
                else:
                    # if the human is not in the range of the robot, then update the aoi of the human
                    human_transmit_data_list[human_id] = 0
                    new_aoi = human.aoi + 1
                    human.set(next_px, next_py, next_theta, aoi=new_aoi, data_amount=human.aoi)

            elif self.env_mode == 'hard':
                if should_reset:
                    # if the human is in the range of the robot, then part of human's data will be transmitted
                    last_data_amount = human.data_amount
                    human.update(next_px, next_py, next_theta, transmitted_data=self.transmit_v)
                    human_transmit_data_list[human_id] = min(last_data_amount + human.collect_v, self.transmit_v)
                    num_updated_human += 1
                else:
                    # if the human is not in the range of the robot, then no data will be transmitted, \
                    # and update aoi and caculate new collected data amount
                    human_transmit_data_list[human_id] = 0
                    human.update(next_px, next_py, next_theta, transmitted_data=0)
            else:
                raise ValueError("env_mode should be 'easy' or 'hard'")

            self.cur_data_amount_timelist[human_id] = human.data_amount
            self.current_human_aoi_list[human_id] = human.aoi
            self.sync_human_df(human_id, self.current_timestep + 1, human.aoi, human.data_amount)
            self.data_collection_distribution[human_id] += human.collect_v
            self.data_transmission_distribution[human_id] += human_transmit_data_list[human_id]

        self.mean_aoi_timelist[self.current_timestep + 1] = np.mean(self.current_human_aoi_list)
        self.update_human_timelist[self.current_timestep] = num_updated_human
        delta_sum_transmit_data = np.sum(human_transmit_data_list)
        self.data_transmission += (delta_sum_transmit_data * 0.3)  # Mb, 0.02M/s per person
        if self.env_mode == 'easy':
            # in easy mode, the data amount generated per step is equal to the number of humans
            self.total_generated_data_amount = self.num_timestep * self.human_num
        elif self.env_mode == 'hard':
            # in hard mode, the data amount generated per step is equal to the sum of the data amount of all humans
            self.total_generated_data_amount += self.generate_data_amount_per_step

        # TODO: need to be well-defined
        aoi_reward = self.mean_aoi_timelist[self.current_timestep] - self.mean_aoi_timelist[self.current_timestep + 1]
        energy_reward = np.sum(current_enenrgy_consume)
        reward = aoi_reward \
                 - self.config.energy_factor * energy_reward

        # if hasattr(self.agent.policy, 'action_values'):
        #     self.action_values.append(self.agent.policy.action_values)
        self.robot_actions.append(action)
        self.rewards.append(reward)
        self.aoi_rewards.append(aoi_reward)
        self.energy_rewards.append(energy_reward)
        distribution_entropy = entropy(
            self.data_collection_distribution/ np.sum(self.data_collection_distribution),
            self.data_transmission_distribution/np.sum(self.data_transmission_distribution) + 1e-10)
        self.plot_states.append([[robot.get_obs() for robot in self.robots],
                                 [human.get_obs() for human in self.humans]])

        next_state = JointState([robot.get_obs() for robot in self.robots], [human.get_obs() for human in self.humans])

        self.current_timestep += 1
        # print('This game is on',self.current_timestep,' step\n')
        if self.current_timestep >= self.num_timestep:
            done = True
        else:
            done = False
        info = {
            "performance_info": {
            "mean_aoi": self.mean_aoi_timelist[self.current_timestep],
            "mean_transmit_data": self.data_transmission / self.human_num,
            "mean_energy_consumption": 1.0 - (
                        np.mean(self.robot_energy_timelist[self.current_timestep]) / self.max_uav_energy),
            "transmitted_data_ratio": self.data_transmission/(self.total_generated_data_amount*0.3),
            "human_coverage": np.mean(self.update_human_timelist) / self.human_num,
            "distribution_entropy": distribution_entropy  # 增加交叉熵信息
            },
        }

        return next_state, reward, done, info

    def render(self):
        """
        Overview:
            Render the environment to an image. The image is generated using the matplotlib library, and it includes the \
            historical trajectories of the robots, the current positions of the robots, the sensing range of the robots, the \
            positions of the humans, and their AoI values. The image is returned as a numpy array.
        Returns:
            - image (:obj:`np.ndarray`): The rendered image of the environment.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import io
        import imageio

        map_max_x = self.config.nlon
        map_max_y = self.config.nlat
        # Create a new figure
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(right=0.75) # 给数据留白

        # Plot the historical trajectories of the robots
        for timestep in range(len(self.robot_x_timelist)):
            for robot_id in range(len(self.robot_x_timelist[timestep])):
                ax.plot(
                    self.robot_x_timelist[timestep][robot_id],
                    self.robot_y_timelist[timestep][robot_id],
                    color='gray',
                    alpha=0.5
                )

        # Plot the current positions of the robots
        for robot in self.robots:
            ax.plot(robot.px, robot.py, marker='o', markersize=5, color='blue')

        # Plot the sensing range of the robots
        for robot in self.robots:
            robot_x, robot_y = robot.px, robot.py
            circle = patches.Circle((robot_x, robot_y), self.config.sensing_range, edgecolor='blue', facecolor='none')
            ax.add_patch(circle)

        # Plot the positions of the humans and their AOI values
        for human in self.humans:
            human_x, human_y, aoi = human.px, human.py, human.aoi
            ax.plot(human_x, human_y, marker='x', markersize=5, color='red')
            ax.text(human_x, human_y, str(aoi), fontsize=8, color='black')

        # Set the title and axis labels
        # ax.set_xlim(0, map_max_x)
        # ax.set_ylim(0, map_max_y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # show reward/aoi_reward/energy_reward/mean_aoi/energy in the upper right corner
        reward_text = f"Reward: {self.rewards[-1] if self.rewards else 0:.2f}\n" \
                      f"AOI Reward: {self.aoi_rewards[-1] if self.aoi_rewards else 0:.2f}\n" \
                      f"Energy Reward: {self.energy_rewards[-1] if self.energy_rewards else 0:.2f}\n" \
                      f"Mean AOI: {self.mean_aoi_timelist[self.current_timestep] if self.current_timestep < len(self.mean_aoi_timelist) else 0:.2f}\n" \
                      f"Energy: {np.mean(self.robot_energy_timelist[self.current_timestep]) if self.current_timestep < len(self.robot_energy_timelist) else 0:.2f}"
        plt.text(1.05, 0.95, reward_text, horizontalalignment='left', verticalalignment='top', 
             transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.6), 
             clip_on=False)  # Ensure text is not clipped
        # Leave some blank space outside of the map
        ax.margins(x=0.1, y=0.1)
        ax.set_title('Crowd Simulation Visualization')

        # Render the figure to an image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        plt.close()

        return image
