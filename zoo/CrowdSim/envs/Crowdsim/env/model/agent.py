import abc
import random
import logging
from zoo.CrowdSim.envs.Crowdsim.env.model.mdp import *


class Agent():
    def __init__(self):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.policy = None

    def print_info(self):
        logging.info('Agent is visible and has "holonomic" kinematic constraint')

    def set_policy(self, policy):
        self.policy = policy

    def act(self, state, current_timestep):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        action = self.policy.predict(state, current_timestep)
        return action


class Human():
    # collect_v_prob = {1: 1, 2: 0}
    def __init__(self, id, config):
        self.id = id
        self.config = config
        self.px = None
        self.py = None
        self.theta = None
        self.aoi = 0
        self.data_queue = InformationQueue()
        self.data_amount = 0
        self.collect_v_prob = getattr(self.config, 'collect_v_prob', {1: 1, 2: 0})
        self.collect_v = random.choices(list(map(int, self.collect_v_prob.keys())), list(self.collect_v_prob.values()))[0]

    def set(self, px, py, theta, aoi, data_amount):
        self.px = px
        self.py = py
        self.theta = theta
        self.aoi = aoi
        self.data_amount = data_amount

    def update(self, px, py, theta, transmitted_data):
        self.px = px  # position
        self.py = py
        self.theta = theta
        self.data_queue.update(self.collect_v, transmitted_data)
        self.aoi = self.data_queue.total_aoi()
        self.data_amount = self.data_queue.total_blocks()

    # TODO: change state,可能需要归一化
    def get_obs(self):
        # obs: (px, py, remaining_data, aoi)
        return HumanState(self.px / self.config.nlon,
                          self.py / self.config.nlat,
                          self.data_amount / self.config.num_timestep,
                          self.aoi / self.config.num_timestep)


class Robot():
    def __init__(self, id, config):
        self.id = id
        self.config = config
        self.px = None  # position
        self.py = None
        self.theta = None
        self.energy = None

    def set(self, px, py, theta, energy):
        self.px = px  # position
        self.py = py
        self.theta = theta
        self.energy = energy

    # TODO: change state,可能需要归一化
    def get_obs(self):
        return RobotState(self.px / self.config.nlon,
                          self.py / self.config.nlat,
                          self.theta / self.config.rotation_limit,
                          self.energy / self.config.max_uav_energy)


class InformationQueue:
    def __init__(self):
        # Initialize the queue to hold the age of each information block
        self.queue = []

    def update(self, arrivals, departures):
        # Increase the age of information (aoi) for each block in the queue
        self.queue = [age + 1 for age in self.queue]
        
        # Add new information blocks with aoi of 0
        self.queue.extend([0] * arrivals)
        
        # Remove the specified number of oldest information blocks
        self.queue = self.queue[departures:] if departures <= len(self.queue) else []

    def total_aoi(self):
        # Return the total age of information in the queue
        return sum(self.queue)
    
    def total_blocks(self):
        # Return the total number of information blocks in the queue
        return len(self.queue)

# # Example of using the InformationQueue class
# info_queue = InformationQueue()
# info_queue.update(arrivals=5, departures=0)  # 5 blocks enter the queue, all with aoi of 0
# info_queue.update(arrivals=3, departures=2)  # 3 new blocks enter, 2 blocks leave
# total_age = info_queue.total_aoi()  # Calculate the total age of information in the queue

# total_age
