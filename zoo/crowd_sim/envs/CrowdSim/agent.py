import abc
import random
import logging
from zoo.crowd_sim.envs.CrowdSim.mdp import *


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
    """
    Overview:
        Human class. Have the physical attributes of a human agent. The human agent has a data queue to store the \
        information blocks. The data queue is updated when the human agent moves and transmits data to the robot. \
        The age of information (aoi) is calculated based on the data queue.
    Interface:
        `__init__`, `set`, `update`, `get_obs`.
    """

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
        """
        Overview:
            Set the physical attributes of the human agent.
        Arguments:
            - px (:obj:`float`): The x-coordinate of the human agent.
            - py (:obj:`float`): The y-coordinate of the human agent.
            - theta (:obj:`float`): The orientation of the human agent.
            - aoi (:obj:`float`): The age of information (aoi) of the human agent.
            - data_amount (:obj:`int`): The amount of data blocks in the data queue of the human agent.
        """
        self.px = px
        self.py = py
        self.theta = theta
        self.aoi = aoi
        self.data_amount = data_amount

    def update(self, px, py, theta, transmitted_data):
        """
        Overview:
            Update the physical attributes of the human agent and the data queue. The age of information (aoi) is \
            calculated based on the data queue.
        Arguments:
            - px (:obj:`float`): The x-coordinate of the human agent.
            - py (:obj:`float`): The y-coordinate of the human agent.
            - theta (:obj:`float`): The orientation of the human agent.
            - transmitted_data (:obj:`int`): The number of data blocks transmitted to the robot.
        """
        self.px = px  # position
        self.py = py
        self.theta = theta
        self.data_queue.update(self.collect_v, transmitted_data)
        self.aoi = self.data_queue.total_aoi()
        self.data_amount = self.data_queue.total_blocks()

    def get_obs(self):
        """
        Overview:
            Get the observation of the human agent. The observation includes the position, age of information (aoi), \
            and the amount of data blocks in the data queue.
        Returns:
            - obs (:obj:`HumanState`): The observation of the human agent.
        """
        # obs: (px, py, remaining_data, aoi)
        return HumanState(
            self.px / self.config.nlon, self.py / self.config.nlat, self.data_amount / self.config.num_timestep,
            self.aoi / self.config.num_timestep
        )


class Robot():
    """
    Overview:
        Robot class. Have the physical attributes of a robot agent.
    Interface:
        `__init__`, `set`, `get_obs`.
    """

    def __init__(self, id, config):
        self.id = id
        self.config = config
        self.px = None  # position
        self.py = None
        self.theta = None
        self.energy = None

    def set(self, px, py, theta, energy):
        """
        Overview:
            Set the physical attributes of the robot agent.
        Arguments:
            - px (:obj:`float`): The x-coordinate of the robot agent.
            - py (:obj:`float`): The y-coordinate of the robot agent.
            - theta (:obj:`float`): The orientation of the robot agent.
            - energy (:obj:`float`): The remaining energy of the robot agent.
        """
        self.px = px  # position
        self.py = py
        self.theta = theta
        self.energy = energy

    def get_obs(self):
        """
        Overview:
            Get the observation of the robot agent. The observation includes the position, orientation, and the remaining \
            energy of the robot agent.
        Returns:
            - obs (:obj:`RobotState`): The observation of the robot agent.
        """
        return RobotState(
            self.px / self.config.nlon, self.py / self.config.nlat, self.theta / self.config.rotation_limit,
            self.energy / self.config.max_uav_energy
        )


class InformationQueue:
    """
    Overview:
        Information queue class. The data queue is updated when the human agent moves and transmits data to the robot. \
        The age of information (aoi) is calculated based on the data queue.

    Interface:
        `__init__`, `update`, `total_aoi`, `total_blocks`.
    """

    def __init__(self):
        # Initialize the queue to hold the age of each information block
        self.queue = []

    def update(self, arrivals, departures):
        """
        Overview:
            Update the data queue. Increase the age of information (aoi) for each block in the queue. Add new information \
            blocks with aoi of 0. Remove the specified number of oldest information blocks.
        Arguments:
            - arrivals (:obj:`int`): The number of new information blocks entering the queue.
            - departures (:obj:`int`): The number of oldest information blocks leaving the queue.
        """
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
# print(total_age)
