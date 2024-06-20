import logging
from gym.envs.registration import register
logger = logging.getLogger(__name__)
register(
    id='CrowdSim-v0',
    entry_point='zoo.CrowdSim.envs.Crowdsim.env.crowd_sim:CrowdSim',
)
