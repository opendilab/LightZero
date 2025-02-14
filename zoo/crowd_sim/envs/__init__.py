from gym.envs.registration import register

register(
    id='CrowdSim-v0',
    entry_point='zoo.crowd_sim.envs.CrowdSim.crowd_sim:CrowdSim',
)
