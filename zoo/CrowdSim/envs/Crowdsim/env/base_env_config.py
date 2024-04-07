from easydict import EasyDict

# define base config
base_config = EasyDict({
    "num_timestep": 120,  # 120x15=1800s=30min
    "step_time": 15,  # second per step
    "max_uav_energy": 359640,  # 359640 J <-- 359.64 kJ (4500mah, 22.2v) 大疆经纬
    "rotation_limit": 360,
    "diameter_of_human_blockers": 0.5,  # m
    "h_rx": 1.3,  # m, height of RX
    "h_b": 1.7,  # m, height of a human blocker
    "velocity": 18,
    "frequence_band": 28,  # GHz
    "h_d": 120,  # m, height of drone-BS
    "alpha_nlos": 113.63,
    "beta_nlos": 1.16,
    "zeta_nlos": 2.58,  # Frequency 28GHz, sub-urban. channel model
    "alpha_los": 84.64,
    "beta_los": 1.55,
    "zeta_los": 0.12,
    "g_tx": 0,  # dB
    "g_rx": 5,  # dB
    "tallest_locs": None,  # obstacle
    "no_fly_zone": None,  # obstacle
    "start_timestamp": 1519894800,
    "end_timestamp": 1519896600,
    "energy_factor": 3,  # TODO: energy factor in reward function
    "robot_num": 2,  # TODO: 多了要用多进程
    "rollout_num": 1,  # 1 2 6 12 15, calculated based on robot_num
})

# define all dataset configs
dataset_configs = {
    'purdue': EasyDict({
        "lower_left": [-86.93, 40.4203],  # 经纬度
        "upper_right": [-86.9103, 40.4313],
        "nlon": 200,
        "nlat": 120,
        "human_num": 59,
        "dataset_dir": '/home/nighoodRen/CrowdSim/CrowdSim/envs/crowd_sim/dataset/purdue/59 users.csv',
        "sensing_range": 23.2,  # unit  23.2
        "one_uav_action_space": [[0, 0], [30, 0], [-30, 0], [0, 30], [0, -30], [21, 21], [21, -21], [-21, 21], [-21, -21]],
        "max_x_distance": 1667,  # m
        "max_y_distance": 1222,  # m
        "density_of_human_blockers": 30000 / 1667 / 1222,  # block/m2
    }),
    'ncsu': EasyDict({
        "lower_left": [-78.6988, 35.7651],  # 经纬度
        "upper_right": [-78.6628, 35.7896],
        "nlon": 3600,
        "nlat": 2450,
        "human_num": 33,
        "dataset_dir": '/home/nighoodRen/CrowdSim/CrowdSim/envs/crowd_sim/dataset/NCSU/33 users.csv',
        "sensing_range": 220,  # unit  220
        "one_uav_action_space": [[0, 0], [300, 0], [-300, 0], [0, 300], [0, -300], [210, 210], [210, -210], [-210, 210], [-210, -210]],
        "max_x_distance": 3255.4913305859623,
        "max_y_distance": 2718.3945272795013,
        "density_of_human_blockers": 30000 / 3255.4913305859623 / 2718.3945272795013,  # block/m2
    }),
    'kaist': EasyDict({
        "lower_left": [127.3475, 36.3597],
        "upper_right": [127.3709, 36.3793],
        "nlon": 2340,
        "nlat": 1960,
        "human_num": 92,
        "dataset_dir": '/home/nighoodRen/CrowdSim/CrowdSim/envs/crowd_sim/dataset/KAIST/92 users.csv',
        "sensing_range": 220,  # unit  220
        "one_uav_action_space": [[0, 0], [300, 0], [-300, 0], [0, 300], [0, -300], [210, 210], [210, -210], [-210, 210], [-210, -210]],
        "max_x_distance": 2100.207579392558,
        "max_y_distance": 2174.930950809533,
        "density_of_human_blockers": 30000 / 2100.207579392558 / 2174.930950809533,  # block/m2
    }),
    # ... could add more dataset configs here
}

# get config according to data set name
def get_selected_config(data_set_name):
    if data_set_name in dataset_configs:
        dataset_config = dataset_configs[data_set_name]
        return EasyDict({**base_config, **dataset_config})
    else:
        raise ValueError(f"Data set '{data_set_name}' not found.")

# r:meters, 2d distance
# threshold: dB
def try_sensing_range(r, data_set_name):
    import math
    config = get_selected_config(data_set_name)
    p_los = math.exp(
        -config.density_of_human_blockers * config.diameter_of_human_blockers * r * (config.h_b - config.h_rx) / (
                config.h_d - config.h_rx))
    p_nlos = 1 - p_los
    PL_los = config.alpha_los + config.beta_los * 10 * math.log10(
        math.sqrt(r * r + config.h_d * config.h_d)) + config.zeta_los
    PL_nlos = config.alpha_nlos + config.beta_nlos * 10 * math.log10(
        math.sqrt(r * r + config.h_d * config.h_d)) + config.zeta_nlos
    PL = p_los * PL_los + p_nlos * PL_nlos
    CL = PL - config.g_tx - config.g_rx
    print(p_los, p_nlos)
    print(CL)


# Maximum Coupling Loss (110dB is recommended)
# purdue:

# 123dB -> 560m -> 60.5 range
# 121dB -> 420m -> 45.4 range
# 119dB -> 300m -> 32.4 range
# 117dB -> 215m -> 23.2 range √
# 115dB -> 140m -> 15 range

# ncsu:
# 123dB -> 600m -> 600 range
# 121dB -> 435m -> 435 range
# 119dB -> 315m -> 315 range
# 117dB -> 220m -> 220 range √
# 115dB -> 145m -> 145 range

# kaist:
# 123dB -> 600m -> 600 range
# 121dB -> 435m -> 435 range
# 119dB -> 315m -> 315 range
# 117dB -> 220m -> 220 range √
# 115dB -> 145m -> 145 range

# san:
# 123dB -> 600m -> 600 range
# 121dB -> 450m -> 450 range
# 119dB -> 330m -> 330 range
# 117dB -> 240m -> 240 range √
# 115dB -> 165m -> 165 range

if __name__ == '__main__':
    # example usage
    data_set_name = 'purdue'
    selected_config = get_selected_config(data_set_name)
    print(selected_config)