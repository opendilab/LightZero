import numpy as np

np.seterr(invalid='ignore')

from zoo.CrowdSim.envs.Crowdsim.env.model.agent import *
from zoo.CrowdSim.envs.Crowdsim.env.model.mdp import JointState
from shapely.geometry import *


def tensor_to_joint_state(state, config):  # 恢复原先尺度
    robot_states, human_states = state

    robot_states = robot_states.cpu().squeeze(0).data.numpy()
    robot_states = [RobotState(robot_state[0] * config.nlon,
                               robot_state[1] * config.nlat,
                               robot_state[2] * config.rotation_limit,
                               robot_state[3] * config.max_uav_energy) for robot_state in robot_states]

    human_states = human_states.cpu().squeeze(0).data.numpy()
    human_states = [HumanState(human_state[0] * config.nlon,
                               human_state[1] * config.nlat,
                               human_state[2] * config.rotation_limit,
                               human_state[3] * config.num_timestep) for human_state in human_states]

    return JointState(robot_states, human_states)


def tensor_to_robot_states(robot_state_tensor, config):
    robot_states = robot_state_tensor.cpu().squeeze(0).data.numpy()
    robot_states = [RobotState(robot_state[0] * config.nlon,
                               robot_state[1] * config.nlat,
                               robot_state[2] * config.rotation_limit,
                               robot_state[3] * config.max_uav_energy) for robot_state in robot_states]
    return robot_states


def get_human_position_list(selected_timestep, human_df, config):
    # config.step_time means the time interval between two timesteps
    selected_timestamp = config.start_timestamp + selected_timestep * config.step_time
    selected_data = human_df[human_df.timestamp == selected_timestamp]
    selected_data = selected_data.set_index("id")

    if selected_timestep < config.num_timestep:
        selected_next_data = human_df[human_df.timestamp == (selected_timestamp + config.step_time)]
        selected_next_data = selected_next_data.set_index("id")
    else:
        selected_next_data = None

    return selected_data, selected_next_data


def get_human_position_from_list(selected_timestep, human_id, selected_data, selected_next_data, config):
    px, py = selected_data.loc[human_id, ["x", "y"]]

    if selected_timestep < config.num_timestep:
        npx, npy = selected_next_data.loc[human_id, ["x", "y"]]
        theta = get_theta(0, 0, npx - px, npy - py)
        # print(px, py, npx, npy, theta)
    else:
        theta = 0

    return px, py, theta


def judge_aoi_update(human_position, robot_position, config):
    """
    Overview:
        Judge whether the AoI should be updated, i.e., the human is in the sensing range of the robot.
    Args:
        - human_position (:obj:`list`): The position of the human.
        - robot_position (:obj:`list`): The position of the robot.
        - config (:obj:`dict`): The configuration of the environment.
    """
    should_reset = False
    for robot_id in range(config.robot_num):
        unit_distance = np.sqrt(np.power(robot_position[robot_id][0] - human_position[0], 2)
                                + np.power(robot_position[robot_id][1] - human_position[1], 2))
        if unit_distance <= config.sensing_range:
            should_reset = True
            break

    return should_reset


def inPoly(polygon, x, y):
    pt = (x, y)
    line = LineString(polygon)
    point = Point(pt)
    polygon = Polygon(line)
    return polygon.contains(point)


def iscrosses(line1, line2):
    if LineString(line1).crosses(LineString(line2)):
        return True
    return False


def crossPoly(square, x1, y1, x2, y2):
    our_line = LineString([[x1, y1], [x2, y2]])
    line1 = LineString([square[0], square[2]])
    line2 = LineString([square[1], square[3]])
    if our_line.crosses(line1) or our_line.crosses(line2):
        return True
    else:
        return False


def judge_collision(new_robot_px, new_robot_py, old_robot_px, old_robot_py, config):
    if config.no_fly_zone is None:
        return False

    for square in config.no_fly_zone:
        if inPoly(square, new_robot_px, new_robot_py):
            return True
        if crossPoly(square, new_robot_px, new_robot_py, old_robot_px, old_robot_py):
            return True
    return False


def get_theta(x1, y1, x2, y2):
    ang1 = np.arctan2(y1, x1)
    ang2 = np.arctan2(y2, x2)
    theta = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    return theta


def consume_uav_energy(fly_time, hover_time, config):
    # configs
    Pu = 0.5  # the average transmitted power of each user, W,  e.g. mobile phone
    P0 = 79.8563  # blade profile power, W
    P1 = 88.6279  # derived power, W
    U_tips = 120  # tip speed of the rotor blade of the UAV,m/s
    v0 = 4.03  # the mean rotor induced velocity in the hovering state,m/s
    d0 = 0.6  # fuselage drag ratio
    rho = 1.225  # density of air,kg/m^3
    s0 = 0.05  # the rotor solidity
    A = 0.503  # the area of the rotor disk, m^2
    Vt = config.velocity  # velocity of the UAV,m/s

    Power_flying = P0 * (1 + 3 * Vt ** 2 / U_tips ** 2) + \
                   P1 * np.sqrt((np.sqrt(1 + Vt ** 4 / (4 * v0 ** 4)) - Vt ** 2 / (2 * v0 ** 2))) + \
                   0.5 * d0 * rho * s0 * A * Vt ** 3

    Power_hovering = P0 + P1

    return fly_time * Power_flying + hover_time * Power_hovering


def get_border(ur, lf):
    upper_left = [lf[0], ur[1]]
    upper_right = [ur[0], ur[1]]
    lower_right = [ur[0], lf[1]]
    lower_left = [lf[0], lf[1]]

    coordinates = [
        upper_left,
        upper_right,
        lower_right,
        lower_left,
        upper_left
    ]

    geo_json = {"type": "FeatureCollection",
                "properties": {
                    "lower_left": lower_left,
                    "upper_right": upper_right
                },
                "features": []}

    grid_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates],
        }
    }

    geo_json["features"].append(grid_feature)

    return geo_json


def traj_to_timestamped_geojson(index, trajectory, robot_num, color):
    point_gdf = trajectory.df.copy()
    point_gdf["previous_geometry"] = point_gdf["geometry"].shift()
    point_gdf["time"] = point_gdf.index
    point_gdf["previous_time"] = point_gdf["time"].shift()

    features = []

    # for Point in GeoJSON type
    for _, row in point_gdf.iterrows():
        corrent_point_coordinates = [
            row["geometry"].xy[0][0],
            row["geometry"].xy[1][0]
        ]
        current_time = [row["time"].isoformat()]

        if index < robot_num:
            radius = 8  # 125(5 units)
            opacity = 0.05
            popup_html = f'<h4>UAV {int(row["id"])}</h4>' + f'<p>raw coord: {corrent_point_coordinates}</p>' \
                         + f'<p>grid coord: ({row["x"]},{row["y"]})</p>' \
                         + f'<p>dist coord: ({row["x_distance"]}m, {row["y_distance"]}m)</p>' \
                         + f'<p>energy: {row["energy"]}J </p>'
        else:
            radius = 2
            opacity = 1
            popup_html = f'<h4>Human {int(row["id"])}</h4>' + f'<p>raw coord: {corrent_point_coordinates}</p>' \
                         + f'<p>grid coord: ({row["x"]},{row["y"]})</p>' \
                         + f'<p>dist coord: ({row["x_distance"]}m, {row["y_distance"]}m)</p>' \
                         + f'<p>aoi: {int(row["aoi"])} </p>'

        # for Point in GeoJSON type  (Temporally Deprecated)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": corrent_point_coordinates,
                },
                "properties": {
                    "times": current_time,
                    'popup': popup_html,
                    "icon": 'circle',  # point
                    "iconstyle": {
                        'fillColor': color,
                        'fillOpacity': opacity,  # 透明度
                        'stroke': 'true',
                        'radius': radius,
                        'weight': 1,
                    },

                    "style": {  # line
                        "color": color,
                    },
                    "code": 11,

                },
            }
        )
    return features


if __name__ == "__main__":
    print(judge_collision(new_robot_px=6505, new_robot_py=5130,
                          old_robot_px=6925, old_robot_py=5130))
