from collections import namedtuple
from operator import attrgetter
import numpy as np
import random
import logging
from easydict import EasyDict
import pysc2.env.sc2_env as sc2_env
from pysc2.env.sc2_env import SC2Env, Agent, MAX_STEP_COUNT, get_default, crop_and_deduplicate_names
from pysc2.lib import protocol
from s2clientprotocol import common_pb2 as sc_common, sc2api_pb2 as sc_pb, raw_pb2 as r_pb, debug_pb2 as d_pb
from ding.envs import BaseEnv
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY, deep_merge_dicts

from .smac_map import get_map_params

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

ally_types = {
    'marines': {
        0: 'marine',
    },
    'stalkers_and_zealots': {
        0: 'stalker',
        1: 'zealot',
    },
    'colossi_stalkers_zealots': {
        0: 'colossus',
        1: 'stalker',
        2: 'zealot',
    },
    'MMM': {
        0: 'marauder',
        1: 'marine',
        2: 'medivac',
    },
    'zealots': {
        0: 'zealot',
    },
    'hydralisks': {
        0: 'hydralisk',
    },
    'stalkers': {
        0: 'stalker',
    },
    'colossus': {
        0: 'colossus',
    },
    'bane': {
        0: 'baneling',
        1: 'zergling',
    },
}

enemey_types = {
    73: 'zealot',
    74: 'stalker',
    4: 'colossus',
    9: 'baneling',
    105: 'zergling',
    51: 'marauder',
    48: 'marine',
    54: 'medivac',
    107: 'hydralisk',
}

switcher = {
    'marine': 15,
    'marauder': 25,
    'medivac': 200,  # max energy
    'stalker': 35,
    'zealot': 22,
    'colossus': 24,
    'hydralisk': 10,
    'zergling': 11,
    'baneling': 1
}

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
    "parasitic_bomb": 2542,  # target: Unit
    'fungal_growth': 74,  # target: PointOrUnit
}

FORCE_RESTART_INTERVAL = 50000


@ENV_REGISTRY.register('smac_lz')
class SMACLZEnv(SC2Env, BaseEnv):
    """
    Overview:
        LightZero version of SMAC environment. This class includes methods for environment reset, step, and close. \
        It also includes methods for updating observations, units, states, and rewards. It also includes properties \
        for accessing the observation space, action space, and reward space of the environment. This environment \
        provides the interface for both single agent and multiple agents (two players) in SC2 environment.
    """

    SMACTimestep = namedtuple('SMACTimestep', ['obs', 'reward', 'done', 'info', 'episode_steps'])
    SMACEnvInfo = namedtuple('SMACEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space', 'episode_limit'])
    SMACActionInfo = namedtuple('SMACActionInfo', ['shape', 'value'])
    SMACRewardInfo = namedtuple('SMACRewardInfo', ['shape', 'value'])
    config = dict(
        difficulty=7,
        save_replay_episodes=None,
        game_steps_per_episode=None,
        reward_death_value=10,
        reward_win=200,
        reward_scale=20,  # if None, reward_scale will be self.reward_max
        reward_type='original',
    )

    def __repr__(self):
        return "LightZero SMAC Env"

    def __init__(
            self,
            cfg,
    ):
        """
        Overview:
            Initialize the environment with a configuration dictionary. Sets up spaces for observations, actions, and rewards.
        """
        cfg = deep_merge_dicts(EasyDict(self.config), cfg)
        self.cfg = cfg
        # Client parameters
        # necessary for compatibility with pysc2
        from absl import flags
        flags.FLAGS(['smac'])
        self.agent_interface_format = sc2_env.parse_agent_interface_format(use_raw_units=True)
        self.save_replay_episodes = cfg.save_replay_episodes
        assert (self.save_replay_episodes is None) or isinstance(
            self.save_replay_episodes, int
        )  # Denote the number of replays to save
        self.game_steps_per_episode = cfg.game_steps_per_episode

        # Map parameters
        map_name = cfg.map_name
        assert map_name is not None
        map_params = get_map_params(map_name)
        self._map_name = map_name
        self.map_type = map_params["map_type"]
        self.agent_race = map_params["a_race"]
        self.bot_race = map_params["b_race"]
        self.difficulty = cfg.difficulty
        self.players = [sc2_env.Agent(races[self.agent_race]), sc2_env.Bot(races[self.bot_race], self.difficulty)]
        self.n_agents = map_params["n_agents"]
        self.n_enemies = map_params["n_enemies"]
        self.n_entities = self.n_agents + self.n_enemies
        self._episode_limit = map_params["limit"]

        # Reset parameters
        self._seed = None
        self._launch_env_flag = True
        self._abnormal_env_flag = False

        # Counter parameters
        self._total_steps = 0
        self._next_reset_steps = FORCE_RESTART_INTERVAL
        self._won_count = 0
        self._episode_count = 0
        self._timeouts = 0
        self._force_restarts = 0

        # Reward parameters
        self.reward_pos_scale = 1.0
        self.reward_neg_scale = 0.0
        self.reward_death_value = cfg.reward_death_value
        self.reward_win = cfg.reward_win
        self.reward_defeat = 0
        self.reward_scale = cfg.reward_scale
        self.reward_max = 1020  # TODO: change after the env is lannched
        self.reward_type = cfg.reward_type

        # Action parameters
        self.n_actions_no_attack = 6
        self.move_amount = 2

    def _create_join(self):
        """
        Overview:
            Create the join requests for the agents. This function is called by the reset function.
        """
        # copy and overwrite original implementation
        map_inst = random.choice(self._maps)
        self._map_name = map_inst.name

        self._step_mul = max(1, self._default_step_mul or map_inst.step_mul)
        self._score_index = get_default(self._default_score_index,
                                        map_inst.score_index)
        self._score_multiplier = get_default(self._default_score_multiplier,
                                             map_inst.score_multiplier)
        self._episode_length = get_default(self._default_episode_length,
                                           map_inst.game_steps_per_episode)
        if self._episode_length <= 0 or self._episode_length > MAX_STEP_COUNT:
            self._episode_length = MAX_STEP_COUNT

        # Create the game. Set the first instance as the host.
        create = sc_pb.RequestCreateGame(
            disable_fog=self._disable_fog,
            realtime=self._realtime)

        if self._battle_net_map:
            create.battlenet_map_name = map_inst.battle_net
        else:
            create.local_map.map_path = map_inst.path
            map_data = map_inst.data(self._run_config)
            if self._num_agents == 1:
                create.local_map.map_data = map_data
            else:
                # Save the maps so they can access it. Don't do it in parallel since SC2
                # doesn't respect tmpdir on windows, which leads to a race condition:
                # https://github.com/Blizzard/s2client-proto/issues/102
                for c in self._controllers:
                    c.save_map(map_inst.path, map_data)
        if self._random_seed is not None:
            create.random_seed = self._random_seed
        for p in self._players:
            if isinstance(p, Agent):
                create.player_setup.add(type=sc_pb.Participant)
            else:
                create.player_setup.add(
                    type=sc_pb.Computer, race=random.choice(p.race),
                    difficulty=p.difficulty, ai_build=random.choice(p.build))
        if self._num_agents > 1:
            self._controllers[1].create_game(create)
        else:
            self._controllers[0].create_game(create)

        # Create the join requests.
        agent_players = [p for p in self._players if isinstance(p, Agent)]
        self.sanitized_names = crop_and_deduplicate_names(p.name for p in agent_players)
        join_reqs = []
        for p, name, interface in zip(agent_players, self.sanitized_names,
                                      self._interface_options):
            join = sc_pb.RequestJoinGame(options=interface)
            join.race = random.choice(p.race)
            join.player_name = name
            if self._ports:
                join.shared_port = 0  # unused
                join.server_ports.game_port = self._ports[0]
                join.server_ports.base_port = self._ports[1]
                for i in range(self._num_agents - 1):
                    join.client_ports.add(game_port=self._ports[i * 2 + 2],
                                          base_port=self._ports[i * 2 + 3])
            join_reqs.append(join)

        # Join the game. This must be run in parallel because Join is a blocking
        # call to the game that waits until all clients have joined.
        self._parallel.run((c.join_game, join)
                           for c, join in zip(self._controllers, join_reqs))

        self._game_info = self._parallel.run(c.game_info for c in self._controllers)
        for g, interface in zip(self._game_info, self._interface_options):
            if g.options.render != interface.render:
                logging.warning(
                    "Actual interface options don't match requested options:\n"
                    "Requested:\n%s\n\nActual:\n%s", interface, g.options)

        self._features = None

    def _launch(self):
        """
        Overview:
            Launch the environment. This function is called by the reset function.
        """
        self.old_unit_tags = set()
        print("*****LAUNCH FUNCTION CALLED*****")
        SC2Env.__init__(
            self,
            map_name=self.map_name,
            battle_net_map=False,
            players=self.players,
            agent_interface_format=self.agent_interface_format,
            discount=None,
            discount_zero_after_timeout=False,
            visualize=False,
            step_mul=8,
            realtime=False,
            save_replay_episodes=self.save_replay_episodes,
            replay_dir=None if self.save_replay_episodes is None else ".",
            replay_prefix=None,
            game_steps_per_episode=self.game_steps_per_episode,
            score_index=None,
            score_multiplier=None,
            random_seed=self._seed,
            disable_fog=False,
            ensure_available_actions=True,
            version=None
        )
        self._parallel.run((c.step, 2) for c in self._controllers)
        self._init_map()

    def _episode_restart(self):
        """
        Overview:
            Restart the environment by killing all units on the map.
            There is a trigger in the SC2Map file, which restarts the
            episode when there are no units left.
        """
        try:
            # save current units' tag
            self._update_obs()
            self.old_unit_tags = set(unit.tag for unit in self._obs.observation.raw_data.units)
            # kill current units
            run_commands = [
                (
                    self._controllers[0].debug,
                    d_pb.DebugCommand(
                        kill_unit=d_pb.DebugKillUnit(
                            tag=[unit.tag for unit in self._obs.observation.raw_data.units]
                        )
                    )
                )
            ]
            # Kill all units on the map.
            self._parallel.run(run_commands)
            # Forward 2 step to make sure all units revive.
            self._parallel.run((c.step, 2) for c in self._controllers)
        except (protocol.ProtocolError, protocol.ConnectionError) as e:
            print("Error happen in _restart. Error: ", e)
            self._env_restart()

    def _env_restart(self):
        self.close()
        self._launch()
        self._force_restarts += 1

    def reset(self):
        """
        Overview:
            Reset the environment. If it hasn't been initialized yet, this method also handles that. It also handles seeding \
            if necessary. Returns the first observation.
        """
        if self._launch_env_flag:
            # Launch StarCraft II
            print("*************LAUNCH TOTAL GAME********************")
            self._launch()
        elif self._abnormal_env_flag or (self._total_steps >= self._next_reset_steps) or (
                self.save_replay_episodes is not None):
            # Avoid hitting the real episode limit of SC2 env
            print("We are full restarting the environment! save_replay_episodes: ", self.save_replay_episodes)
            self._env_restart()
            self._next_reset_steps += FORCE_RESTART_INTERVAL
        else:
            self._episode_restart()

        init_flag = False
        for i in range(5):
            for j in range(10):
                self._update_obs()
                init_flag = self._init_units()
                if init_flag:
                    break
                else:
                    self._episode_restart()
            if init_flag:
                break
            else:
                self._env_restart()
        if not init_flag:
            raise RuntimeError("reset 5 times error")

        self._episode_steps = 0
        self._final_eval_fake_reward = 0.

        self._launch_env_flag = False
        self._abnormal_env_flag = False

        self._init_units_attr()
        self._init_rewards()
        self._init_states()
        ori_obs = self.get_obs()
        # process obs to marl_obs format
        obs_marl = {}
        # agent_state:
        obs_marl['agent_state'] = ori_obs['states'][:self.n_agents]
        # global_state:
        obs_marl['global_state'] = ori_obs['states'].flatten()
        # agent_specific_global_state
        obs_marl['agent_specific_global_state'] = np.concatenate(
            (obs_marl['agent_state'], np.repeat(obs_marl['global_state'].reshape(1, -1), self.n_agents, axis=0)),
            axis=1)
        ori_obs['states'] = obs_marl

        action_mask = None
        obs = {'observation': ori_obs, 'action_mask': action_mask, 'to_play': -1}
        return obs

    def _init_map(self):
        """
        Overview:
            Initialize the map. This function is called by the launch function.
        """
        game_info = self._game_info[0]
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y
        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(self.map_x, int(self.map_y / 8))
            self.pathing_grid = np.transpose(
                np.array([[(b >> i) & 1 for b in row for i in range(7, -1, -1)] for row in vals], dtype=np.bool_)
            )
        else:
            self.pathing_grid = np.invert(
                np.flip(
                    np.transpose(
                        np.array(list(map_info.pathing_grid.data), dtype=np.bool_).reshape(self.map_x, self.map_y)),
                    axis=1
                )
            )
        self.terrain_height = np.flip(
            np.transpose(np.array(list(map_info.terrain_height.data)).reshape(self.map_x, self.map_y)), 1
        ) / 255

    def _init_units(self):
        """
        Overview:
            Initialize the units. This function is called by the reset function. It checks if all units have been created \
            and if all units are healthy. If not, it returns False. Otherwise, it returns True.
        """
        # Sometimes not all units have yet been created by SC2 ToDO: check if use list not dict is a bug
        self.agents = [
            unit for unit in self._obs.observation.raw_data.units
            if (unit.owner == 1) and (unit.tag not in self.old_unit_tags)
        ]
        self.agents = sorted(
            self.agents,
            key=attrgetter("unit_type", "pos.x", "pos.y"),
            reverse=False,
        )
        self.enemies = [
            unit for unit in self._obs.observation.raw_data.units
            if (unit.owner == 2) and (unit.tag not in self.old_unit_tags)
        ]

        all_agents_created = (len(self.agents) == self.n_agents)
        all_enemies_created = (len(self.enemies) == self.n_enemies)
        all_agents_health = all(u.health > 0 for u in self.agents)
        all_enemies_health = all(u.health > 0 for u in self.enemies)

        if all_agents_created and all_enemies_created and all_agents_health and all_enemies_health:  # all good
            return True
        else:
            if not all_agents_created:
                print('not all agents created: {} vs {}'.format(len(self.agents), self.n_agents))
            if not all_agents_created:
                print('not all enemies created: {} vs {}'.format(len(self.enemies), self.n_enemies))
            if not all_agents_health:
                print('not all agents health')
            if not all_enemies_health:
                print('not all enemies health')
            return False

    def _init_units_attr(self):
        """
        Overview:
            Initialize the attributes of the units. This function is called by the reset function. It sets the unit types, \
            unit type ids, cooldowns, and shoot ranges.
        """
        # type
        self.min_unit_type = min([u.unit_type for u in self.agents])
        self.types = [ally_types[self.map_type][u.unit_type - self.min_unit_type] for u in self.agents] + \
                     [enemey_types[u.unit_type] for u in self.enemies]
        self.is_medivac = np.array([(t == 'medivac') for t in self.types], dtype=np.bool_)
        # type id
        type_to_id = {t: i for i, t in enumerate(sorted(list(set(self.types))))}
        self.unit_type_bits = len(type_to_id)
        self.type_id = np.array([type_to_id[t] for t in self.types], dtype=np.int64)
        # cooldown
        self.cooldown_max = np.array([switcher[t] for t in self.types], dtype=np.float32)
        # shoot_range
        self.shoot_range = np.array([6 for t in self.types], dtype=np.float32)

    def _init_rewards(self):
        """
        Overview:
            Initialize the rewards. This function is called by the reset function. It sets the rewards for injury, death, \
            and the end of the game. It also sets the maximum reward and the reward scale.
        """
        self.reward_injury = np.zeros(self.n_agents + self.n_enemies, dtype=np.float32)
        self.reward_dead = np.zeros(self.n_agents + self.n_enemies, dtype=np.float32)
        self.reward_end = 0
        if self.reward_type == 'original':
            self.reward_max = (self.n_enemies * self.reward_death_value + self.reward_win) + sum(
                [(u.health_max + u.shield_max) for u in self.enemies])
        elif self.reward_type == 'unit_norm':
            self.reward_max = max([(u.health_max + u.shield_max) for u in self.enemies])
        self.reward_scale = self.reward_max if self.reward_scale is None else self.reward_scale

    def _init_states(self):
        """
        Overview:
            Initialize the states. This function is called by the reset function. It sets the state length, the states, \
            the relations, and the actions.
        """
        self.state_len = 1 + self.unit_type_bits + 2 + 1 + 2 + 1 + 7 + 9 + 9  # ally or enemy, unit_type, pos.x and y, health, sheld(abs value and whether shield is zero), cooldown, last action(stop,move,attack), path, height
        self.relation_len = 2 + 2 + 2  # distance, cos(theta) and sin(theta), whether can attack
        self.action_len = self.n_actions_no_attack + self.n_entities  # (dead, stop, move) + help ally(currentlly only for medivac) + attack enemy
        # init satetes
        self.states = np.zeros((self.n_entities, self.state_len), dtype=np.float32)
        row_ind = np.arange(self.n_entities)
        self.states[:, 0] = (row_ind >= self.n_agents)  # ally or enemy
        self.states[row_ind, 1 + self.type_id] = 1  # unit_type
        self.alive_mask = np.ones(self.n_entities, dtype=np.bool_)
        # init relations
        self.relations = np.zeros((self.n_entities, self.n_entities, self.relation_len), dtype=np.float32)
        # init actions
        self.action_mask = np.zeros((self.n_agents, self.action_len), dtype=np.bool_)
        self.dead_action = np.array([[1] + [0] * (self.action_len - 1)], dtype=np.bool_)
        self.last_actions = np.ones(self.n_agents, dtype=np.int64)
        row_inds = np.arange(self.n_agents)
        # update states with current units
        self._update_states()
        self._eval_episode_return = 0

    def step(self, actions):
        """
        Overview:
            Take a step in the environment. This function is called by the reset function. It sets the action mask, \
            processes the actions, submits the actions, updates the observations, updates the units, updates the states, \
            gets the reward, and gets the info. It then returns the timestep.
        """
        processed_actions = self._process_actions(actions)
        try:
            self._submit_actions(processed_actions)
        except (protocol.ProtocolError, protocol.ConnectionError, ValueError) as e:
            print("Error happen in step! Error: ", e)
            self._abnormal_env_flag = True
            return self.SMACTimestep(obs=None, reward=None, done=True, info={'abnormal': True},
                                     episode_steps=self._episode_steps)
        self.last_actions = np.minimum(np.array(actions, dtype=np.int64), 6)  # dead, stop, N, S, E, W, attack
        self._total_steps += 1
        self._episode_steps += 1
        # Update states
        self._update_obs()
        game_end_code = self._update_units()
        self._update_states()
        # Get return
        reward = self.get_reward()
        self._final_eval_fake_reward += sum(reward)
        done, info = self.get_info(game_end_code)
        ori_obs = self.get_obs()
        # process obs to marl_obs format
        obs_marl = {}
        # agent_state:
        obs_marl['agent_state'] = ori_obs['states'][:self.n_agents]
        # global_state:
        obs_marl['global_state'] = ori_obs['states'].flatten()
        # agent_specific_global_state
        obs_marl['agent_specific_global_state'] = np.concatenate(
            (obs_marl['agent_state'], np.repeat(obs_marl['global_state'].reshape(1, -1), self.n_agents, axis=0)),
            axis=1)
        ori_obs['states'] = obs_marl

        self._eval_episode_return += reward
        if done:
            info['eval_episode_return'] = self._eval_episode_return
        action_mask = None
        obs = {'observation': ori_obs, 'action_mask': action_mask, 'to_play': -1}
        return BaseEnvTimestep(obs, reward, done, info)

    def _process_actions(self, my_actions):
        """
        Overview:
            Process the actions. This function is called by the step function. It constructs the action for the agent \
            based on the input action. It then returns the processed actions. The input action here is *absolute* and \
            is not mirrored! We use skip_mirror=True in get_avail_agent_actions to avoid error.
        """
        processed_actions = []
        for i, (unit, action) in enumerate(zip(self.agents, my_actions)):
            assert self.action_mask[i][action] == 1
            tag = unit.tag
            x = unit.pos.x
            y = unit.pos.y
            ma = self.move_amount
            offset = [[0, ma], [0, -ma], [ma, 0], [-ma, 0]]

            if action == 0:
                # no-op (valid only when dead)
                assert unit.health == 0, "No-op only available for dead agents."
                return None
            elif action == 1:
                # stop
                cmd = r_pb.ActionRawUnitCommand(ability_id=actions["stop"], unit_tags=[tag], queue_command=False)
            elif action in [2, 3, 4, 5]:
                # move
                o = offset[action - 2]
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=actions["move"],
                    target_world_space_pos=sc_common.Point2D(x=x + o[0], y=y + o[1]),
                    unit_tags=[tag],
                    queue_command=False
                )
            else:
                # attack or heal
                if self.map_type == "MMM" and self.is_medivac[i]:
                    target_unit = self.agents[action - self.n_actions_no_attack]
                    action_name = "heal"
                else:
                    target_unit = self.enemies[action - self.n_actions_no_attack - self.n_agents]
                    action_name = "attack"
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=actions[action_name], target_unit_tag=target_unit.tag, unit_tags=[tag],
                    queue_command=False
                )
            processed_actions.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd)))

        return processed_actions

    def _submit_actions(self, actions):
        """
        Overview:
            Submit the actions. This function is called by the step function. It sends the actions to the SC2 environment.
        """
        # actions is a sequence
        # Send action request
        req_actions = sc_pb.RequestAction(actions=actions)
        self._controllers[0].actions(req_actions)
        self._controllers[0].step(self._step_mul)

    def _update_obs(self):
        """
        Overview:
            Update the observations. This function is called by the step function. It gets the observations from the SC2 \
            environment and sets the observations to the environment observations.
        """
        # Transform in the thread so it runs while waiting for other observations.
        self._obs = self._controllers[0].observe()

    def _update_units(self):
        """
        Overview:
            Update units after an environment step. \
            This function assumes that self._obs is up-to-date.
        """
        n_ally_alive = 0
        n_enemy_alive = 0
        units = {unit.tag: unit for unit in self._obs.observation.raw_data.units}
        for a_id, a_unit in enumerate(self.agents):
            prev_health, prev_shield = a_unit.health, a_unit.shield
            if a_unit.tag in units:
                self.agents[a_id] = units[a_unit.tag]
            else:
                a_unit.health, a_unit.shield = 0, 0
            curr_health, curr_shield = self.agents[a_id].health, self.agents[a_id].shield
            self.reward_injury[a_id] = self.reward_neg_scale * (prev_health + prev_shield - curr_health - curr_shield)
            self.reward_dead[
                a_id] = self.reward_neg_scale * self.reward_death_value if prev_health > 0 and curr_health == 0 else 0
            if a_unit.health > 0 and not self.is_medivac[
                a_id]:  # only count entities capable of attacking, 54 is medivac.
                n_ally_alive += 1

        for e_id, e_unit in enumerate(self.enemies):
            prev_health, prev_shield = e_unit.health, e_unit.shield
            if e_unit.tag in units:
                self.enemies[e_id] = units[e_unit.tag]
            else:
                e_unit.health, e_unit.shield = 0, 0
            curr_health, curr_shield = self.enemies[e_id].health, self.enemies[e_id].shield
            self.reward_injury[e_id + self.n_agents] = self.reward_pos_scale * (
                    prev_health + prev_shield - curr_health - curr_shield)
            self.reward_dead[
                e_id + self.n_agents] = self.reward_pos_scale * self.reward_death_value if prev_health > 0 and curr_health == 0 else 0
            if e_unit.health > 0 and not self.is_medivac[
                e_id + self.n_agents]:  # only count entities capable of attacking, 54 is medivac.
                n_enemy_alive += 1

        self.reward_end = 0
        if n_ally_alive == 0 and n_enemy_alive > 0:
            self.reward_end = self.reward_defeat
            return -1  # lost
        elif n_ally_alive > 0 and n_enemy_alive == 0:
            self.reward_end = self.reward_win
            return 1  # won
        elif n_ally_alive == 0 and n_enemy_alive == 0:
            return 0  # draw
        else:
            return None  # not end

    def _update_states(self):
        """
        Overview:
            Update the states. This function is called by the step function. It updates the states, relations, and \
            actions. It also updates the surrounding terrain and the unit states.
        """
        # update unit states
        self.states[:, 1 + self.unit_type_bits + 6: 1 + self.unit_type_bits + 13] = 0
        self.states[np.arange(self.n_agents), 1 + self.unit_type_bits + 6 + self.last_actions] = 1
        for i, u in enumerate(self.agents + self.enemies):
            self.states[i, 1 + self.unit_type_bits] = u.pos.x
            self.states[i, 1 + self.unit_type_bits + 1] = u.pos.y
            self.states[i, 1 + self.unit_type_bits + 2] = u.health / 100.0
            self.states[i, 1 + self.unit_type_bits + 3] = u.shield / 100.0
            self.states[i, 1 + self.unit_type_bits + 4] = (u.shield > 0)
            self.states[i, 1 + self.unit_type_bits + 5] = u.weapon_cooldown / 50 if not self.is_medivac[
                i] else u.energy / 150
            self.alive_mask[i] = (u.health > 0)
        # update surrounding terrain
        pos = self.states[:, 1 + self.unit_type_bits:1 + self.unit_type_bits + 2]
        ma = self.move_amount
        border = np.array([[0, 0], [self.map_x, self.map_y]]).astype(np.int64)
        offset = np.array([
            [0, 2 * ma],
            [0, -2 * ma],
            [2 * ma, 0],
            [-2 * ma, 0],
            [ma, ma],
            [-ma, -ma],
            [ma, -ma],
            [-ma, ma],
            [0, 0],
            [0, ma / 2],  # move north
            [0, -ma / 2],  # move south
            [ma / 2, 0],  # move east
            [-ma / 2, 0],  # move west
        ])
        surround = (pos[:, None, :] + offset[None, :, :]).astype(np.int64)
        surround_clip = np.clip(surround, border[0], border[1] - 1)
        in_path = self.pathing_grid[surround_clip[:, :, 0], surround_clip[:, :, 1]]
        in_border = (surround >= border[0]).all(axis=-1) & (surround < border[1]).all(axis=-1)
        surround_path = in_path & in_border
        surround_height = self.terrain_height[surround_clip[:, :, 0], surround_clip[:, :, 1]]
        self.states[:, -18:] = np.concatenate([surround_path[:, :9], surround_height[:, :9]], axis=1)
        # update relations
        disp = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt(np.sum(disp * disp, axis=2))
        self.relations[:, :, 0] = np.minimum((dist - self.shoot_range[None, :]) / 6,
                                             3)  # minus enemy's shoot range, clip to < 3
        self.relations[:, :, 1] = np.minimum((dist - self.shoot_range[:, None]) / 6,
                                             3)  # minus ego's shoot range, clip to < 3
        self.relations[:, :, 2] = disp[:, :, 0] / (dist + 1e-8)  # cos(theta)
        self.relations[:, :, 3] = disp[:, :, 1] / (dist + 1e-8)  # sin(theta)
        self.relations[:, :, 4] = (dist <= self.shoot_range[None, :])  # whetehr can be shooted
        self.relations[:, :, 5] = (dist <= self.shoot_range[:, None])  # whetehr can shoot
        self.states[:, 1 + self.unit_type_bits] = self.states[:, 1 + self.unit_type_bits] / self.map_x
        self.states[:, 1 + self.unit_type_bits + 1] = self.states[:, 1 + self.unit_type_bits + 1] / self.map_y
        # update actions
        self.action_mask[:, 0] = 0  # dead action
        self.action_mask[:, 1] = 1  # stop action
        self.action_mask[:, 2:6] = surround_path[:self.n_agents, 9:]  # move action
        medivac_mask = self.is_medivac[:self.n_agents]
        alive_mask = self.alive_mask[:self.n_agents]
        shoot_mask = self.relations[: self.n_agents, :, 5].astype(bool) & self.alive_mask
        self.action_mask[~medivac_mask, 6 + self.n_agents:6 + self.n_entities] = shoot_mask[~medivac_mask,
                                                                                 self.n_agents:self.n_entities]  # attack action
        self.action_mask[medivac_mask, 6:6 + self.n_agents] = shoot_mask[medivac_mask,
                                                              :self.n_agents] & ~medivac_mask  # heal action
        self.action_mask[~alive_mask, :] = self.dead_action  # dead action

    def get_reward(self):
        """
        Overview:
            Get the reward. This function is called by the step function. It calculates the reward based on the injury, \
            death, and the end of the game. It then returns the reward.
        """
        reward = (self.reward_injury + self.reward_dead)[None, :].sum(axis=1) + self.reward_end
        reward *= self.reward_scale / self.reward_max
        return reward

    def get_obs(self):
        """
        Overview:
            Returns all agent observations in a list. This function is called by the step function. It returns the \
            observations for each agent.
            NOTE: Agents should have access only to their local observations
            during decentralised execution.
        """
        obs = {
            'states': self.states.copy(),
            'relations': self.relations.copy(),
            'action_mask': self.action_mask.copy(),
            'alive_mask': self.alive_mask.copy(),
        }
        return obs

    # def get_avail_agent_actions(self, agent_id):
    #     return self.action_mask[agent_id]

    def get_info(self, game_end_code):
        """
        Overview:
            This function is called only once at each step, no matter whether you take opponent as agent.
            We already return dicts for each term, as in Multi-agent scenario.
        """
        info = {
            "battle_won": False,
            "battle_lost": False,
            "draw": False,
            'final_eval_reward': 0.,
            'final_eval_fake_reward': 0.
        }
        done = False

        if game_end_code is not None:
            done = True
            if game_end_code == 1:
                self._won_count += 1
                info["battle_won"] = True
                info['final_eval_reward'] = 1.
            elif game_end_code == -1:
                info["battle_lost"] = True
            else:
                info["draw"] = True
        elif self._episode_steps >= self._episode_limit:
            done = True
            self._timeouts += 1

        if done:
            self._episode_count += 1
            dead_allies = sum(~self.alive_mask[:self.n_agents])
            dead_enemies = sum(~self.alive_mask[self.n_agents:self.n_entities])
            info['final_eval_fake_reward'] = self._final_eval_fake_reward
            info['dead_allies'] = dead_allies
            info['dead_enemies'] = dead_enemies
            info['episode_info'] = {
                'final_eval_fake_reward': info['final_eval_fake_reward'],
                'dead_allies': dead_allies,
                'dead_enemies': dead_enemies
            }
        return done, info

    def info(self):
        """
        Overview:
            Return the environment information. This function is called by the reset function. It returns the number of \
            agents, the number of enemies, the observation space, the action space, the reward space, and the episode limit.
        """
        agent_num = self.n_agents
        enemy_num = self.n_enemies
        obs_space = {}  # TODO: Now, obs_space is only accessible after self.reset().
        obs = self.reset()
        for k, v in obs.items():
            obs_space[k] = v.shape
        self.close()
        obs_space = EnvElementInfo(obs_space, None)
        return self.SMACEnvInfo(
            agent_num=agent_num,
            obs_space=obs_space,
            act_space=self.SMACActionInfo((self.n_agents,), {'min': 0, 'max': 1}),
            rew_space=self.SMACRewardInfo((1,), {'min': 0, 'max': self.reward_max}),
            episode_limit=self._episode_limit,
        )

    def seed(self, seed, dynamic_seed=False):
        """
        Overview:
            Set the seed. This function is called by the reset function. It sets the seed for the environment.
        """
        self._seed = seed
        if self.cfg.get('subprocess_numpy_seed', False):
            np.random.seed(self._seed)

    def close(self):
        """
        Overview:
            Close the environment.
        """
        SC2Env.close(self)


SMACTimestep = SMACLZEnv.SMACTimestep
SMACEnvInfo = SMACLZEnv.SMACEnvInfo
SMACActionInfo = SMACLZEnv.SMACActionInfo
SMACRewardInfo = SMACLZEnv.SMACRewardInfo
