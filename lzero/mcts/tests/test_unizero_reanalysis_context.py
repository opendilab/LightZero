from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lzero.mcts.buffer.game_buffer_unizero as unizero_buffer_module
import lzero.mcts.buffer.game_buffer_sampled_unizero as sampled_unizero_buffer_module
from lzero.model.unizero_world_models.world_model import WorldModel
from lzero.policy import DiscreteSupport
from lzero.mcts.buffer.game_buffer_unizero import (
    UniZeroGameBuffer,
    _world_model_reanalysis_phase,
)
from lzero.mcts.buffer.game_buffer_sampled_unizero import (
    SampledUniZeroGameBuffer,
)


class _TargetModelStub:

    def to(self, device):
        return self

    def eval(self):
        return self


def test_unizero_sample_uses_current_reanalysis_policy_api():
    buffer = UniZeroGameBuffer.__new__(UniZeroGameBuffer)
    buffer._cfg = SimpleNamespace(device='cpu', reanalyze_ratio=1)
    buffer.action_space_size = 2
    actions = np.zeros((1, 1), dtype=np.int64)
    current_batch = [None, actions, None, None, None, None, None, np.zeros(1, dtype=np.int64)]
    buffer._make_batch = lambda batch_size, ratio: ('reward', 'policy-re', 'policy-non-re', current_batch)
    buffer._compute_target_reward_value = lambda *args: (np.zeros(1), np.zeros(1))
    buffer._compute_target_policy_non_reanalyzed = lambda *args: np.zeros((0, 2))
    received = []

    def compute_reanalyzed(context, model, batch_action):
        received.append((context, model, batch_action))
        return np.zeros((1, 2))

    buffer._compute_target_policy_reanalyzed = compute_reanalyzed
    model = _TargetModelStub()
    policy = SimpleNamespace(_target_model=model)

    train_data = buffer.sample(1, policy)

    assert received == [('policy-re', model, actions)]
    assert train_data[1][2].shape == (1, 2)


def test_unizero_buffer_refresh_uses_current_reanalysis_policy_api():
    buffer = UniZeroGameBuffer.__new__(UniZeroGameBuffer)
    buffer._cfg = SimpleNamespace(device='cpu')
    actions = np.zeros((1, 1), dtype=np.int64)
    current_batch = [None, actions, None, None, None, None, None, np.zeros(1, dtype=np.int64)]
    buffer._make_batch_for_reanalyze = lambda batch_size: ('policy-re', current_batch)
    received = []
    buffer._compute_target_policy_reanalyzed = lambda *args: received.append(args)
    model = _TargetModelStub()

    buffer.reanalyze_buffer(1, SimpleNamespace(_target_model=model))

    assert received == [('policy-re', model, actions)]


@pytest.mark.parametrize('buffer_cls', [UniZeroGameBuffer, SampledUniZeroGameBuffer])
def test_reanalysis_timesteps_align_with_flattened_h_plus_one_roots(buffer_cls):
    buffer = buffer_cls.__new__(buffer_cls)
    buffer._cfg = SimpleNamespace(num_unroll_steps=2)

    root_timesteps, sequence_starts = buffer._preprocess_reanalyze_timesteps(
        [np.array([10, 11, 12, 13]), np.array([30, 31])],
        [1, 1],
    )

    assert root_timesteps.dtype == sequence_starts.dtype == np.int64
    np.testing.assert_array_equal(sequence_starts, [11, 31])
    np.testing.assert_array_equal(root_timesteps, [11, 12, 13, 31, 0, 0])


def test_reanalysis_position_normalization_handles_flat_and_legacy_matrices():
    flat = WorldModel._flatten_reanalysis_positions([10, 11, 12])
    legacy_matrix = WorldModel._flatten_reanalysis_positions(
        [[10, 11], [20, 21]], append_terminal=True
    )

    np.testing.assert_array_equal(flat, [10, 11, 12])
    np.testing.assert_array_equal(legacy_matrix, [10, 11, 0, 20, 21, 0])


def test_contextual_multitask_task_embedding_fails_during_buffer_setup_validation():
    buffer = UniZeroGameBuffer.__new__(UniZeroGameBuffer)
    buffer.task_id = 0
    buffer._cfg = SimpleNamespace(
        contextual_reanalysis=True,
        model=SimpleNamespace(
            world_model_cfg=SimpleNamespace(task_embed_option='concat_task_embed')
        ),
    )

    with pytest.raises(NotImplementedError, match='task-token conditioning'):
        buffer._validate_contextual_reanalysis_config()


def test_reanalysis_phase_is_restored_after_failure():
    world_model = SimpleNamespace(reanalyze_phase=False)

    try:
        with _world_model_reanalysis_phase(world_model):
            assert world_model.reanalyze_phase is True
            raise RuntimeError('synthetic MCTS failure')
    except RuntimeError:
        pass

    assert world_model.reanalyze_phase is False


@pytest.mark.parametrize('buffer_cls', [UniZeroGameBuffer, SampledUniZeroGameBuffer])
def test_bootstrap_legacy_diagnostic_cadence_is_sparse(buffer_cls):
    buffer = buffer_cls.__new__(buffer_cls)
    assert buffer._bootstrap_value_context_diagnostic_due() is True
    assert all(
        buffer._bootstrap_value_context_diagnostic_due() is False
        for _ in range(998)
    )
    assert buffer._bootstrap_value_context_diagnostic_due() is True


def test_direct_bootstrap_encoder_uses_the_requested_task_tokenizer():
    calls = []

    class _Tokenizer:

        def encode_to_obs_embeddings(self, observations, task_id=None):
            calls.append(task_id)
            return observations + task_id

    buffer = UniZeroGameBuffer.__new__(UniZeroGameBuffer)
    buffer.task_id = 2
    observations = torch.zeros(3, 4)
    encoded = buffer._encode_bootstrap_root_latents(
        SimpleNamespace(world_model=SimpleNamespace(tokenizer=_Tokenizer())),
        observations,
    )

    assert calls == [2]
    assert torch.equal(encoded, torch.full_like(observations, 2))


@pytest.mark.parametrize('buffer_cls', [UniZeroGameBuffer, SampledUniZeroGameBuffer])
def test_contextual_bootstrap_skips_legacy_transformer_between_diagnostics(
        monkeypatch, buffer_cls
):
    initial_calls = []
    direct_encode_calls = []

    class _Tokenizer:

        def encode_to_obs_embeddings(self, observations):
            direct_encode_calls.append(len(observations))
            return torch.zeros(len(observations), 1, 4)

    class _WorldModel:
        tokenizer = _Tokenizer()

        def build_reanalysis_root_token_contexts(self, latent_state_roots, **kwargs):
            return [torch.zeros(1, 4) for _ in range(len(latent_state_roots))]

        def evaluate_root_token_context_values(self, contexts, task_id=None):
            return torch.zeros(len(contexts), 3)

    class _Model:
        world_model = _WorldModel()

        def initial_inference(self, observations, actions, start_pos=None):
            initial_calls.append(len(observations))
            return SimpleNamespace(
                latent_state=np.zeros((len(observations), 1, 4), dtype=np.float32),
                value=torch.zeros(len(observations), 3),
                policy_logits=torch.zeros(len(observations), 2),
            )

    monkeypatch.setattr(
        unizero_buffer_module, 'prepare_observation', lambda value, *_: np.asarray(value)
    )
    monkeypatch.setattr(
        sampled_unizero_buffer_module, 'prepare_observation',
        lambda value, *_: np.asarray(value),
    )
    buffer = buffer_cls.__new__(buffer_cls)
    buffer.task_id = None
    buffer.action_space_size = 2
    buffer.value_support = DiscreteSupport(-1, 2)
    buffer._cfg = SimpleNamespace(
        model=SimpleNamespace(
            model_type='unit_test', continuous_action_space=False
        ),
        device='cpu',
        use_root_value=False,
        bootstrap_value_context=True,
        num_unroll_steps=1,
        env_type='not_board_games',
        discount_factor=1.0,
    )
    buffer._preprocess_to_play_and_action_mask = lambda *args: (
        [-1, -1], np.ones((2, 2), dtype=np.int8)
    )
    context = (
        np.zeros((2, 1), dtype=np.float32),
        np.ones(2, dtype=np.float32),
        [0],
        [np.zeros(2, dtype=np.float32)],
        [0.0, 0.0],
        [2],
        np.ones(2, dtype=np.int64),
        [None],
        [[-1, -1]],
        [[]],
        [[]],
    )
    actions = np.zeros((1, 1), dtype=np.int64)
    timesteps = np.zeros(1, dtype=np.int64)

    diagnostic_targets = buffer._compute_target_reward_value(
        context, _Model(), actions, timesteps
    )
    fast_targets = buffer._compute_target_reward_value(
        context, _Model(), actions, timesteps
    )

    assert initial_calls == [2]
    assert direct_encode_calls == [2]
    assert all(np.array_equal(left, right) for left, right in zip(
        diagnostic_targets, fast_targets
    ))


@pytest.mark.parametrize('buffer_cls', [UniZeroGameBuffer, SampledUniZeroGameBuffer])
def test_contextual_reanalysis_is_opt_in_and_recovers_history(buffer_cls):
    class _Game:

        def __init__(self):
            self.obs_segment = np.arange(9, dtype=np.float32).reshape(9, 1)
            self.action_segment = np.arange(8, dtype=np.int64)
            self.reward_segment = np.zeros(8, dtype=np.float32)
            self.action_mask_segment = [np.ones(2, dtype=np.int8)] * 8
            self.to_play_segment = [-1] * 8
            self.timestep_segment = np.arange(8)
            self.child_visit_segment = [[0.5, 0.5] for _ in range(8)]
            self.root_sampled_actions = [[0, 1] for _ in range(8)]
            self.root_value_segment = np.zeros(8, dtype=np.float32)

        def __len__(self):
            return 8

        def zero_obs(self):
            return [np.zeros((1,), dtype=np.float32)]

        def get_unroll_obs(self, timestep, num_unroll_steps=0, padding=False):
            return self.obs_segment[timestep:timestep + 1 + num_unroll_steps]

    buffer = buffer_cls.__new__(buffer_cls)
    buffer._cfg = SimpleNamespace(
        num_unroll_steps=2,
        contextual_reanalysis=True,
        model=SimpleNamespace(
            frame_stack_num=1,
            world_model_cfg=SimpleNamespace(context_length=10),
        ),
    )
    context = buffer._prepare_policy_reanalyzed_context([0], [_Game()], [5])
    history_observations, history_actions = context[-2:]

    assert history_actions == [[0, 1, 2, 3, 4]]
    assert [float(obs[0, 0]) for obs in history_observations[0]] == [0., 1., 2., 3., 4.]

    buffer._cfg.contextual_reanalysis = False
    legacy_context = buffer._prepare_policy_reanalyzed_context([0], [_Game()], [5])
    assert legacy_context[-2:] == [[[]], [[]]]


def test_bootstrap_context_diagnostics_exclude_legacy_h_plus_one_placeholders(monkeypatch):
    records = []
    monkeypatch.setattr(
        unizero_buffer_module.logging,
        'info',
        lambda *args, **kwargs: records.append(args),
    )
    buffer = UniZeroGameBuffer.__new__(UniZeroGameBuffer)
    buffer._cfg = SimpleNamespace(num_unroll_steps=2)
    buffer._bootstrap_value_context_batch_count = 1
    legacy = np.array([0., 0., 100., 0., 0., 100.])
    contextual = np.zeros(6)
    contexts = [torch.zeros(index + 1, 4) for index in range(6)]

    buffer._log_bootstrap_value_context_diagnostics(
        legacy, contextual, np.ones(6, dtype=bool), contexts
    )

    assert len(records) == 1
    assert records[0][2] == 4  # valid roots after dropping each final placeholder
    assert records[0][9] == 0.0  # max absolute delta excludes placeholder-only spikes


def test_bootstrap_value_context_recovers_history_before_td_state():
    class _Game:

        def __init__(self):
            self.obs_segment = np.arange(10, dtype=np.float32).reshape(10, 1)
            self.action_segment = np.arange(9, dtype=np.int64)
            self.reward_segment = np.zeros(9, dtype=np.float32)
            self.action_mask_segment = [np.ones(2, dtype=np.int8)] * 9
            self.to_play_segment = [-1] * 9

        def __len__(self):
            return 9

        def zero_obs(self):
            return [np.zeros((1,), dtype=np.float32)]

        def get_unroll_obs(self, timestep, num_unroll_steps=0, padding=False):
            return self.obs_segment[timestep:timestep + 1 + num_unroll_steps]

    buffer = UniZeroGameBuffer.__new__(UniZeroGameBuffer)
    buffer._cfg = SimpleNamespace(
        td_steps=2,
        num_unroll_steps=2,
        bootstrap_value_context=True,
        model=SimpleNamespace(
            frame_stack_num=1,
            world_model_cfg=SimpleNamespace(context_length=10),
        ),
    )
    context = buffer._prepare_reward_value_context([0], [_Game()], [1], 9)
    history_observations, history_actions = context[-2:]

    # The sampled state is t=1 and TD bootstrap starts at t=3.  The recovered
    # prefix must therefore include transitions 0, 1 and 2, not stop at t=1.
    assert history_actions == [[0, 1, 2]]
    assert [float(obs[0, 0]) for obs in history_observations[0]] == [0., 1., 2.]


def test_default_bootstrap_target_does_not_prepare_unused_replay_history():
    class _Game:

        def __init__(self):
            self.obs_segment = np.arange(10, dtype=np.float32).reshape(10, 1)
            self.action_segment = np.arange(9, dtype=np.int64)
            self.reward_segment = np.zeros(9, dtype=np.float32)
            self.action_mask_segment = [np.ones(2, dtype=np.int8)] * 9
            self.to_play_segment = [-1] * 9
            self.get_unroll_obs_calls = 0

        def __len__(self):
            return 9

        def zero_obs(self):
            return [np.zeros((1,), dtype=np.float32)]

        def get_unroll_obs(self, timestep, num_unroll_steps=0, padding=False):
            self.get_unroll_obs_calls += 1
            return self.obs_segment[timestep:timestep + 1 + num_unroll_steps]

    game = _Game()
    buffer = UniZeroGameBuffer.__new__(UniZeroGameBuffer)
    buffer._cfg = SimpleNamespace(
        td_steps=2,
        num_unroll_steps=2,
        bootstrap_value_context=False,
        model=SimpleNamespace(
            frame_stack_num=1,
            world_model_cfg=SimpleNamespace(context_length=10),
        ),
    )

    context = buffer._prepare_reward_value_context([0], [game], [1], 9)

    assert len(context) == 9
    assert game.get_unroll_obs_calls == 1


def test_atari_unizero_reanalysis_passes_h_plus_one_positions_and_restores_phase(monkeypatch):
    captured = {}

    class _Roots:
        num = 2

        def prepare(self, *args):
            pass

    class _FailingMCTS:

        def __init__(self, cfg):
            pass

        @staticmethod
        def roots(root_num, legal_actions):
            assert root_num == 2
            return _Roots()

        def search(self, roots, model, latent_state_roots, to_play, timestep):
            captured['root_timesteps'] = timestep
            raise RuntimeError('synthetic MCTS failure')

    class _Model:

        def __init__(self):
            self.world_model = SimpleNamespace(reanalyze_phase=False, clear_caches=lambda: None)

        def initial_inference(self, batch_obs, batch_action, start_pos):
            captured['sequence_start_timesteps'] = start_pos
            return SimpleNamespace(
                latent_state=torch.zeros(2, 1, 4),
                value=torch.zeros(2, 1),
                policy_logits=torch.zeros(2, 2),
            )

    monkeypatch.setattr(unizero_buffer_module, 'MCTSCtree', _FailingMCTS)
    monkeypatch.setattr(unizero_buffer_module, 'prepare_observation', lambda obs, model_type: np.asarray(obs))
    monkeypatch.setattr(unizero_buffer_module, 'inverse_scalar_transform', lambda value, support: value)
    monkeypatch.setattr(
        unizero_buffer_module,
        'concat_output',
        lambda outputs, data_type: (
            None,
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 1, 4), dtype=np.float32),
        ),
    )

    buffer = UniZeroGameBuffer.__new__(UniZeroGameBuffer)
    buffer._cfg = SimpleNamespace(
        num_unroll_steps=1,
        model=SimpleNamespace(continuous_action_space=False, model_type='conv'),
        device='cpu',
        root_dirichlet_alpha=0.3,
        root_noise_weight=0.25,
        mcts_ctree=True,
        reanalyze_search_chunk_size=2,
        env_type='not_board_games',
    )
    buffer.action_space_size = 2
    buffer.reanalyze_num = 1
    buffer.task_id = None
    buffer.value_support = object()
    buffer._preprocess_to_play_and_action_mask = lambda *args: (
        [-1, -1],
        [np.ones(2, dtype=np.int8), np.ones(2, dtype=np.int8)],
    )
    policy_context = [
        [np.zeros((3, 4, 4)), np.zeros((3, 4, 4))],
        [1, 1],
        [0],
        [0],
        [[[0.5, 0.5], [0.5, 0.5]]],
        [2],
        [[[1, 1], [1, 1]]],
        [[-1, -1]],
        [np.array([10, 11])],
    ]
    model = _Model()

    with pytest.raises(RuntimeError, match='synthetic MCTS failure'):
        buffer._compute_target_policy_reanalyzed(
            policy_context, model, np.zeros((1, 1), dtype=np.int64)
        )

    np.testing.assert_array_equal(captured['sequence_start_timesteps'], [10])
    np.testing.assert_array_equal(captured['root_timesteps'], [10, 11])
    assert model.world_model.reanalyze_phase is False


def test_atari_unizero_reanalysis_chunks_roots_to_online_cache_capacity(monkeypatch):
    searched_timestep_chunks = []

    class _Roots:

        def __init__(self, legal_actions):
            self.legal_actions = legal_actions

        def prepare(self, *args):
            pass

        def get_distributions(self):
            return [[actions[0]] for actions in self.legal_actions]

    class _ChunkedMCTS:

        def __init__(self, cfg):
            pass

        @staticmethod
        def roots(root_num, legal_actions):
            assert root_num == len(legal_actions)
            return _Roots(legal_actions)

        def search(self, roots, model, latent_state_roots, to_play, timestep):
            assert len(latent_state_roots) == len(to_play) == len(timestep)
            searched_timestep_chunks.append(list(timestep))

    monkeypatch.setattr(unizero_buffer_module, 'MCTSCtree', _ChunkedMCTS)
    clear_count = {'value': 0}
    world_model = SimpleNamespace(
        clear_caches=lambda: clear_count.__setitem__('value', clear_count['value'] + 1)
    )
    buffer = UniZeroGameBuffer.__new__(UniZeroGameBuffer)
    buffer._cfg = SimpleNamespace(
        mcts_ctree=True,
        reanalyze_search_chunk_size=2,
        root_noise_weight=0.25,
        model=SimpleNamespace(world_model_cfg=SimpleNamespace(env_num=8)),
    )
    buffer.task_id = None

    distributions = buffer._search_reanalyzed_roots_in_chunks(
        model=SimpleNamespace(world_model=world_model),
        latent_state_roots=np.arange(5).reshape(5, 1),
        legal_actions=[[0], [1], [2], [3], [4]],
        noises=[[0.0]] * 5,
        reward_pool=[0.0] * 5,
        policy_logits_pool=[[0.0]] * 5,
        to_play=[-1] * 5,
        root_timesteps=[10, 11, 12, 13, 14],
    )

    assert searched_timestep_chunks == [[10, 11], [12, 13], [14]]
    assert distributions == [[0], [1], [2], [3], [4]]
    assert clear_count['value'] == 3


def test_atari_unizero_reanalysis_seeds_each_root_chunk_before_search(monkeypatch):
    events = []
    prepared_policy_logits = []

    class _Roots:

        def __init__(self, root_num):
            self.num = root_num

        def prepare(self, root_noise_weight, noises, rewards, policy_logits, to_play):
            prepared_policy_logits.extend(policy_logits)

        def get_distributions(self):
            return [[1] for _ in range(self.num)]

    class _MCTS:

        def __init__(self, cfg):
            pass

        @staticmethod
        def roots(root_num, legal_actions):
            return _Roots(root_num)

        def search(self, roots, model, latent_state_roots, to_play, timestep):
            events.append(('search', list(np.asarray(latent_state_roots).reshape(-1))))

    class _WorldModel:
        env_num = 2

        def clear_caches(self):
            events.append(('clear',))

        def seed_reanalysis_root_caches(self, roots, contexts):
            events.append((
                'seed',
                list(np.asarray(roots).reshape(-1)),
                list(contexts),
            ))
            return [[float(value + 10)] for value in np.asarray(roots).reshape(-1)]

    monkeypatch.setattr(unizero_buffer_module, 'MCTSCtree', _MCTS)
    buffer = UniZeroGameBuffer.__new__(UniZeroGameBuffer)
    buffer._cfg = SimpleNamespace(
        mcts_ctree=True,
        reanalyze_search_chunk_size=2,
        root_noise_weight=0.25,
        model=SimpleNamespace(world_model_cfg=SimpleNamespace(env_num=2)),
    )
    buffer.task_id = None
    contexts = ['ctx0', 'ctx1', 'ctx2']

    buffer._search_reanalyzed_roots_in_chunks(
        model=SimpleNamespace(world_model=_WorldModel()),
        latent_state_roots=np.arange(3).reshape(3, 1),
        legal_actions=[[0], [0], [0]],
        noises=[[0.0]] * 3,
        reward_pool=[0.0] * 3,
        policy_logits_pool=[[0.0]] * 3,
        to_play=[-1] * 3,
        root_timesteps=[10, 11, 12],
        root_token_contexts=contexts,
    )

    assert events == [
        ('clear',),
        ('seed', [0, 1], ['ctx0', 'ctx1']),
        ('search', [0, 1]),
        ('clear',),
        ('seed', [2], ['ctx2']),
        ('search', [2]),
    ]
    assert prepared_policy_logits == [[10.0], [11.0], [12.0]]


def test_reanalysis_reuses_descendant_recurrent_cache():
    cloned_cache = object()

    class _CachedKV:
        size = 3

        @staticmethod
        def clone():
            return cloned_cache

    queried_keys = []
    world_model = WorldModel.__new__(WorldModel)
    world_model.rebuild_kv_window_from_tokens = False
    world_model.env_num = 2
    world_model.total_query_count = 0
    world_model.hit_count = 0
    world_model.reanalyze_phase = True
    world_model.use_new_cache_manager = True
    world_model.current_infer_env_ids = None
    world_model.kv_cache_manager = SimpleNamespace(
        get_init_cache=lambda env_id, key: None,
        get_recur_cache=lambda key: (queried_keys.append(key), _CachedKV())[1]
    )
    world_model.keys_values_wm_list = []
    world_model.keys_values_wm_size_list = []

    sizes = world_model.retrieve_or_generate_kvcache(
        latent_state=np.zeros((1, 1, 4), dtype=np.float32),
        ready_env_num=1,
    )

    assert len(queried_keys) == 1
    assert world_model.keys_values_wm_list == [cloned_cache]
    assert sizes == [3]
    assert world_model.hit_count == 1


def test_sampled_reanalysis_preserves_chunked_outputs(monkeypatch):
    searched_timestep_chunks = []

    class _Roots:

        def __init__(self, legal_actions):
            self.ids = [actions[0] for actions in legal_actions]

        def prepare(self, *args):
            pass

        def get_distributions(self):
            return [[value] for value in self.ids]

        def get_values(self):
            return [value + 10 for value in self.ids]

        def get_sampled_actions(self):
            return [value + 20 for value in self.ids]

    class _ChunkedSampledMCTS:

        def __init__(self, cfg):
            pass

        @staticmethod
        def roots(root_num, legal_actions, action_space_size, sampled_action_num, continuous):
            assert root_num == len(legal_actions)
            assert action_space_size == 5
            assert sampled_action_num == 3
            assert continuous is False
            return _Roots(legal_actions)

        def search(self, roots, model, latent_state_roots, to_play, timestep):
            assert len(latent_state_roots) == len(to_play) == len(timestep)
            searched_timestep_chunks.append(list(timestep))

    monkeypatch.setattr(sampled_unizero_buffer_module, 'MCTSCtree', _ChunkedSampledMCTS)
    clear_count = {'value': 0}
    world_model = SimpleNamespace(
        env_num=2,
        clear_caches=lambda: clear_count.__setitem__('value', clear_count['value'] + 1),
    )
    buffer = SampledUniZeroGameBuffer.__new__(SampledUniZeroGameBuffer)
    buffer._cfg = SimpleNamespace(
        reanalyze_search_chunk_size=2,
        root_noise_weight=0.25,
        model=SimpleNamespace(
            num_of_sampled_actions=3,
            continuous_action_space=False,
        ),
    )
    buffer.action_space_size = 5
    buffer.task_id = None

    distributions, values, sampled_actions = buffer._search_sampled_reanalyzed_roots_in_chunks(
        model=SimpleNamespace(world_model=world_model),
        latent_state_roots=np.arange(5).reshape(5, 1),
        legal_actions=[[0], [1], [2], [3], [4]],
        noises=[[0.0]] * 5,
        reward_pool=[0.0] * 5,
        policy_logits_pool=[[0.0]] * 5,
        to_play=[-1] * 5,
        root_timesteps=[10, 11, 12, 13, 14],
    )

    assert searched_timestep_chunks == [[10, 11], [12, 13], [14]]
    assert distributions == [[0], [1], [2], [3], [4]]
    assert values == [10, 11, 12, 13, 14]
    assert sampled_actions == [20, 21, 22, 23, 24]
    assert clear_count['value'] == 3
