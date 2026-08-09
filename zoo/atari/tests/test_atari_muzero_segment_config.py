import os

import pytest

import lzero.entry
from zoo.atari.config import atari_muzero_segment_config


def test_muzero_segment_run_identity_and_budget(monkeypatch, tmp_path):
    captured = {}

    def fake_train(configs, seed, model_path, max_env_step):
        captured['configs'] = configs
        captured['seed'] = seed
        captured['max_env_step'] = max_env_step
        captured['model_path'] = model_path

    monkeypatch.setattr(lzero.entry, 'train_muzero_segment', fake_train)
    monkeypatch.chdir(tmp_path)

    atari_muzero_segment_config.main(
        'ALE/Pong-v5',
        0,
        output_root=str(tmp_path),
        run_name='mz-pong-seed0',
        max_env_step_override=1_000_000,
        resume_from='/tmp/source.pth.tar',
        resume_buffer_min_transitions_override=12345,
        save_ckpt_after_iter_override=1000,
        periodic_ckpt_keep_last_override=2,
    )

    main_config = captured['configs'][0]
    assert captured['seed'] == 0
    assert captured['max_env_step'] == 1_000_000
    assert captured['model_path'] == '/tmp/source.pth.tar'
    assert os.path.abspath(main_config.exp_name) == str(tmp_path / 'mz-pong-seed0')
    assert main_config.env.env_id == 'ALE/Pong-v5'
    assert main_config.policy.resume_buffer_min_transitions == 12345
    assert main_config.policy.learn.learner.hook.save_ckpt_after_iter == 1000
    assert main_config.policy.periodic_ckpt_keep_last == 2


def test_muzero_segment_rejects_invalid_budget_and_existing_run(tmp_path):
    with pytest.raises(ValueError, match='max_env_step must be positive'):
        atari_muzero_segment_config.main(
            'ALE/Pong-v5', 0, output_root=str(tmp_path), run_name='invalid', max_env_step_override=0
        )

    (tmp_path / 'existing').mkdir()
    with pytest.raises(FileExistsError, match='Run directory already exists'):
        atari_muzero_segment_config.main(
            'ALE/Pong-v5', 0, output_root=str(tmp_path), run_name='existing', max_env_step_override=1
        )


def test_prepare_run_directory_allows_only_explicit_checkpoint_resume(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    atari_muzero_segment_config._prepare_run_directory(
        str(run_dir), resume_from='/tmp/checkpoint.pth.tar', resume_in_place=True
    )
    with pytest.raises(ValueError, match='requires resume_from'):
        atari_muzero_segment_config._prepare_run_directory(
            str(run_dir), resume_from=None, resume_in_place=True
        )
