import pytest

from zoo.atari.config import atari_unizero_segment_config


def test_prepare_run_directory_allows_only_explicit_checkpoint_resume(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    atari_unizero_segment_config._prepare_run_directory(
        str(run_dir), resume_from='/tmp/checkpoint.pth.tar', resume_in_place=True
    )
    with pytest.raises(ValueError, match='requires resume_from'):
        atari_unizero_segment_config._prepare_run_directory(
            str(run_dir), resume_from=None, resume_in_place=True
        )


def test_prepare_run_directory_preserves_default_collision_protection(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    with pytest.raises(FileExistsError, match='Run directory already exists'):
        atari_unizero_segment_config._prepare_run_directory(str(run_dir))


def test_disable_encoder_clip_turns_off_both_projection_owners():
    assert atari_unizero_segment_config._encoder_clip_settings(False) == (True, 10.0)
    assert atari_unizero_segment_config._encoder_clip_settings(True) == (False, 0.0)
