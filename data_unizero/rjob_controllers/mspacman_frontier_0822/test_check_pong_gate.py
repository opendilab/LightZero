import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from check_pong_gate import require_latest_score
from find_valid_checkpoint import find_valid_checkpoint


class TestPongGate(unittest.TestCase):

    def test_latest_evaluation_passes(self):
        require_latest_score([-21.0, 19.375, 19.25], 15.0)

    def test_earlier_peak_does_not_hide_final_regression(self):
        with self.assertRaisesRegex(ValueError, r'latest=12\.750 < 15\.000'):
            require_latest_score([-21.0, 19.375, 12.75], 15.0)

    def test_needs_two_evaluations(self):
        with self.assertRaisesRegex(ValueError, 'at least two evaluations'):
            require_latest_score([19.25], 15.0)


class TestCheckpointRecovery(unittest.TestCase):

    @staticmethod
    def _write_archive(path: Path):
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('checkpoint/data.pkl', b'complete')

    def test_skips_newest_truncated_checkpoint(self):
        with TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / 'ckpt'
            checkpoint_dir.mkdir()
            older = checkpoint_dir / 'iteration_20000.pth.tar'
            newest = checkpoint_dir / 'iteration_40000.pth.tar'
            self._write_archive(older)
            newest.write_bytes(b'truncated')

            self.assertEqual(find_valid_checkpoint(Path(directory)), older)

    def test_falls_back_to_complete_best_checkpoint(self):
        with TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / 'ckpt'
            checkpoint_dir.mkdir()
            (checkpoint_dir / 'iteration_20000.pth.tar').write_bytes(b'truncated')
            best = checkpoint_dir / 'ckpt_best.pth.tar'
            self._write_archive(best)

            self.assertEqual(find_valid_checkpoint(Path(directory)), best)


if __name__ == '__main__':
    unittest.main()
