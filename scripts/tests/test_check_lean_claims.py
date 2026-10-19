"""Regression checks for the fail-closed, byte-for-byte directory comparison."""

from pathlib import Path
import importlib.util
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("check_lean_claims", Path(__file__).resolve().parents[1] / "check_lean_claims.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ClaimFreshnessTests(unittest.TestCase):
    def test_detects_changed_missing_and_obsolete_modules(self):
        with tempfile.TemporaryDirectory() as temporary:
            committed, regenerated = (Path(temporary) / name for name in ("committed", "regenerated"))
            committed.mkdir()
            regenerated.mkdir()
            for directory in (committed, regenerated):
                (directory / "Claim.lean").write_bytes(b"claim\n")
            self.assertEqual(checker.differences(committed, regenerated), [])
            (regenerated / "Claim.lean").write_bytes(b"claim\r\n")
            self.assertEqual(checker.differences(committed, regenerated), ["bytes differ: Claim.lean"])
            (regenerated / "Claim.lean").unlink()
            (regenerated / "NewStage.lean").write_bytes(b"stage\n")
            self.assertEqual(checker.differences(committed, regenerated), ["not regenerated: Claim.lean", "not committed: NewStage.lean"])

    def test_missing_empty_or_symlink_output_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            with self.assertRaises(ValueError):
                checker.files(output)
            output.mkdir()
            with self.assertRaises(ValueError):
                checker.files(output)
            (output / "Claim.lean").symlink_to("missing.lean")
            with self.assertRaises(ValueError):
                checker.files(output)


if __name__ == "__main__":
    unittest.main()
