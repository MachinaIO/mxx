"""Filesystem-only regression tests for selecting a WE editor candidate."""

from pathlib import Path
import tempfile
import unittest

from select_we_lean_candidate import select_candidate


class SelectCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "lean"
        self.project.mkdir()
        self.project.joinpath("lakefile.toml").write_text(
            '[[lean_lib]]\nname = "DiamondCandidate"\nsrcDir = "generated"\n'
            'roots = ["Claim", "NumericCertificate"]\n'
        )
        self.project.joinpath("Certificate.lean").write_text("handwritten source")

    def artifact(self, name):
        artifact = self.root / name
        artifact.mkdir()
        for module in ["Claim", "NumericCertificate", "Certificate"]:
            artifact.joinpath(f"{module}.lean").write_text(name)
        artifact.joinpath("Claim.olean").write_text("must not be copied")
        return artifact

    def test_switch_copies_only_generated_sources_and_retains_previous_snapshot(self):
        first = self.artifact("first")
        selected = select_candidate(first, self.project)
        previous = selected.resolve()
        self.assertEqual((selected / "Claim.lean").read_text(), "first")
        self.assertFalse((selected / "Certificate.lean").exists())
        self.assertFalse((selected / "Claim.olean").exists())
        select_candidate(self.artifact("second"), self.project)
        self.assertEqual((selected / "Claim.lean").read_text(), "second")
        self.assertEqual((previous / "Claim.lean").read_text(), "first")
        self.assertEqual((self.project / "Certificate.lean").read_text(), "handwritten source")

    def test_incomplete_candidate_preserves_selection(self):
        selected = select_candidate(self.artifact("first"), self.project)
        previous = selected.resolve()
        incomplete = self.artifact("incomplete")
        (incomplete / "NumericCertificate.lean").unlink()
        with self.assertRaises(ValueError):
            select_candidate(incomplete, self.project)
        self.assertEqual(selected.resolve(), previous)

    def test_existing_real_directory_is_not_replaced(self):
        (self.project / "generated").mkdir()
        with self.assertRaises(ValueError):
            select_candidate(self.artifact("first"), self.project)
        self.assertFalse((self.project / "generated").is_symlink())


if __name__ == "__main__":
    unittest.main()
