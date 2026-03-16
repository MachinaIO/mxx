from __future__ import annotations

import io
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from repo_validation import (
    DEFAULT_GPU_REPEAT_COUNT,
    edited_paths_from_git,
    gpu_repeat_validation_trigger_paths,
    gpu_single_run_validation_trigger_paths,
    maybe_run_gpu_repeat_validation,
    parse_cargo_test_executables,
    run_gpu_repeat_suite,
)


class RepoValidationTests(unittest.TestCase):
    def test_gpu_validation_trigger_paths_split_repeat_and_single_run_modes(self) -> None:
        paths = [
            "cuda/src/kernel.cu",
            "src/matrix/gpu_mul.rs",
            "src/poly/subdir/gpu_fft.rs",
            "src/lookup/ggh15/poly_encoding_gpu.rs",
            "tests/test_gpu_case.rs",
            "src/matrix/mul.rs",
            "README.md",
        ]

        self.assertEqual(
            gpu_repeat_validation_trigger_paths(paths),
            ["cuda/src/kernel.cu", "src/matrix/gpu_mul.rs", "src/poly/subdir/gpu_fft.rs"],
        )
        self.assertEqual(
            gpu_single_run_validation_trigger_paths(paths),
            ["src/lookup/ggh15/poly_encoding_gpu.rs", "tests/test_gpu_case.rs"],
        )

    def test_parse_cargo_test_executables_collects_unique_test_binaries(self) -> None:
        stdout_text = "\n".join(
            [
                '{"reason":"compiler-artifact","target":{"test":true},"executable":"/tmp/bin-a"}',
                '{"reason":"compiler-artifact","target":{"test":false},"executable":"/tmp/not-a-test"}',
                '{"reason":"compiler-artifact","target":{"test":true},"executable":"/tmp/bin-b"}',
                '{"reason":"compiler-artifact","target":{"test":true},"executable":"/tmp/bin-a"}',
                "not-json",
            ]
        )

        self.assertEqual(
            parse_cargo_test_executables(stdout_text),
            [Path("/tmp/bin-a"), Path("/tmp/bin-b")],
        )

    def test_run_gpu_repeat_suite_counts_failed_iterations_and_keeps_running(self) -> None:
        log = io.StringIO()
        calls: list[str] = []
        outcomes = {
            (1, "bin-a"): 0,
            (1, "bin-b"): 0,
            (2, "bin-a"): 1,
            (2, "bin-b"): 0,
            (3, "bin-a"): 0,
            (3, "bin-b"): 2,
        }
        state = {"iteration": 1, "count": 0}

        def executor(binary: Path) -> int:
            key = (state["iteration"], binary.name)
            calls.append(f"{state['iteration']}:{binary.name}")
            state["count"] += 1
            if state["count"] == 2:
                state["iteration"] += 1
                state["count"] = 0
            return outcomes[key]

        summary = run_gpu_repeat_suite(
            binaries=[Path("/tmp/bin-a"), Path("/tmp/bin-b")],
            repeat_count=3,
            executor=executor,
            log=log,
        )

        self.assertEqual(summary.failed_iterations, [2, 3])
        self.assertEqual(
            calls,
            [
                "1:bin-a",
                "1:bin-b",
                "2:bin-a",
                "2:bin-b",
                "3:bin-a",
                "3:bin-b",
            ],
        )
        self.assertIn("iteration 2/3: FAIL", log.getvalue())
        self.assertIn("iteration 3/3: FAIL", log.getvalue())

    def test_edited_paths_from_git_combines_unstaged_staged_and_untracked(self) -> None:
        outputs = iter(
            [
                subprocess.CompletedProcess(args=("git",), returncode=0, stdout="src/lib.rs\ncuda/src/kernel.cu\n", stderr=""),
                subprocess.CompletedProcess(args=("git",), returncode=0, stdout="src/lib.rs\n", stderr=""),
                subprocess.CompletedProcess(args=("git",), returncode=0, stdout="tests/test_gpu_case.rs\n", stderr=""),
            ]
        )

        def runner(*args, **kwargs) -> subprocess.CompletedProcess[str]:
            return next(outputs)

        self.assertEqual(
            edited_paths_from_git(Path("/tmp/repo"), runner=runner),
            ["src/lib.rs", "cuda/src/kernel.cu", "tests/test_gpu_case.rs"],
        )

    def test_edited_paths_from_git_includes_deleted_gpu_related_files(self) -> None:
        outputs = iter(
            [
                subprocess.CompletedProcess(
                    args=("git",),
                    returncode=0,
                    stdout="cuda/src/removed_kernel.cu\n",
                    stderr="",
                ),
                subprocess.CompletedProcess(
                    args=("git",),
                    returncode=0,
                    stdout="tests/test_gpu_removed.rs\n",
                    stderr="",
                ),
                subprocess.CompletedProcess(args=("git",), returncode=0, stdout="", stderr=""),
            ]
        )

        def runner(*args, **kwargs) -> subprocess.CompletedProcess[str]:
            return next(outputs)

        paths = edited_paths_from_git(Path("/tmp/repo"), runner=runner)

        self.assertEqual(
            paths,
            ["cuda/src/removed_kernel.cu", "tests/test_gpu_removed.rs"],
        )
        self.assertEqual(
            gpu_repeat_validation_trigger_paths(paths),
            ["cuda/src/removed_kernel.cu"],
        )
        self.assertEqual(
            gpu_single_run_validation_trigger_paths(paths),
            ["tests/test_gpu_removed.rs"],
        )

    def test_maybe_run_gpu_repeat_validation_uses_repeat_mode_for_strong_triggers(self) -> None:
        log = io.StringIO()
        binary = Path("/tmp/gpu-bin")
        executed: list[Path] = []

        with (
            patch("repo_validation.edited_paths_from_git", return_value=["src/matrix/gpu_mul.rs"]),
            patch("repo_validation.compile_gpu_test_binaries", return_value=[binary]),
            patch(
                "repo_validation.run_gpu_binary",
                side_effect=lambda path, _repo_root, _env: executed.append(path) or 0,
            ),
        ):
            status = maybe_run_gpu_repeat_validation(Path("/tmp/repo"), DEFAULT_GPU_REPEAT_COUNT, log)

        self.assertEqual(status, 0)
        self.assertEqual(executed, [binary] * DEFAULT_GPU_REPEAT_COUNT)
        self.assertIn("repeat mode triggered", log.getvalue())
        self.assertIn(f"running {DEFAULT_GPU_REPEAT_COUNT} sequential iterations", log.getvalue())

    def test_maybe_run_gpu_repeat_validation_uses_single_run_mode_for_other_gpu_rs(self) -> None:
        log = io.StringIO()
        binary = Path("/tmp/gpu-bin")
        executed: list[Path] = []

        with (
            patch(
                "repo_validation.edited_paths_from_git",
                return_value=["src/lookup/ggh15/poly_encoding_gpu.rs"],
            ),
            patch("repo_validation.compile_gpu_test_binaries", return_value=[binary]),
            patch(
                "repo_validation.run_gpu_binary",
                side_effect=lambda path, _repo_root, _env: executed.append(path) or 0,
            ),
        ):
            status = maybe_run_gpu_repeat_validation(Path("/tmp/repo"), DEFAULT_GPU_REPEAT_COUNT, log)

        self.assertEqual(status, 0)
        self.assertEqual(executed, [binary])
        self.assertIn("single-run mode triggered", log.getvalue())
        self.assertIn("running 1 sequential iteration", log.getvalue())


if __name__ == "__main__":
    unittest.main()
