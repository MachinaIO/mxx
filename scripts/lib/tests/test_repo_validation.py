from __future__ import annotations

import io
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from repo_validation import (
    DEFAULT_GPU_REPEAT_COUNT,
    compile_gpu_test_binaries,
    edited_paths_from_git,
    gpu_repeat_validation_trigger_paths,
    gpu_validation_environment,
    run_gpu_binary,
    warning_free_rustflags,
    gpu_single_run_validation_trigger_paths,
    maybe_run_gpu_repeat_validation,
    parse_cargo_test_executables,
    run_gpu_repeat_suite,
)


class RepoValidationTests(unittest.TestCase):
    def test_gpu_validation_trigger_paths_split_repeat_and_single_run_modes(self) -> None:
        paths = [
            "crates/backends/cuda/src/kernel.cu",
            "crates/backends/src/matrix/gpu_mul.rs",
            "crates/backends/src/poly/subdir/gpu_fft.rs",
            "crates/gadgets/src/lookup/ggh15/poly_encoding_gpu.rs",
            "crates/gadgets/tests/test_gpu_case.rs",
            "crates/backends/src/matrix/mul.rs",
            "README.md",
        ]

        self.assertEqual(
            gpu_repeat_validation_trigger_paths(paths),
            ["crates/backends/cuda/src/kernel.cu", "crates/backends/src/matrix/gpu_mul.rs", "crates/backends/src/poly/subdir/gpu_fft.rs"],
        )
        self.assertEqual(
            gpu_single_run_validation_trigger_paths(paths),
            ["crates/gadgets/src/lookup/ggh15/poly_encoding_gpu.rs"],
        )

    def test_moved_native_runtime_and_build_paths_trigger_repetition(self) -> None:
        paths = ["crates/backends/gpu/include/GpuPlatform.h",
                 "crates/backends/src/gpu_runtime.rs",
                 "crates/backends/src/backend/poly_gpu/gpu_physical_control.rs",
                 "crates/backends/build.rs", "crates/fhe/build.rs",
                 "crates/fhe/cuda/src/Tfhe.cu"]
        self.assertEqual(gpu_repeat_validation_trigger_paths(paths), paths)

    def test_backend_environment_defaults_and_preserves_explicit_target(self) -> None:
        self.assertEqual(gpu_validation_environment({})["MXX_GPU_BACKEND"], "cuda")
        env = gpu_validation_environment({"MXX_GPU_BACKEND": "hip", "HIP_ARCH": "gfx942"})
        self.assertEqual(env["CARGO_TARGET_DIR"], "target/gpu-hip")
        self.assertEqual(env["HIP_ARCH"], "gfx942")
        self.assertEqual(gpu_validation_environment({"CARGO_TARGET_DIR": "/tmp/t"})["CARGO_TARGET_DIR"], "/tmp/t")
        with self.assertRaises(ValueError):
            gpu_validation_environment({"MXX_GPU_BACKEND": "other"})

    def test_warning_flags_preserve_repository_openfhe_rpath(self) -> None:
        with patch("repo_validation.sys.platform", "linux"):
            flags = warning_free_rustflags({}).split("\x1f")
        self.assertEqual(flags, ["-C", "link-arg=-Wl,-rpath,/usr/local/lib", "-D", "warnings"])

    def test_warning_flags_respect_explicit_cargo_flag_precedence(self) -> None:
        flags = warning_free_rustflags({"RUSTFLAGS": "-C 'link-arg=a b'"}).split("\x1f")
        self.assertEqual(flags, ["-C", "link-arg=a b", "-D", "warnings"])
        flags = warning_free_rustflags({"RUSTFLAGS": "-A unused", "CARGO_ENCODED_RUSTFLAGS": "-C\x1fopt-level=2"}).split("\x1f")
        self.assertEqual(flags, ["-C", "opt-level=2", "-D", "warnings"])

    def test_gpu_binary_selects_ignored_device_unit_tests(self) -> None:
        with patch("repo_validation.subprocess.run") as runner:
            runner.return_value.returncode = 0
            self.assertEqual(run_gpu_binary(Path("/tmp/bin"), Path("/tmp/repo"), {"GPU_TEST_FILTER": "test_gpu_case"}), 0)
        self.assertEqual(runner.call_args.args[0], ("/tmp/bin", "test_gpu_case", "--ignored"))

    def test_default_gpu_selection_skips_long_unit_tests(self) -> None:
        with patch("repo_validation.subprocess.run") as runner:
            runner.return_value.returncode = 0
            self.assertEqual(run_gpu_binary(Path("/tmp/bin"), Path("/tmp/repo"), {}), 0)
        self.assertEqual(
            runner.call_args.args[0],
            ("/tmp/bin", "gpu", "--ignored", "--skip", "test_gpu_ring_gsw_arithmetic_executes_through_dsl_ir_runtime_and_decrypts"),
        )

    def test_force_runs_on_clean_checkout(self) -> None:
        with (patch("repo_validation.edited_paths_from_git", return_value=[]),
              patch("repo_validation.compile_gpu_test_binaries", return_value=[Path("/tmp/bin")]),
              patch("repo_validation.run_gpu_binary", return_value=0) as run):
            self.assertEqual(maybe_run_gpu_repeat_validation(Path("/tmp/repo"), 3, io.StringIO(), force=True), 0)
            self.assertEqual(run.call_count, 3)

    def test_gpu_compile_refuses_empty_device_gate(self) -> None:
        outputs = iter([
            subprocess.CompletedProcess([], 0, '{"reason":"compiler-artifact","target":{"test":true},"executable":"/tmp/bin"}', ""),
            subprocess.CompletedProcess([], 0, "0 tests, 0 benchmarks", ""),
        ])
        with self.assertRaisesRegex(RuntimeError, "No ignored GPU unit tests"):
            compile_gpu_test_binaries(Path("/tmp/repo"), {}, runner=lambda *a, **kw: next(outputs))

    def test_gpu_compile_selects_matching_device_binaries(self) -> None:
        outputs = iter([
            subprocess.CompletedProcess([], 0, '{"reason":"compiler-artifact","target":{"test":true},"executable":"/tmp/bin"}', ""),
            subprocess.CompletedProcess([], 0, "test_gpu_case: test\n1 test, 0 benchmarks", ""),
        ])
        self.assertEqual(compile_gpu_test_binaries(Path("/tmp/repo"), {}, runner=lambda *a, **kw: next(outputs)), [Path("/tmp/bin")])

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
                subprocess.CompletedProcess(args=("git",), returncode=0, stdout="crates/gadgets/src/lib.rs\ncrates/backends/cuda/src/kernel.cu\n", stderr=""),
                subprocess.CompletedProcess(args=("git",), returncode=0, stdout="crates/gadgets/src/lib.rs\n", stderr=""),
                subprocess.CompletedProcess(args=("git",), returncode=0, stdout="crates/gadgets/tests/test_gpu_case.rs\n", stderr=""),
            ]
        )

        def runner(*args, **kwargs) -> subprocess.CompletedProcess[str]:
            return next(outputs)

        self.assertEqual(
            edited_paths_from_git(Path("/tmp/repo"), runner=runner),
            ["crates/gadgets/src/lib.rs", "crates/backends/cuda/src/kernel.cu", "crates/gadgets/tests/test_gpu_case.rs"],
        )

    def test_edited_paths_from_git_includes_deleted_gpu_related_files(self) -> None:
        outputs = iter(
            [
                subprocess.CompletedProcess(
                    args=("git",),
                    returncode=0,
                    stdout="crates/backends/cuda/src/removed_kernel.cu\n",
                    stderr="",
                ),
                subprocess.CompletedProcess(
                    args=("git",),
                    returncode=0,
                    stdout="crates/gadgets/tests/test_gpu_removed.rs\n",
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
            ["crates/backends/cuda/src/removed_kernel.cu", "crates/gadgets/tests/test_gpu_removed.rs"],
        )
        self.assertEqual(
            gpu_repeat_validation_trigger_paths(paths),
            ["crates/backends/cuda/src/removed_kernel.cu"],
        )
        self.assertEqual(
            gpu_single_run_validation_trigger_paths(paths),
            [],
        )

    def test_maybe_run_gpu_repeat_validation_uses_repeat_mode_for_strong_triggers(self) -> None:
        log = io.StringIO()
        binary = Path("/tmp/gpu-bin")
        executed: list[Path] = []

        with (
            patch("repo_validation.edited_paths_from_git", return_value=["crates/backends/src/matrix/gpu_mul.rs"]),
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
                return_value=["crates/gadgets/src/lookup/ggh15/poly_encoding_gpu.rs"],
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
