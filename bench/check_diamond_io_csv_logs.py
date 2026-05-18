#!/usr/bin/env python3
"""Check DiamondIO CSV rows against their simulation and benchmark logs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path


BENCHMARK_FIELDS = [
    "obfuscate_latency",
    "obfuscate_total_time_nanos",
    "obfuscate_max_parallelism",
    "eval_latency",
    "eval_total_time_nanos",
    "eval_max_parallelism",
    "obfuscate_input_injection_latency_percent",
    "obfuscate_input_injection_total_time_percent",
    "eval_input_injection_latency_percent",
    "eval_input_injection_total_time_percent",
    "obfuscated_circuit_bytes",
    "input_injection_bytes",
]

CONFIG_CHECKS = [
    ("ring_dim_config", "ring_dim"),
    ("min_log_ring_dim", "min_log_ring_dim"),
    ("max_log_ring_dim", "max_log_ring_dim"),
    ("input_size", "input_size"),
    ("injector_batch_bits", "injector_batch_bits"),
    ("output_size", "output_size"),
    ("crt_bits", "crt_bits"),
    ("base_bits", "base_bits"),
    ("p_moduli_bits", "p_moduli_bits"),
    ("max_unreduced_muls", "max_unreduced_muls"),
    ("scale", "scale"),
    ("min_crt_depth", "min_crt_depth"),
    ("max_crt_depth", "max_crt_depth"),
    ("security_bits", "security_bits"),
    ("noise_refresh_cbd_n", "noise_refresh_cbd_n"),
    ("error_sigma", "error_sigma"),
    ("trapdoor_sigma", "trapdoor_sigma"),
    ("d_secret", "d_secret"),
]

SELECTED_CHECKS = [
    ("selected_crt_depth", "crt_depth"),
    ("selected_log_ring_dim", "log_ring_dim"),
    ("selected_ring_dim", "ring_dim"),
    ("achieved_secpar_for_gauss_in_run", "achieved_secpar_for_gauss"),
    ("achieved_secpar_for_cbd_in_run", "achieved_secpar_for_cbd"),
    ("prf_mask_output_coeff_bits", "prf_mask_output_coeff_bits"),
    ("noise_refresh_v_bits", "noise_refresh_v_bits"),
    ("final_seed_bits", "final_seed_bits"),
    ("noisy_plaintext_error_bits", "noisy_plaintext_error_bits"),
    ("input_injection_error_bits", "input_injection_error_bits"),
]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CONFIG_RE = re.compile(r"DiamondIOGpuBenchConfig \{([^}]*)\}")
KV_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check DiamondIO simulation and benchmark-estimation CSV values against logs."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="bench/security_bits_100_diamond_io_simulation_parameters.csv",
        help="CSV file to check",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_log_path(raw_path: str, root: Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    workspace_prefix = "/workspace/logs/"
    if raw_path.startswith(workspace_prefix):
        local_path = root / "logs" / raw_path[len(workspace_prefix) :]
        if local_path.exists():
            return local_path

    matches = sorted((root / "logs").rglob(path.name)) if (root / "logs").exists() else []
    if len(matches) == 1:
        return matches[0]

    return path


def read_log(raw_path: str, root: Path) -> tuple[Path, str]:
    path = resolve_log_path(raw_path, root)
    if not path.exists():
        raise ValueError(f"log file not found: {raw_path}")
    return path, strip_ansi(path.read_text(errors="replace"))


def last_line_with(text: str, needle: str) -> str:
    matches = [line for line in text.splitlines() if needle in line]
    if not matches:
        raise ValueError(f"missing log line containing {needle!r}")
    return matches[-1]


def parse_config(line: str) -> dict[str, str]:
    match = CONFIG_RE.search(line)
    if not match:
        raise ValueError("missing DiamondIOGpuBenchConfig payload")
    result: dict[str, str] = {}
    for item in match.group(1).split(","):
        key, value = item.strip().split(":", 1)
        result[key.strip()] = value.strip()
    return result


def parse_kv_line(line: str) -> dict[str, str]:
    return {key: value.strip('"') for key, value in KV_RE.findall(line)}


def is_decimal(value: str) -> bool:
    try:
        Decimal(value)
        return True
    except (InvalidOperation, ValueError):
        return False


def values_match(csv_value: str, log_value: str) -> bool:
    left = csv_value.strip().strip('"')
    right = log_value.strip().strip('"')
    if left == "" or right == "":
        return left == right
    if left.lower() in {"true", "false"} or right.lower() in {"true", "false"}:
        return left.lower() == right.lower()
    if is_decimal(left) and is_decimal(right):
        return Decimal(left) == Decimal(right)
    return left == right


def require_match(
    errors: list[str],
    data_no: str,
    source: str,
    column: str,
    csv_value: str,
    log_key: str,
    log_values: dict[str, str],
) -> None:
    if log_key not in log_values:
        if csv_value == "":
            return
        errors.append(f"row {data_no} {source}: log key {log_key!r} missing, CSV {column}={csv_value!r}")
        return

    log_value = log_values[log_key]
    if not values_match(csv_value, log_value):
        errors.append(
            f"row {data_no} {source}: {column}={csv_value!r} does not match "
            f"log {log_key}={log_value!r}"
        )


def check_simulation(row: dict[str, str], root: Path, errors: list[str]) -> int:
    data_no = row["data_no"]
    _, text = read_log(row["simulation_log_file"], root)
    checks = 0

    config = parse_config(last_line_with(text, "DiamondIO GPU bench config:"))
    for csv_column, log_key in CONFIG_CHECKS:
        require_match(errors, data_no, "simulation config", csv_column, row[csv_column], log_key, config)
        checks += 1
    require_match(errors, data_no, "simulation config", "bench_iterations", row["bench_iterations"], "bench_iterations", config)
    require_match(errors, data_no, "simulation config", "search_only", row["search_only"], "search_only", config)
    checks += 2

    selected = parse_kv_line(last_line_with(text, "DiamondIO CRT-depth search selected parameters"))
    for csv_column, log_key in SELECTED_CHECKS:
        require_match(errors, data_no, "simulation selected", csv_column, row[csv_column], log_key, selected)
        checks += 1

    return checks


def check_benchmark(row: dict[str, str], root: Path, errors: list[str]) -> int:
    data_no = row["data_no"]
    _, text = read_log(row["benchmark_estimation_log_file"], root)
    checks = 0

    config = parse_config(last_line_with(text, "DiamondIO GPU bench config:"))
    for csv_column, log_key in CONFIG_CHECKS:
        require_match(
            errors,
            data_no,
            "benchmark config",
            csv_column,
            row[csv_column],
            log_key,
            config,
        )
        checks += 1
    require_match(
        errors,
        data_no,
        "benchmark config",
        "benchmark_estimation_bench_iterations",
        row["benchmark_estimation_bench_iterations"],
        "bench_iterations",
        config,
    )
    require_match(errors, data_no, "benchmark config", "expected_search_only", "false", "search_only", config)
    checks += 2

    selected = parse_kv_line(
        last_line_with(text, "DiamondIO selected simulation parameters provided; skipping error simulation")
    )
    for csv_column, log_key in SELECTED_CHECKS:
        if csv_column.startswith("achieved_secpar_"):
            continue
        require_match(errors, data_no, "benchmark selected", csv_column, row[csv_column], log_key, selected)
        checks += 1

    estimate = parse_kv_line(last_line_with(text, "DiamondIO GPU benchmark estimate"))
    for field in BENCHMARK_FIELDS:
        csv_column = f"benchmark_estimation_{field}"
        require_match(errors, data_no, "benchmark estimate", csv_column, row[csv_column], field, estimate)
        checks += 1

    return checks


def main() -> int:
    args = parse_args()
    root = repo_root()
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = root / csv_path

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    errors: list[str] = []
    check_count = 0
    for row in rows:
        try:
            check_count += check_simulation(row, root, errors)
            check_count += check_benchmark(row, root, errors)
        except ValueError as exc:
            errors.append(f"row {row.get('data_no', '<unknown>')}: {exc}")

    if errors:
        print("CSV/log consistency check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"CSV/log consistency check ok: {len(rows)} rows, {check_count} field checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
