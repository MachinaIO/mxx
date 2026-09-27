#!/usr/bin/env python3
"""Validate explicit GPU FHE evidence files and summarize completed measurements."""
import argparse
import csv
import datetime
import decimal
import json
import math
from pathlib import Path
import statistics

HE_TABLE_SOURCE = "https://github.com/encryptorion-lab/phantom-fhe/blob/1f4a198443b3af77118e51f53d5b8f332154b875/include/host/hestdparms.h"
HE_TABLE_LIMITS = {1024: 27, 2048: 54, 4096: 109, 8192: 218, 16384: 438, 32768: 881, 65536: 1777, 131072: 3576}
MATCHED_OPERATIONS = {"multiply_relinearize", "multiply_relinearize_modswitch"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def samples(values, count, label):
    require(len(values) == count, f"{label}: expected {count} samples, got {len(values)}")
    require(all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in values),
            f"{label}: samples must be finite positive seconds")
    return statistics.median(values), statistics.mean(values)


def security(result, bgv):
    sec = result["security"]
    require(sec["ring_dimension"] == result["n"], "estimator ring dimension mismatch")
    require(sec["error_sigma"] == result["sigma"], "estimator sigma mismatch")
    require(int(sec["error_cutoff"]) == int(result["cutoff"]), "estimator cutoff mismatch")
    expected_cutoff = int(decimal.Decimal(str(result["sigma"])) * decimal.Decimal("6.5"))
    require(int(result["cutoff"]) == expected_cutoff, "cutoff differs from sigma-derived helper")
    require(sec["samples"] == "infinity", "unexpected estimator sample assumption")
    require(sec["mode"] in ("exact", "rough"), "unknown estimator mode")
    require(sec["cost_model"] == ("ADPS16 / Core-SVP" if sec["mode"] == "rough" else "MATZOV (estimator default)"),
            "estimator cost model inconsistent with mode")
    require(isinstance(sec["attack_set"], str) and sec["attack_set"], "missing estimator attack set")
    require(sec["cli_revision"] != "unavailable" and sec["estimator_revision"] != "unavailable",
            "estimator revisions unavailable")
    expected = {"Q": math.prod(result["q"])}
    if bgv:
        expected["QP"] = expected["Q"] * math.prod(result["p"])
        require(sec["secret_distribution"] == "Ternary", "BGV comparison requires ternary secret")
    else:
        require((int(result["secret_min"]), int(result["secret_max"])) in ((0, 1), (-1, 1)),
                "unexpected secret support")
        require(sec["secret_distribution"] == ("Binary" if int(result["secret_min"]) == 0 else "Ternary"),
                "estimator secret mismatch")
    estimates = {entry["basis"]: entry for entry in sec["estimates"]}
    require(len(estimates) == len(sec["estimates"]) and estimates.keys() == expected.keys(),
            "missing or duplicate security basis")
    for basis, modulus in expected.items():
        entry = estimates[basis]
        require(int(entry["modulus"]) == modulus, f"{basis}: estimator modulus mismatch")
        require(entry["security_bits"] >= sec["required_bits"], f"{basis}: security requirement failed")
        require(int(entry["stdout"].strip().splitlines()[-1]) == entry["security_bits"],
                "security result differs from estimator stdout")
    table = sec["he_standard"]
    basis = "QP" if bgv else "Q"
    limit = HE_TABLE_LIMITS.get(result["n"])
    matches = None if limit is None else (sec["secret_distribution"] == "Ternary" and result["sigma"] == 3.2
                                         and expected[basis].bit_length() <= limit)
    require(table["source"] == HE_TABLE_SOURCE and table["basis"] == basis,
            "HE Standard table source or basis mismatch")
    require(table["actual_modulus_bits"] == expected[basis].bit_length()
            and table["table_modulus_bits"] == limit and table["table_parameters_match"] == matches,
            "HE Standard table metadata mismatch")
    return sec, estimates


def validate_report(path, bgv):
    document = load(path)
    result = document["result"]
    require(result["correct"] is True, f"{path}: round trip did not pass")
    require(result["gpu_event_seconds"] is None, "mxx device time must remain unmeasured")
    require("resident output retrieval" in result["timing_contract"]
            and "output serialization and transfers" in result["timing_contract"]
            and "excludes prior-iteration cleanup" in result["timing_contract"],
            "primary comparison requires resident outputs and prior-iteration cleanup outside timing; archived earlier timings are excluded")
    require(result["n"] > 0 and result["q"], "empty ring")
    require(len(document["gpu"].splitlines()) == 1, "comparison requires one visible GPU")
    sec, estimates = security(result, bgv)
    common = {
        "date": datetime.datetime.fromtimestamp(int(document["timestamp_unix_ns"]) / 1e9,
                                                 datetime.timezone.utc).date().isoformat(),
        "library": "mxx", "scheme": "bgv" if bgv else "ring_gsw", "report": str(path),
        "commit": document["commit"], "device": document["gpu"].split(",")[0].strip(),
        "n": result["n"], "q": json.dumps(result["q"], separators=(",", ":")),
        "p": json.dumps(result.get("p", []), separators=(",", ":")),
        "t": result.get("t", ""), "digit_size": result.get("digit_size", ""),
        "sigma": result["sigma"], "cutoff": result["cutoff"],
        "security_q_bits": estimates["Q"]["security_bits"],
        "security_qp_bits": estimates.get("QP", {}).get("security_bits", ""),
        "security_required_bits": sec["required_bits"], "estimator_mode": sec["mode"],
        "estimator_cost_model": sec["cost_model"], "estimator_attack_set": sec["attack_set"],
        "he_standard_classification": sec["he_standard"]["classification"],
        "he_standard_parameters_match": sec["he_standard"]["table_parameters_match"],
        "he_standard_classical_bits": 128 if sec["he_standard"]["table_parameters_match"] else None,
        "he_standard_modulus_bits": sec["he_standard"]["table_modulus_bits"],
        "he_standard_actual_modulus_bits": sec["he_standard"]["actual_modulus_bits"],
        "he_standard_source": sec["he_standard"]["source"],
        "cli_revision": sec["cli_revision"], "estimator_revision": sec["estimator_revision"],
        "timing_contract": result["timing_contract"],
    }
    return result, common


def validate_bgv(result):
    stages = {stage["stage"]: stage for stage in result["stages"]}
    expected = {"lhs": 2, "rhs": 2, "quadratic": 3, "relinearized": 2, "modswitched": 2}
    require(len(stages) == len(result["stages"]) and stages.keys() == expected.keys(),
            "missing or duplicate BGV diagnostics")
    top = len(result["q"]) - 1
    require(0 < result["modswitch_steps"] <= top, "invalid modswitch depth")
    for name, components in expected.items():
        stage = stages[name]
        level = top - result["modswitch_steps"] if name == "modswitched" else top
        require(stage["components"] == components and stage["level"] == level,
                f"{name}: unexpected components or level")
        bound, actual = int(stage["noise_bound"]), int(stage["observed_noise"])
        require(0 <= actual <= bound, f"{name}: measured noise exceeds bound")
        require(2 * (result["t"] // 2 + result["t"] * bound) < math.prod(result["q"][:level + 1]),
                f"{name}: correctness inequality failed")
    require(stages["quadratic"]["correction_factor"] == stages["relinearized"]["correction_factor"],
            "relinearization changed correction factor")
    factor = stages["relinearized"]["correction_factor"]
    for prime in result["q"][-result["modswitch_steps"]:]:
        factor = factor * pow(prime, -1, result["t"]) % result["t"]
    require(factor == stages["modswitched"]["correction_factor"], "modswitch correction factor mismatch")


def bgv_rows(report_path, phantom_path, manifest_path, count):
    result, common = validate_report(report_path, True)
    validate_bgv(result)
    phantom, manifest = load(phantom_path), load(manifest_path)
    require(phantom["correct"] is True, "Phantom round trip failed")
    require(phantom["device"] == common["device"], "GPU models differ")
    require(phantom["secret_distribution"] == "uniform_ternary", "Phantom secret distribution differs")
    require(result["modswitch_steps"] == 1, "Phantom comparison performs one modswitch")
    for key in ("n", "q", "p", "t", "digit_size", "x", "y"):
        require(result[key] == manifest[key] == phantom[key], f"comparison mismatch: {key}")
    require(result["sigma"] == manifest["sigma"], "mxx sigma differs from manifest")
    for key in ("x", "y"):
        require(len(manifest[key]) == result["n"] and all(0 <= v < result["t"] for v in manifest[key]),
                f"invalid {key} slots")
    operations = {entry["operation"]: entry["seconds"] for entry in result["timings"]}
    require(len(operations) == len(result["timings"]), "duplicate mxx operation")
    require(operations.keys() == MATCHED_OPERATIONS | {"multiply", "relinearize", "modswitch"},
            "missing stage timings")
    phantom_operations = {entry["operation"] for entry in phantom["samples"]}
    require(phantom_operations in (MATCHED_OPERATIONS, operations.keys()),
            "Phantom must contain both composites or all five operations")
    rows = []
    for operation, values in operations.items():
        median, mean = samples(values, count, operation)
        row = dict(common, operation=operation, samples=count, host_median_seconds=median,
                   host_mean_seconds=mean, gpu_median_seconds=None, gpu_mean_seconds=None,
                   manifest=str(manifest_path), phantom_error_distribution=phantom["error_distribution"],
                   phantom_actual_sigma=phantom["actual_sigma"], phantom_secret_distribution=phantom["secret_distribution"])
        if operation in phantom_operations:
            matching = [entry for entry in phantom["samples"] if entry["operation"] == operation]
            require(sorted(entry["iteration"] for entry in matching) == list(range(count)),
                    "missing or duplicate Phantom iteration")
            wall_median, wall_mean = samples([v["wall_seconds"] for v in matching], count, operation)
            gpu_median, gpu_mean = samples([v["gpu_seconds"] for v in matching], count, operation)
            row.update(phantom_report=str(phantom_path), phantom_revision=phantom["revision"],
                       phantom_host_median_seconds=wall_median, phantom_host_mean_seconds=wall_mean,
                       phantom_gpu_median_seconds=gpu_median, phantom_gpu_mean_seconds=gpu_mean,
                       host_median_slowdown=median / wall_median, host_mean_slowdown=mean / wall_mean)
        rows.append(row)
    return rows


def ring_rows(path, count):
    result, common = validate_report(path, False)
    cases = result["cases"]
    require(len(cases) >= 3 and {0, 1} <= {case["bit"] for case in cases}, "both bits and random case required")
    rows = []
    for index, case in enumerate(cases):
        require(case["bit"] in (0, 1) and 0 <= case["monomial_exponent"] < 2 * result["n"],
                "invalid bit or monomial")
        stages = {stage["stage"]: stage for stage in case["stages"]}
        require(len(stages) == len(case["stages"]) and stages.keys() == {"fresh", "external_product"},
                "missing Ring-GSW diagnostics")
        for stage in stages.values():
            bound, actual, plain = int(stage["noise_bound"]), int(stage["observed_noise"]), int(stage["plaintext_bound"])
            scale, q = int(result["scale"]), math.prod(result["q"])
            require(0 <= actual <= bound and 2 * bound < scale and 2 * (scale * plain + bound) < q,
                    "Ring-GSW correctness or measured-noise bound failed")
        median, mean = samples(case["external_product_seconds"], count, "external_product")
        rows.append(dict(common, operation="external_product", case=index, bit=case["bit"],
                         monomial_exponent=case["monomial_exponent"], scale=result["scale"],
                         base_bits=result["base_bits"], digits=result["digits"], samples=count,
                         host_median_seconds=median, host_mean_seconds=mean,
                         gpu_median_seconds=None, gpu_mean_seconds=None))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bgv", nargs=3, action="append", default=[], metavar=("MXX", "PHANTOM", "MANIFEST"))
    parser.add_argument("--ring-gsw", action="append", default=[], metavar="MXX")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(args.samples > 0 and (args.bgv or args.ring_gsw), "positive count and explicit input files required")
    paths = [Path(paths[0]).resolve() for paths in args.bgv] + [Path(path).resolve() for path in args.ring_gsw]
    require(len(paths) == len(set(paths)), "duplicate mxx report")
    rows = []
    for inputs in args.bgv:
        rows.extend(bgv_rows(*inputs, args.samples))
    for path in args.ring_gsw:
        rows.extend(ring_rows(path, args.samples))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validated {len(paths)} mxx reports; wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Invalid measurement evidence: {error}") from error
