#!/usr/bin/env python3
"""Validate and pool explicit matched FHE runs, retaining raw timing evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import summarize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bgv", nargs=3, action="append", required=True,
                        metavar=("MXX", "PHANTOM", "MANIFEST"))
    parser.add_argument("--ring-gsw", action="append", default=[])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports, groups = [], {}
    used_mxx, used_phantom = set(), set()
    for mxx_path, phantom_path, manifest_path in args.bgv:
        summarize.require(Path(mxx_path).resolve() not in used_mxx, "duplicate mxx report")
        summarize.require(Path(phantom_path).resolve() not in used_phantom, "duplicate Phantom report")
        used_mxx.add(Path(mxx_path).resolve())
        used_phantom.add(Path(phantom_path).resolve())
        rows = summarize.bgv_rows(mxx_path, phantom_path, manifest_path, args.samples)
        mxx, phantom = summarize.load(mxx_path), summarize.load(phantom_path)
        result = mxx["result"]
        parameters = {key: result[key] for key in ("n", "q", "p", "t", "digit_size", "sigma", "cutoff")}
        key = json.dumps(parameters, sort_keys=True)
        group = groups.setdefault(key, {"parameters": parameters, "gpu": mxx["gpu"], "runs": 0,
                                        "operations": {}})
        summarize.require(group["gpu"] == mxx["gpu"], "physical mxx GPU or driver differs")
        group["runs"] += 1
        report = {"mxx_report": mxx_path, "mxx_sha256": digest(mxx_path),
                  "phantom_report": phantom_path, "phantom_sha256": digest(phantom_path),
                  "manifest": manifest_path, "manifest_sha256": digest(manifest_path),
                  "timestamp_unix_ns": mxx["timestamp_unix_ns"],
                  "mxx_base_commit": mxx["commit"], "phantom_revision": phantom["revision"],
                  "gpu": mxx["gpu"], "parameters": parameters,
                  "security": result["security"], "stages": result["stages"],
                  "timing_contract": result["timing_contract"],
                  "phantom_error_distribution": phantom["error_distribution"],
                  "phantom_actual_sigma": phantom["actual_sigma"], "timings": []}
        for row in rows:
            op = row["operation"]
            wall = next(t["seconds"] for t in result["timings"] if t["operation"] == op)
            reference = [s["wall_seconds"] for s in phantom["samples"] if s["operation"] == op]
            summarize.samples(reference, args.samples, op)
            values = group["operations"].setdefault(op, {"mxx_seconds": [], "phantom_seconds": []})
            values["mxx_seconds"].extend(wall)
            values["phantom_seconds"].extend(reference)
            report["timings"].append({"operation": op, "mxx_seconds": wall, "phantom_seconds": reference})
        reports.append(report)
    for group in groups.values():
        for values in group["operations"].values():
            for library in ("mxx", "phantom"):
                raw = values.pop(f"{library}_seconds")
                values[f"{library}_samples"] = len(raw)
                values[f"{library}_median_seconds"] = statistics.median(raw)
                values[f"{library}_mean_seconds"] = statistics.mean(raw)
            values["median_ratio"] = values["mxx_median_seconds"] / values["phantom_median_seconds"]
        multiplication = group["operations"].get("multiply")
        group["multiply_below_two"] = (
            multiplication["median_ratio"] < 2 if multiplication is not None else None
        )
        group["multiply_relinearize_below_two"] = group["operations"]["multiply_relinearize"]["median_ratio"] < 2
    ring = []
    for path in args.ring_gsw:
        summarize.require(Path(path).resolve() not in used_mxx, "duplicate mxx report")
        used_mxx.add(Path(path).resolve())
        rows = summarize.ring_rows(path, args.samples)
        document = summarize.load(path)
        ring.append({"report": path, "sha256": digest(path), "gpu": document["gpu"],
                     "result": document["result"], "validated_cases": len(rows)})
    output = {"aggregation": "Pool all raw samples, then take the median; no sample exclusions.",
              "groups": list(groups.values()), "bgv_runs": reports, "ring_gsw_runs": ring}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Validated {len(reports)} BGV and {len(ring)} Ring-GSW runs; wrote {args.output}")
    for group in groups.values():
        print(group["parameters"]["q"], group["operations"]["multiply_relinearize"])


if __name__ == "__main__":
    main()
