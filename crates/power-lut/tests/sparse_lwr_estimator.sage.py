#!/usr/bin/env sage -python
"""Full pinned-commit LWE-surrogate estimate for one sparse-binary LWR candidate.

The caller supplies the official lattice-estimator checkout through
MXX_LATTICE_ESTIMATOR_ROOT.  The script emits JSON only on stdout; diagnostics
and estimator failures go to stderr.  Direct-LWR attacks are intentionally
outside this surrogate and remain NotEvaluated in the Rust assessment.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# Sage writes an import cache during startup.  Keep that generated state out of
# the checkout so the subprocess remains usable in read-only worktrees.
os.environ.setdefault("DOT_SAGE", "/tmp/mxx-sage-cache")
from sage.all import Integer, Rational, RealNumber, oo


EXPECTED_ATTACKS = (
    "arora-gb",
    "bkw",
    "usvp",
    "bdd",
    "bdd_hybrid",
    "bdd_mitm_hybrid",
    "dual",
    "dual_hybrid",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu", type=int, required=True)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args()


def estimator_root():
    configured = os.environ.get("MXX_LATTICE_ESTIMATOR_ROOT")
    if configured:
        return Path(configured).resolve()
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents, Path.cwd()):
        if (candidate / "estimator" / "estimator" / "lwe.py").is_file():
            return candidate
    raise RuntimeError("set MXX_LATTICE_ESTIMATOR_ROOT to the official estimator checkout")


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    try:
        if value == oo:
            return None
        if isinstance(value, Integer):
            return int(value)
        if isinstance(value, (Rational, RealNumber)):
            return float(value)
        return float(value)
    except Exception:
        return repr(value)


def main():
    args = parse_args()
    root = estimator_root()
    sys.path.insert(0, str(root))
    from estimator.estimator import LWE, ND, RC
    from sage.version import version as sage_version

    if args.nu <= 0 or args.h <= 0 or args.h > args.nu:
        raise ValueError("require 0 < h <= nu")
    if args.q <= 0 or args.p <= 0 or args.p > args.q or args.q % args.p:
        raise ValueError("require 0 < p <= q and p | q")
    delta = args.q // args.p
    low = -(delta // 2)
    high = (delta - 1) // 2
    params = LWE.Parameters(
        n=args.nu,
        q=args.q,
        Xs=ND.SparseBinary(args.h, args.nu),
        Xe=ND.Uniform(low, high),
        m=oo,
        tag="sparse-binary-lwr-lift",
    )
    results = LWE.estimate(
        params,
        red_cost_model=RC.MATZOV,
        red_shape_model="gsa",
        jobs=args.jobs,
        catch_exceptions=True,
        quiet=True,
    )

    attacks = {}
    failures = []
    infinite_attacks = []
    for name in EXPECTED_ATTACKS:
        result = results.get(name)
        if result is None:
            failures.append(name)
            continue
        fields = json_safe(dict(result))
        rop = result.get("rop", oo)
        if rop == oo or not math.isfinite(float(rop)):
            infinite_attacks.append(name)
            fields["rop_log2"] = None
        else:
            fields["rop_log2"] = math.log2(float(rop))
        attacks[name] = fields
    finite_bits = [fields["rop_log2"] for fields in attacks.values() if fields["rop_log2"] is not None]
    minimum = min(finite_bits) if finite_bits and not failures and not infinite_attacks else float("nan")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    report = {
        "schema_version": 1,
        "repository_url": "https://github.com/malb/lattice-estimator.git",
        "git_commit": commit,
        "sage_version": str(sage_version),
        "python_version": sys.version.split()[0],
        "cost_model": "MATZOV",
        "shape_model": "gsa",
        "quantum": False,
        "sample_count": "infinity",
        "attacks": attacks,
        "failures": failures,
        "infinite_attacks": infinite_attacks,
        "minimum_classical_bits": minimum,
    }
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
