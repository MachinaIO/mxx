#!/usr/bin/env python3
"""Regenerate committed Lean claims from Rust, without GPU features or execution.

Outputs go only to a temporary directory. Compare the complete file sets and raw
bytes, so changed, missing, and obsolete generated modules all fail the check.
"""

from pathlib import Path
import argparse
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = {
    "tfhe": Path("crates/fhe/lean/tfhe/generated"),
    "bgv": Path("crates/fhe/lean/bgv/generated"),
    "rlwe": Path("crates/dsl/examples/rlwe/generated"),
}


def files(directory: Path) -> dict[Path, bytes]:
    if not directory.is_dir():
        raise ValueError(f"missing generated directory: {directory}")
    result = {}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"unexpected generated symlink: {path}")
        if path.is_file():
            result[path.relative_to(directory)] = path.read_bytes()
    if not result:
        raise ValueError(f"empty generated directory: {directory}")
    return result


def differences(committed: Path, regenerated: Path) -> list[str]:
    expected, actual = files(committed), files(regenerated)
    errors = []
    for path in sorted(expected.keys() | actual.keys()):
        if path not in actual:
            errors.append(f"not regenerated: {path}")
        elif path not in expected:
            errors.append(f"not committed: {path}")
        elif expected[path] != actual[path]:
            errors.append(f"bytes differ: {path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cargo", default="cargo", help="Cargo executable (e.g. a rustup proxy)")
    args = parser.parse_args()
    # Compare the committed default fixtures. Do not silently inherit local GPU
    # test overrides that would select a different protocol or parameter profile.
    overrides = sorted(name for name in os.environ if name.startswith("FHE_TEST_"))
    if overrides:
        parser.error("unset FHE fixture overrides before checking freshness: " + ", ".join(overrides))
    with tempfile.TemporaryDirectory(prefix="mxx-lean-claims-") as temporary:
        output = Path(temporary)
        for command in [
            [args.cargo, "run", "--locked", "-p", "mxx-fhe", "--example", "export_claims", "--", str(output)],
            [args.cargo, "run", "--locked", "-p", "mxx-dsl", "--example", "rlwe_encrypt", "--", "--export-lean", str(output / "rlwe")],
        ]:
            subprocess.run(command, cwd=ROOT, check=True)
        errors = []
        for name, committed in CLAIMS.items():
            errors.extend(f"{committed}: {error}" for error in differences(ROOT / committed, output / name))
        if errors:
            raise SystemExit("Stale Lean claims:\n" + "\n".join(errors) +
                             "\nRegenerate with the documented host-only export commands and review the proofs.")
        print("TFHE, BGV and RLWE generated claims match byte-for-byte.", flush=True)


if __name__ == "__main__":
    main()
