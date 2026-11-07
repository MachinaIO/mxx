#!/usr/bin/env python3
"""Write the Lean proof package of one parameter row of `parameters.csv`.

The proof does not depend on the parameters except through the constants of the generated claim:
the modulus `Q`, the ring dimension `N`, the encoding width `M`, the gadget base and digits, and
the fresh error cutoff. This script reads them from `<id>/generated/` (written by
`tests/test_gpu_ring_gsw_boolean_bgg.rs` with `MXX_KHE_RING_GSW_BGG_PARAMETER_ID=<id>`) and fills
the templates, so the proof of a new row is `python3 generator/gen.py <id>` followed by
`lake build` in `<id>/`.
"""

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def constants(generated: pathlib.Path) -> dict:
    backend = (generated / "Backend.lean").read_text()
    evaluate = (generated / "Stage_evaluate.lean").read_text()
    encrypt = (generated / "Stage_encrypt.lean").read_text()
    q = re.search(r"MxxRuntime\.RegularLayout (\d+)", backend).group(1)
    n = re.search(r"n = (\d+) then", backend).group(1)
    base, digits = re.search(r"gadgetDecomposeRuns backend (\d+) (\d+) ", evaluate).groups()
    # The modulus appears literally or through a local notation `"mxxNat<k>" => Q`; the encoding
    # width `M > 1` is the column count of both the `2 x M` and the `1 x M` matrices.
    modulus = "|".join([q, *re.findall(rf'local notation "(mxxNat\d+)" => {q}\b', evaluate)])
    shape = rf"Mxx\.Primitives\.ExactMatrix (?:{modulus}) {n} (\d+) (\d+)"
    shapes = set(re.findall(shape, evaluate))
    (m,) = {columns for rows, columns in shapes if rows == "2" and columns != "1" and ("1", columns) in shapes}
    cutoff = re.search(
        r"gaussianSampleAt tape \(path \+\+ \[\d+\]\) \(\([0-9/]+\)\) \((\d+)\)", encrypt
    ).group(1)
    return {"Q": q, "N": n, "M": m, "BASE": base, "DIGITS": digits, "CUTOFF": cutoff}


def package_name(parameter_id: str) -> str:
    words = re.split(r"[^A-Za-z0-9]+", parameter_id)
    return "MxxKheRingGswBgg" + "".join(word[:1].upper() + word[1:] for word in words if word)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: gen.py <parameter id>")
    parameter_id = sys.argv[1]
    directory = ROOT / parameter_id
    values = constants(directory / "generated")
    values["PACKAGE"] = package_name(parameter_id)
    for template in sorted((HERE / "templates").iterdir()):
        text = template.read_text()
        for key, value in values.items():
            text = text.replace("{{" + key + "}}", value)
        if "{{" in text:
            sys.exit(f"{template.name} has an unfilled placeholder")
        (directory / template.name).write_text(text)
        print(f"wrote {directory / template.name}")


if __name__ == "__main__":
    main()
