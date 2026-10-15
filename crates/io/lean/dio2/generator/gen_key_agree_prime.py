#!/usr/bin/env python3
"""Generate the step-zero key agreements for CRT primes one through four.

Usage: gen_key_agree_prime.py [package directory] [--check]

Only Dio2KeyAgree0P1 through Dio2KeyAgree0P4 are emitted. They import the
prime-independent agreements from Dio2KeyAgree0. The prime-zero generator's
part-level proof template is specialized to the exported wire and Facts paths;
the shared PRF, LUT, family, gadget and claim-anchor proofs are not regenerated.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
import textwrap


HERE = Path(__file__).resolve().parent
ONLINE_SCOPE = "parallel_generatedRoot_1281"


def load_source(package):
    """Import the existing generator without running it or writing bytecode."""
    arguments = sys.argv
    sys.argv = [str(HERE / "gen_key_agree0.py"), str(package)]
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(
            "gen_key_agree0", HERE / "gen_key_agree0.py"
        )
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        return source
    finally:
        sys.argv = arguments


def theorem_block(template, name):
    """Extract one documented theorem and its local heartbeat option."""
    declaration = template.index(f"theorem {name} ")
    start = template.rfind("/--", 0, declaration)
    option = re.search(r"set_option[^\n]+ in\n$", template[:start])
    if option:
        start = option.start()
    following = re.search(
        r"\n\n(?=/--|/-!|set_option|@@)|\nend Dio2Proof", template[declaration:]
    )
    if following is None:
        raise ValueError(f"No end marker for {name}")
    return template[start:declaration + following.start()].rstrip() + "\n"


def online_wire(number, prime):
    """Map a prime-zero part wire to the corresponding exported prime wire."""
    if number == 1:
        return 9 + 56 * prime
    if 9 <= number <= 12:
        return number + 56 * prime + 5
    if 14 <= number <= 28:
        return number + 56 * prime + 4
    if number == 30:
        return number + 56 * prime + 3
    if 32 <= number <= 46:
        return number + 56 * prime + 2
    if number == 48:
        return number + 56 * prime + 1
    if 50 <= number <= 56:
        return number + 56 * prime
    return number


def verify_interfaces(source, package, prime):
    """Check the prime interface, proof wires and smudge-cell slice offsets."""
    stages_path = package.parent / "worst-n8-p5x50/generated/stages.json"
    stages = json.loads(stages_path.read_text())
    assert stages["keys_online_0_0"]["outputs"][prime] == f"dio2/key/{prime}"
    assert stages["preimages_0_0"]["outputs"][prime] == f"dio2/preimage/{prime}"
    root = 1267 + prime
    base_fields, base_lets = source.load(source.KS, "parallel_generatedRoot_1267")
    fields, lets = source.load(source.KS, f"parallel_generatedRoot_{root}")
    assert fields.keys() == base_fields.keys()
    assert lets.keys() == base_lets.keys()
    root_fields, _ = source.load(source.KS)
    assert f"parallel_generatedRoot_{root} " in root_fields[f"h{root}_0"][1]
    for field in ("h44_0", "h96_0"):
        assert f"({625 * prime} + (3125 * (Int.ofNat i_0)))" in fields[field][1]
    primes_text = (package / "proof/Dio2Online0PartPrimes.lean").read_text()
    theorem = theorem_block(primes_text, f"online0_part_prime{prime}")
    preimage, digits, target = (online_wire(number, prime) for number in (1, 11, 25))
    assert re.search(rf"\(hP : {ONLINE_SCOPE}\.w_{preimage}_0 ctx =", theorem)
    assert f"topRows ({ONLINE_SCOPE}.w_{digits}_0 ctx)" in theorem
    assert f"Kx k' * {ONLINE_SCOPE}.w_{target}_0 ctx" in theorem
    fresh_text = (package / "proof/Dio2Online0PartFresh.lean").read_text()
    fresh = theorem_block(fresh_text, f"online0_part_fresh{prime}")
    smudge = theorem_block(fresh_text, f"online0_part_smudge{prime}")
    for number in (37, 42):
        assert f"{ONLINE_SCOPE}.w_{online_wire(number, prime)}_0 ctx" in fresh
    assert f"{ONLINE_SCOPE}.w_{online_wire(55, prime)}_0 ctx" in smudge
    assert f"125 * i + {25 * prime} + c.val" in smudge


def specialize(text, source, prime):
    """Specialize only prime-dependent names, scoped wires and Facts fields."""
    root = 1267 + prime
    names = {
        "keys_part_rows0": f"keys_part_rows{prime}",
        "part_agree0": f"part_agree{prime}",
        "freshKey0": f"freshKey{prime}",
        "smudgeKey0": f"smudgeKey{prime}",
        "online0_hpre0": f"online0_hpre{prime}",
        "online0_hpre0'": f"online0_hpre{prime}'",
        "online0_hpre0_full": f"online0_hpre{prime}_full",
        "claim_link0": f"claim_link0_prime{prime}",
        "claim_hpre0": f"claim_hpre{prime}",
        "online1281_diagF0": f"online1281_diagF{prime}",
        "online1281_diagS0": f"online1281_diagS{prime}",
    }
    pattern = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    text = re.sub(rf"(?<!\w)({pattern})(?!\w)", lambda match: names[match[0]], text)
    text = text.replace("parallel_generatedRoot_1267", f"parallel_generatedRoot_{root}")
    text = text.replace(
        "W_Stage_keys_online_0_0.generatedRoot.w_1267_0",
        f"W_Stage_keys_online_0_0.generatedRoot.w_{root}_0",
    )
    text = text.replace("keys1267_diag", f"keys{root}_diag")
    text = re.sub(
        rf"{ONLINE_SCOPE}\.w_(\d+)(_\w+)",
        lambda match: f"{ONLINE_SCOPE}.w_{online_wire(int(match[1]), prime)}{match[2]}",
        text,
    )
    fields, _ = source.load(source.OS, ONLINE_SCOPE)

    def fact_path(match):
        field = f"h{online_wire(int(match[1]), prime)}{match[2]}"
        return "Fc." + fields[field][0].removeprefix("F.")

    text = re.sub(r"Fc\.c\d+\.h(\d+)(_\w+)", fact_path, text)
    text = text.replace("Fk.c49.h1267_0", root_field(source, root))
    text = text.replace(
        "W_Stage_online_0.generatedRoot.w_1_0 Con j",
        f"W_Stage_online_0.generatedRoot.w_{1276 + prime}_0 Con j",
    )
    _, lets = source.load(source.OS, ONLINE_SCOPE)
    preimage_input = lets[f"w_{online_wire(1, prime)}_0"].replace("ctx.", "c.")
    text = text.replace("show A * c.inputs.2.1 = _", f"show A * {preimage_input} = _")
    text = text.replace("125 * i + cc.val", f"125 * i + {25 * prime} + cc.val")
    text = text.replace("(125 * i)", f"(125 * i + {25 * prime})")
    projection = ".1" + ".2" * prime + (".1" if prime < 4 else "")
    text = text.replace("(claim_preimages h).1.1", f"(claim_preimages h){projection}")
    text = text.replace("@@RFL@@", ", ".join("rfl" for _ in source.link_pairs()))
    text = text.replace("prime-0", f"prime-{prime}")
    text = text.replace("prime `0`", f"prime `{prime}`")
    text = text.replace("online0_part_prime0", f"online0_part_prime{prime}")
    text = text.replace("online0_part_fresh0", f"online0_part_fresh{prime}")
    text = text.replace("online0_part_smudge0", f"online0_part_smudge{prime}")
    text = text.replace("cells `125 i + c`", f"cells `125 i + {25 * prime} + c`")
    preimage = online_wire(1, prime)
    text = text.replace("the part's input `w_1`", f"the part's input `w_{preimage}`")
    return re.sub(r"^\s*--[^\n]*\n", "", text, flags=re.MULTILINE)


def root_field(source, root):
    """Resolve the exported Facts path for one prime's keys-stage loop."""
    fields, _ = source.load(source.KS)
    return "Fk." + fields[f"h{root}_0"][0].removeprefix("F.")


def wrap_lean(text):
    """Keep generated Lean within the hundred-column proof-module convention."""
    lines = []
    for line in text.splitlines():
        indentation = line[:len(line) - len(line.lstrip())]
        lines.extend(textwrap.wrap(
            line.lstrip(), width=100, initial_indent=indentation,
            subsequent_indent=indentation + "    ", break_long_words=False,
            break_on_hyphens=False,
        ) if line else [""])
    return "\n".join(lines) + "\n"


def online_diagonals(source, prime):
    """Prove the two prime-local diagonal witnesses using their actual Facts."""
    fields, _ = source.load(source.OS, ONLINE_SCOPE)
    producers = {}
    for path, atom in fields.values():
        if atom.startswith("MxxRuntime.concatDiagonal "):
            _, arguments = source.app_split(atom)
            producers[source.strip_parens(arguments[-1])] = path
    scope = f"W_Stage_online_0.{ONLINE_SCOPE}"
    blocks = []
    for label, base, output in (("F", 31, 32), ("S", 49, 50)):
        output = online_wire(output, prime)
        lines = [f"  have e0 := isDiag_self ({scope}.w_{base}_0 c)"]
        for index in range(1, 25):
            wire = f"w_{output}_0" if index == 24 else f"w_{output}_concat_{index}"
            lines.append(
                f"  have e{index} := isDiag_step (k := {index}) (by norm_num) "
                f"(by norm_num) (by norm_num) (by norm_num) e{index - 1} {producers[wire]}"
            )
        lines.append("  exact e24")
        blocks.append(f"""/-- Prime `{prime}`'s {label}-digit matrix is diagonal in its shared input block. -/
theorem online1281_diag{label}{prime} {{backend : BackendContext}} {{hashModel : HashModel}}
    {{params : W_Stage_online_0.Params}} {{i : Nat}}
    {{c : {scope}.Ctx backend hashModel params i}} {{o : _}}
    (F : {scope}.Facts c o) :
    IsDiag ({scope}.w_{base}_0 c) ({scope}.w_{output}_0 c) := by
""" + "\n".join(lines))
    return "\n\n".join(blocks)


def generate(source, package, prime):
    """Emit only the prime-specific layers of the complete hpre proof."""
    verify_interfaces(source, package, prime)
    template = source.TEMPLATE
    keys_start = template.index("section KeysRows")
    keys_end = template.index("end KeysRows", keys_start) + len("end KeysRows")
    part_start = template.index("section PartAgree")
    part_end = template.index("end PartAgree", part_start) + len("end PartAgree")
    diagonals = source.diag_lemmas().split("/-- The online part", 1)[0]
    blocks = [diagonals, template[keys_start:keys_end], template[part_start:part_end]]
    for name in (
        "online0_hpre0", "online0_hpre0'", "online0_hpre0_full", "claim_link0", "claim_hpre0"
    ):
        blocks.append(theorem_block(template, name))
    header = f"""import Dio2KeyAgree0

/-!
# Step-zero key agreement for CRT prime {prime}

The keys stage's `parallel_generatedRoot_{1267 + prime}` exports `dio2/key/{prime}`.
Its two rows give the key combination required by `online0_part_prime{prime}`.
The fresh cells are `25 i + c`; the smudge cells are `125 i + {25 * prime} + c`.
The PRF totals, public and target families, gadget, and `Link0` are reused from
`Dio2KeyAgree0`. The final `claim_hpre{prime}` discharges precisely `hpre`, for
every part and both mask components, under the generated claim's run semantics.

Generated by `crates/io/lean/dio2/generator/gen_key_agree_prime.py`.
-/

set_option maxRecDepth 16384

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof
  RefreshTopProof.Gen RefreshTopProof.GenInst RefreshTopProof.GenFinal GenDecode

"""
    specialized = [specialize(block, source, prime).rstrip() for block in blocks]
    specialized.insert(1, online_diagonals(source, prime))
    body = "\n\n".join(specialized)
    result = header + body + "\n\nend Dio2Proof\n"
    assert "@@" not in result
    return wrap_lean(re.sub(r"\n{3,}", "\n\n", result))


def main():
    """Write all four assigned modules, or check their reproducibility."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", nargs="?", type=Path, default=HERE / "../average-n8-p5x50")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    package = arguments.package.resolve()
    source = load_source(package)
    outputs = {
        package / f"proof/Dio2KeyAgree0P{prime}.lean": generate(source, package, prime)
        for prime in range(1, 5)
    }
    for path, text in outputs.items():
        if arguments.check:
            if path.read_text() != text:
                raise SystemExit(f"Generated module is stale: {path.name}")
        else:
            if not path.is_file():
                raise SystemExit(f"Assigned placeholder is missing: {path.name}")
            if path.read_text() != text:
                path.write_text(text)
        print(f"{path.name}: {len(text.splitlines())} lines")


if __name__ == "__main__":
    main()
