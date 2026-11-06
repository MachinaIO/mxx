#!/usr/bin/env python3
"""Generate PeChain{0,1,2,3}.lean: one online step with its preprocessed keys."""
import re

facts = open("generated/Stage_preprocess_Facts.lean").read()
start = facts.index("structure generatedRoot.Facts_0 ")
chunk_of = {}
cur = None
for line in facts[start:].splitlines():
    m = re.match(r"structure generatedRoot\.Facts_(\d+) ", line)
    if m:
        cur = int(m.group(1))
        continue
    if line.startswith("structure generatedRoot.Facts "):
        break
    m = re.match(r"  (h[0-9_]+) :", line)
    if m and cur is not None:
        chunk_of.setdefault(m.group(1), cur)

def online(base: int) -> dict:
    """The online root wires of a step whose state concatenation is `base` (not step 0)."""
    return dict(oTI0=str(base + 27), oTI1=str(base + 28), oTL0=str(base + 45),
                oTL1=str(base + 46), oOut0=str(base + 112), oOut1=str(base + 118))


def pre(s: int) -> dict:
    """The preprocessing root wires of step `s`: lane families and the output key."""
    offset = 138 * s
    return dict(pI=str(44 + offset), pL=str(45 + offset), pS0=str(126 + offset),
                pS1=str(132 + offset), pS2=str(138 + offset), pS3=str(144 + offset))


STEPS = {
    0: dict(oR0="8_0", oR1="9_0", oTI0="47", oTI1="48", oTL0="67", oTL1="68",
            oOut0="134", oOut1="140", off="0", pA0="", pA1="", C_="", kind="FIRST", **pre(0)),
    1: dict(oR0="134_0", oR1="140_0", off="128", pA0="147", pA1="148", C_="", kind="MID", **online(141), **pre(1)),
    2: dict(oR0="253_0", oR1="259_0", off="256", pA0="285", pA1="286", C_="", kind="MID", **online(260), **pre(2)),
    3: dict(oR0="372_0", oR1="378_0", off="384", pA0="423", pA1="424", C_="C", kind="MID", **online(379), **pre(3)),
}
for k in (1, 2, 3):
    prev = pre(k - 1)
    STEPS[k].update(pP0=prev["pS0"], pP1=prev["pS1"], pP2=prev["pS2"], pP3=prev["pS3"])
STEPS[0].update(pP0="", pP1="", pP2="", pP3="")

TEMPLATE = open("PeChain.lean.in").read()
for k, d in STEPS.items():
    d = dict(d, k=str(k), table="tableConst" if d["C_"] else "tableMono")
    keep = {d["kind"], "CONST" if d["C_"] else "MONO"}
    lines = []
    for line in TEMPLATE.splitlines():
        m = re.search(r" --(FIRST|MID|MONO|CONST)$", line)
        if m:
            if m.group(1) not in keep:
                continue
            line = line[: m.start()]
        lines.append(line)
    text = "\n".join(lines) + "\n"
    text = re.sub(r"«(\w+)»", lambda m: d[m.group(1)], text)
    text = re.sub(r"@@(h[0-9_]+)@@", lambda m: f"G.c{chunk_of[m.group(1)]}.{m.group(1)}", text)
    open(f"PeChain{k}.lean", "w").write(text)

all_fields = {f"{field}_{step}": value for step, fields in STEPS.items()
              for field, value in fields.items()}
text = re.sub(r"«(\w+)»", lambda match: all_fields[match.group(1)],
              open("PeChainAll.lean.in").read())
open("PeChainAll.lean", "w").write(text)
