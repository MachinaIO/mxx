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

STEPS = {
    0: dict(oR0="8_0", oR1="9_0", oOut="70", oT0="57", oT1="58", off="0", pT0="50", pT1="51", pSum="66",
            pPrev="", pA0="", pA1="", C_="", kind="FIRST"),
    1: dict(oR0="70_0", oR1="70_1", oOut="119", oT0="108", oT1="109", off="128", pT0="111", pT1="112",
            pSum="126", pPrev="66", pA0="69", pA1="70", C_="", kind="MID"),
    2: dict(oR0="119_0", oR1="119_1", oOut="168", oT0="157", oT1="158", off="256", pT0="171", pT1="172",
            pSum="186", pPrev="126", pA0="129", pA1="130", C_="", kind="MID"),
    3: dict(oR0="168_0", oR1="168_1", oOut="217", oT0="206", oT1="207", off="384", pT0="231", pT1="232",
            pSum="246", pPrev="186", pA0="189", pA1="190", C_="C", kind="MID"),
}

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
