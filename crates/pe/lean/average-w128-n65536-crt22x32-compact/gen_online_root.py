#!/usr/bin/env python3
"""Generate PeOnlineRoot{1,2,3}.lean: the online steps after the first at the root wires."""
import re

facts = open("generated/Stage_online_eval_Facts.lean").read()
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
    1: dict(r0="70_0", r1="70_1", C="71", pos="72", t0="73", t1="74", top="75", y0="76", y1="77", bot="78",
            col="79", a00="81", a01="82", A0="83", a10="84", a11="85", A1="86", mul="87", P="88", h0="89",
            h1="98", coef="92", g="94", V="96", m00="100", m01="102", m10="104", m11="106", M0="111",
            M1="112", lut="113", T0="108", T1="109", int="114", sum="115", leaf="116", lut2="118",
            leaf2="119", off="128", C_="", ),
    2: dict(r0="119_0", r1="119_1", C="120", pos="121", t0="122", t1="123", top="124", y0="125", y1="126",
            bot="127", col="128", a00="130", a01="131", A0="132", a10="133", a11="134", A1="135", mul="136",
            P="137", h0="138", h1="147", coef="141", g="143", V="145", m00="149", m01="151", m10="153",
            m11="155", M0="160", M1="161", lut="162", T0="157", T1="158", int="163", sum="164",
            leaf="165", lut2="167", leaf2="168", off="256", C_=""),
    3: dict(r0="168_0", r1="168_1", C="169", pos="170", t0="171", t1="172", top="173", y0="174", y1="175",
            bot="176", col="177", a00="179", a01="180", A0="181", a10="182", a11="183", A1="184", mul="185",
            P="186", h0="187", h1="196", coef="190", g="192", V="194", m00="198", m01="200", m10="202",
            m11="204", M0="209", M1="210", lut="211", T0="206", T1="207", int="212", sum="213",
            leaf="214", lut2="216", leaf2="217", off="384", C_="C"),
}

TEMPLATE = open("PeOnlineRoot.lean.in").read()

for k, d in STEPS.items():
    d = dict(d, k=str(k), table="tableConst" if d["C_"] else "tableMono")
    keep = "--CONST" if d["C_"] else "--MONO"
    drop = "--MONO" if d["C_"] else "--CONST"
    lines = []
    for line in TEMPLATE.splitlines():
        if line.endswith(drop):
            continue
        if line.endswith(keep):
            line = line[: -len(keep)].rstrip()
        lines.append(line)
    text = "\n".join(lines) + "\n"
    text = re.sub(r"«(\w+)»", lambda m: d[m.group(1)], text)
    text = re.sub(r"@@(h[0-9_]+)@@", lambda m: f"F.c{chunk_of[m.group(1)]}.{m.group(1)}", text)
    open(f"PeOnlineRoot{k}.lean", "w").write(text)
