#!/usr/bin/env python3
"""Generate PeOnlineRoot{0,1,2,3}.lean: each online step at the root wires."""
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


def step(base: int, k: int) -> dict:
    """The root wires of step `k`; `base` is the step's state concatenation."""
    w = lambda o: str(base + o)
    d = dict(C=w(0), pos=w(1), t0=w(2), t1=w(3), top=w(4), y0=w(5), y1=w(6), bot=w(7), col=w(8),
             a00=w(10), a01=w(11), A0=w(12), a10=w(13), a11=w(14), A1=w(15), mul=w(16), P=w(17),
             h0=w(18), coef=w(21), g=w(23), V=w(25), TI0=w(27), TI1=w(28), h1=w(29), m00=w(30),
             m01=w(31), M0=w(32), m10=w(33), m11=w(34), M1=w(35), int=w(36), sa=w(37),
             sb=w(38), s0=w(39), sc=w(40), sd=w(41), s1=w(42), sum=w(43), TL0=w(45),
             TL1=w(46), out0=w(112), out1=w(118), k=str(k))
    d.update(group_fields(base + 53))
    d['families0'] = '![%s]' % ', '.join(f'generatedRoot.w_{base + 45 + 17 * group + 2 * slot}_0 ctx'
                                      for group in range(4) for slot in range(4))
    d['families1'] = '![%s]' % ', '.join(f'generatedRoot.w_{base + 46 + 17 * group + 2 * slot}_0 ctx'
                                      for group in range(4) for slot in range(4))
    return d


def group_fields(first):
    fields = {}
    for group in range(4):
        root = first + 17 * group
        for key, offset in [('leaf', 0), ('ms_a', 1), ('ms_b', 2), ('ms0', 3),
                            ('ms_c', 4), ('ms_d', 5), ('ms1', 6), ('msum', 7)]:
            fields[f'{key}{group}'] = str(root + offset)
    return fields


# Step 0 differs before its product: its state is an input and its key the hashed input key.
S0 = dict(C="10", pos="14", t0="15", t1="16", top="17", y0="19", y1="21", bot="22", col="23", mul="33",
          P="34", h0="35", coef="39", g="41", V="43", h1="49", TI0="47", TI1="48", m00="52", m01="53",
          M0="54", m10="55", m11="56", M1="57", int="58", sa="59", sb="60", s0="61", sc="62",
          sd="63", s1="64", sum="65", TL0="67", TL1="68", out0="134", out1="140",
          r0="8_0", r1="9_0", k="0",
          families0='![%s]' % ', '.join(f'generatedRoot.w_{67 + 17 * group + 2 * slot}_0 ctx'
                                      for group in range(4) for slot in range(4)),
          families1='![%s]' % ', '.join(f'generatedRoot.w_{68 + 17 * group + 2 * slot}_0 ctx'
                                      for group in range(4) for slot in range(4)), **group_fields(75))
STEPS = {0: dict(S0, kind="FIRST", off="0", C_="")}
for k, base, prev in ((1, 141, (134, 140)), (2, 260, (253, 259)), (3, 379, (372, 378))):
    d = step(base, k)
    d.update(r0=f"{prev[0]}_0", r1=f"{prev[1]}_0", kind="MID", off=str(128 * k), C_="C" if k == 3 else "")
    STEPS[k] = d

TEMPLATE = open("PeOnlineRoot.lean.in").read()

for k, d in STEPS.items():
    d = dict(d, table="tableConst" if d["C_"] else "tableMono")
    d['runs'] = ' '.join(f'((lanes{group} member).1 {slot})'
                         for group in range(4) for slot in range(4))
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
    text = re.sub(r'«GROUPS»(.*?)«ENDGROUPS»',
                  lambda match: '\n'.join(re.sub(r'«(\w+)g»',
                    lambda field: '«' + field[1] + str(group) + '»',
                    match[1].replace('«group»', str(group))) for group in range(4)),
                  text, flags=re.S)
    text = re.sub(r"«(\w+)»", lambda m: d[m.group(1)], text)
    text = re.sub(r"@@(h[0-9_]+)@@", lambda m: f"F.c{chunk_of[m.group(1)]}.{m.group(1)}", text)
    open(f"PeOnlineRoot{k}.lean", "w").write(text)
