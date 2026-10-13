#!/usr/bin/env python3
"""Generate PePreRoot{0,1,2,3}.lean: the preprocessing steps at the root wires."""
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

def step(posW, tops, sa, sb, sc, sd, sel, A, M, lut, li, intw, lut2, l2i, T, coef, g, V, sums, off, C_):
    return dict(posW=posW, tops=tops, sa=sa, sb=sb, sc=sc, sd=sd, m00=sel[0], m01=sel[1], m10=sel[2],
                m11=sel[3], A0=A[0], A1=A[1], M0=M[0], M1=M[1], lt=lut[0], lk=lut[1], int=intw,
                l2t=lut2[0], l2k=lut2[1], T0=T[0], T1=T[1], coef=coef, g=g, V=V, sI=sums[0], sL=sums[1],
                sL2=sums[2], off=off, C_=C_)

STEPS = {
    0: step(1, [(10, 11), (14, 15), (18, 19), (22, 23)], [12, 16, 20, 24], [26, 27, 28, 29],
            [31, 32, 33, 34], [36, 37, 38, 39], [25, 30, 35, 40], (6, 9), (42, 43), (44, 45), 41, 47,
            (48, 49), 46, (50, 51), 55, 57, 59, (64, 65, 66), 0, ""),
    1: step(67, [(71, 72), (75, 76), (79, 80), (83, 84)], [73, 77, 81, 85], [87, 88, 89, 90],
            [92, 93, 94, 95], [97, 98, 99, 100], [86, 91, 96, 101], (69, 70), (103, 104), (105, 106), 102,
            108, (109, 110), 107, (111, 112), 115, 117, 119, (124, 125, 126), 128, ""),
    2: step(127, [(131, 132), (135, 136), (139, 140), (143, 144)], [133, 137, 141, 145],
            [147, 148, 149, 150], [152, 153, 154, 155], [157, 158, 159, 160], [146, 151, 156, 161],
            (129, 130), (163, 164), (165, 166), 162, 168, (169, 170), 167, (171, 172), 175, 177, 179,
            (184, 185, 186), 256, ""),
    3: step(187, [(191, 192), (195, 196), (199, 200), (203, 204)], [193, 197, 201, 205],
            [207, 208, 209, 210], [212, 213, 214, 215], [217, 218, 219, 220], [206, 211, 216, 221],
            (189, 190), (223, 224), (225, 226), 222, 228, (229, 230), 227, (231, 232), 235, 237, 239,
            (244, 245, 246), 384, "C"),
}

TEMPLATE = open("PePreRoot.lean.in").read()
head, rest = TEMPLATE.split("@@POS@@\n")
pos, tail = rest.split("@@ENDPOS@@\n")

def subst(text, d):
    text = re.sub(r"«(\w+)»", lambda m: str(d[m.group(1)]), text)
    return re.sub(r"@@(h[0-9_]+)@@", lambda m: f"F.c{chunk_of[m.group(1)]}.{m.group(1)}", text)

for s, d in STEPS.items():
    d = dict(d, s=s)
    blocks = []
    for q in range(4):
        dq = dict(d, q=q, top=d["tops"][q][0], key=d["tops"][q][1], sa=d["sa"][q], sb=d["sb"][q],
                  sc=d["sc"][q], sd=d["sd"][q])
        blocks.append(subst(pos, dq))
    text = subst(head, d) + "".join(blocks) + subst(tail, d)
    open(f"PePreRoot{s}.lean", "w").write(text)
