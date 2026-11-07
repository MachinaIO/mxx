#!/usr/bin/env python3
"""Generate PePreRoot{0,1,2,3}.lean: the preprocessing steps at the root wires."""
import re
from pathlib import Path
from date_parameters import write_lean, reference_facts

facts = reference_facts("pre")
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


def step(s):
    """The root wires of preprocessing step `s`.

    The product part of step `s ≥ 1` sits `139 + 138 (s - 1)` wires after step zero.
    Each LUT step has an internal family and four leaf groups, at stride 138.
    """
    p = 0 if s == 0 else 139 + 138 * (s - 1)
    q = 138 * s
    d = dict(s=s, posW=1 if s == 0 else 145 + 138 * (s - 1),
             tops=[(10 + p, 11 + p), (14 + p, 15 + p), (18 + p, 19 + p), (22 + p, 23 + p)],
             sa=[12 + p, 16 + p, 20 + p, 24 + p], sb=[26 + p + k for k in range(4)],
             sc=[31 + p + k for k in range(4)], sd=[36 + p + k for k in range(4)],
             m00=25 + p, m01=30 + p, m10=35 + p, m11=40 + p, M0=41 + p, M1=42 + p,
             I=44 + q, L=45 + q, L1=46 + q, L2=47 + q, L3=48 + q,
             C=51 + q, g=53 + q, V=55 + q,
             k0=126 + q, k1=132 + q, k2=138 + q, k3=144 + q, SI=69 + q, SL=83 + q,
             off=128 * s, C_="C" if s == 1 else "")
    d["A0"], d["A1"] = (6, 9) if s == 0 else (147 + 138 * (s - 1), 148 + 138 * (s - 1))
    d["families"] = "fun slot r member ↦ (![" + ", ".join(
        f"![generatedRoot.w_{45 + q + slot // 4}_{2 * (slot % 4)} ctx member, generatedRoot.w_{45 + q + slot // 4}_{2 * (slot % 4) + 1} ctx member]"
        for slot in range(16)) + "]) slot r"
    d['group_families'] = 'fun group slot r member ↦ (![' + ', '.join(
        '![' + ', '.join(f'![generatedRoot.w_{45 + q + group}_{2 * slot} ctx member, '
                         f'generatedRoot.w_{45 + q + group}_{2 * slot + 1} ctx member]'
                         for slot in range(4)) + ']' for group in range(4)) + ']) group slot r'
    def cases(proofs):
        result = '(by intro index; exact Fin.elim0 index)'
        for proof in reversed(proofs):
            result = f'Fin.cases ({proof}) ({result})'
        return result
    d['chunk_proof'] = cases([cases(['rfl'] * 4)] * 4)
    d['leafKeys'] = ' +\n    '.join(
        f'keyOf (generatedRoot.w_{45 + q + group}_8 ctx member, '
        f'generatedRoot.w_{45 + q + group}_9 ctx member, '
        f'generatedRoot.w_{45 + q + group}_10 ctx member, '
        f'generatedRoot.w_{45 + q + group}_11 ctx member, ()) r j' for group in range(4))
    leaf_sums = []
    sum_keys = []
    for group in range(4):
        base = 71 + 14 * group + q
        fields = [base, base + 1, base + 3, base + 4, base + 6, base + 7, base + 9, base + 10]
        proofs = ' '.join(f'@@h{wire}_0@@' for wire in fields)
        sl = 83 + 14 * group + q
        leaf_sums.append(f'    have S2{group} := PreSum{s}.sumLeaf' + (f'G{group}' if group else '') +
            ' (model := model) (tape := tape) (path := path) (params := params)' +
                         f' (member_init (n := 13) {proofs}) @@h{sl}_3@@\n')
        sum_keys.append('keyOf (' + ', '.join(f'generatedRoot.w_{sl}_{port} ctx' for port in range(4)) + ', ()) r j')
    d['leaf_sum_proofs'] = ''.join(leaf_sums)
    d['leaf_sum_keys'] = ' +\n        '.join(sum_keys)
    d['key_association'] = '\n'.join('      · change ' +
        ' + '.join([f'(preM_{s} ctx {row} {column} * generatedRoot.w_{55 + q}_0 ctx)',
                    f'generatedRoot.w_{69 + q}_{2 * row + column} ctx'] +
                   [f'generatedRoot.w_{83 + q + 14 * group}_{2 * row + column} ctx' for group in range(4)]) +
        ' = ' + f'preM_{s} ctx {row} {column} * generatedRoot.w_{55 + q}_0 ctx + '
        f'generatedRoot.w_{69 + q}_{2 * row + column} ctx + (' +
        ' + '.join(f'generatedRoot.w_{83 + q + 14 * group}_{2 * row + column} ctx' for group in range(4)) +
        ')\n        abel' for row in range(2) for column in range(2))
    for name, base in (("rI", 57),):
        for k, w in enumerate([base, base + 1, base + 3, base + 4, base + 6, base + 7, base + 9, base + 10]):
            d[f"{name}{k}"] = w + q
    return d


STEPS = {s: step(s) for s in range(2)}

TEMPLATE = open("PePreRoot.lean.in").read()
head, rest = TEMPLATE.split("@@POS@@\n")
pos, tail = rest.split("@@ENDPOS@@\n")


def subst(text, d):
    text = re.sub(r"«(\w+)»", lambda m: str(d[m.group(1)]), text)
    return re.sub(r"@@(h[0-9_]+)@@", lambda m: f"F.c{chunk_of[m.group(1)]}.{m.group(1)}", text)


for s, d in STEPS.items():
    blocks = []
    for q in range(2):
        dq = dict(d, q=q, top=d["tops"][q][0], key=d["tops"][q][1], sa=d["sa"][q], sb=d["sb"][q],
                  sc=d["sc"][q], sd=d["sd"][q])
        blocks.append(subst(pos, dq))
    text = subst(head, d) + "".join(blocks) + subst(tail, d)
    write_lean(Path(f"PePreRoot{s}.lean"), text, "pre", root=True)
