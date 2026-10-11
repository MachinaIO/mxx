"""Write `Part_<c_j>_EncRoot.lean`: the encryptions of the encrypt stage's root: the encoding of
one of every PRF child and the source input (`p μ (s_p, 1) g` under the mask), as segment theorems.

Usage: `python3 gen_enc_root.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_part import parts_of
from patterns import _block, parse
from segment import abbreviations, segment, short
from wiring import root_outputs

WIRE = re.compile(r'\bw_\d+_(?:\d+|concat_\d+)\b')


def one_enc(path, ns, names, atoms, vals, ab, c0, c1, tag):
    atom = dict(zip(names, atoms))
    def args(w, op):
        m = re.match(rf'MxxRuntime\.{op} (\S+) (\S+)$', vals[w])
        return m.group(1), m.group(2)
    def producer(w):
        return next(x for x in names if x != '-' and re.search(rf'\b{w}$', atom[x]))
    x0, e0 = args(c0, 'matrixAdd')
    s0, t0 = args(x0, 'matrixSub')
    a0, b0 = args(s0, 'matrixAdd')
    smR, A00 = args(a0, 'matrixMulScalarLeft')
    oneR, A10 = args(b0, 'matrixMulScalarLeft')
    u0, v = args(t0, 'matrixMulScalarRight')
    spR, g = args(u0, 'matrixMulScalarLeft')
    pc, msg = args(v, 'matrixMulScalarLeft')
    p = short(re.search(r'matrixPolynomial \[\((\w+)\)\]', vals[pc]).group(1), ab)
    lc, eI = args(msg, 'multiplyMonomial')
    cI = re.match(r'MxxRuntime\.liftInteger (\S+)$', vals[lc]).group(1)
    x1, e1 = args(c1, 'matrixAdd')
    s1, t1 = args(x1, 'matrixSub')
    a1, b1 = args(s1, 'matrixAdd')
    _, A01 = args(a1, 'matrixMulScalarLeft')
    _, A11 = args(b1, 'matrixMulScalarLeft')
    u1, _ = args(t1, 'matrixMulScalarRight')
    oneR1, _ = args(u1, 'matrixMulScalarLeft')
    used = [producer(e0), producer(e1)]
    reb = {}
    for w in {smR, oneR, spR, oneR1, g}:
        if w in vals:
            continue
        pr = next((x for x in names if x != '-' and re.search(rf'\b{w}$', atom[x])), None)
        if pr is None:
            continue
        reb[w] = pr
        used.append(reb[w])
    gad = WIRE.findall(atom[reb[g]])[0]
    gad_atom = producer(gad)
    used.append(gad_atom)
    used = [x for x in names if x in used]
    binders, _ = segment(path, ns, 'generatedRoot', used, ab, [c0, c1, A00, A01, A10, A11, cI, eI])
    eqs = [b.split(' : ')[0][1:] for b in binders if b.startswith('(e_')]
    keep = [e for e in eqs if e not in (f'e_{c0[2:]}', f'e_{c1[2:]}')]
    types = {b.split(' : ')[0][1:]: b.split(' : ', 1)[1][:-1] for b in binders if b.startswith('{w_')}
    Q, M = re.match(r'ExactMatrix (\S+) \d+ 1 (\d+)', types[g]).groups()
    smIn = WIRE.findall(atom[reb[smR]])[0] if smR in reb else smR
    spIn = WIRE.findall(atom[reb[spR]])[0] if spR in reb else spR
    Qin = re.match(r'ExactMatrix (\S+) ', types[smIn]).group(1)
    gq = re.match(r'ExactMatrix (\S+) ', types[gad]).group(1)
    bl = '\n    '.join(binders)
    def red(w, name):
        if w not in reb:
            return ''
        src = WIRE.findall(atom[reb[w]])[0]
        if name == 'one':
            return f"  have r_{w[2:]} : {w} = 1 := rebase_one (q := {Qin}) (by norm_num) (by rw [← {'hone' if src == oneIn else 'hone'}]; exact {reb[w]})\n"
        return ''
    oneIn = WIRE.findall(atom[reb[oneR]])[0] if oneR in reb else oneR
    proof = (f'  subst {" ".join(keep)}\n' if keep else '')
    lines = []
    lines.append(f"  rw [gadgetMatrixRuns_unique {gad_atom} hg'] at {reb[g]}")
    lines.append(f"  have hgg := centeredRebaseRuns_unique {reb[g]} hg")
    lines.append(f"  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound {producer(e0)}")
    lines.append(f"  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound {producer(e1)}")
    sim = []
    if smR in reb:
        lines.append(f"  have rsm : {smR} = fun _ _ ↦ reducePoly {Q} N sm := rebase_small (q := {Qin}) (by norm_num) hsm (by rw [← hsmIn]; exact {reb[smR]})")
    else:
        lines.append(f"  have rsm : {smR} = fun _ _ ↦ reducePoly {Q} N sm := hsmIn")
    sim.append('rsm')
    if spR in reb:
        lines.append(f"  have rsp : {spR} = fun _ _ ↦ reducePoly {Q} N sp := rebase_small (q := {Qin}) (by norm_num) hsp (by rw [← hspIn]; exact {reb[spR]})")
    else:
        lines.append(f"  have rsp : {spR} = fun _ _ ↦ reducePoly {Q} N sp := hspIn")
    sim.append('rsp')
    for k_, w in enumerate(sorted({oneR, oneR1})):
        if w in reb:
            src = WIRE.findall(atom[reb[w]])[0]
            lines.append(f"  have rone{k_} : {w} = 1 := rebase_one (q := {Qin}) (by norm_num) (by rw [← hone]; exact {reb[w]})")
        elif w in vals:
            continue
        else:
            lines.append(f"  have rone{k_} : {w} = 1 := hone")
        sim.append(f'rone{k_}')
    oneSrc = sorted({WIRE.findall(atom[reb[w]])[0] for w in {oneR, oneR1} if w in reb})
    lets0 = [c0, x0, s0, t0, a0, b0, u0, v, msg, lc] + [w for w in (pc,) if w in vals]
    lets1 = [c1, x1, s1, t1, a1, b1, u1]
    body = '\n'.join(lines)
    return f"""
set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The encryption `{tag}`: `p μ (s_p, 1) g` under the mask `(s_m, 1)`. -/
theorem encOne_{tag} {{model : HashModel}}
    {bl}
    {{sm sp : ErrorPoly N}} (hsm : polyNorm sm ≤ 1) (hsp : polyNorm sp ≤ 1)
    (hsmIn : {smIn} = fun _ _ ↦ reducePoly {Qin} N sm) (hspIn : {spIn} = fun _ _ ↦ reducePoly {Qin} N sp)
    {"(hone : " + oneSrc[0] + " = 1)" if oneSrc else ""}
    {{g' : ExactMatrix {gq} N 1 {M}}} {{gg : ExactMatrix {Q} N 1 {M}}}
    (hg' : gadgetMatrixRuns Backend.backend 32768 {M} g') (hg : centeredRebaseRuns g' gg) :
    Rel sm ![![{A00}, {A01}], ![{A10}, {A11}]]
      (fun j ↦ reducePoly {Q} N ((({p} : Nat) : Int) * message (n := N) {cI} {eI} * unitPair sp j) • gg)
      ![{c0}, {c1}] 26 := by
{proof}{body}
  subst hgg
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show {c0} = _
    rw [e_{c0[2:]}]
    simp only [{', '.join(sim)}, hE0, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show {c1} = _
    rw [e_{c1[2:]}]
    simp only [{', '.join(sim)}, hE1, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
"""


d = sys.argv[1]
ab = abbreviations(open(f'{d}/TopSetup.lean').read())
for part in parts_of(d):
    path = f'{d}/generated/Stage_encrypt_{part}.lean'
    ns = f'Stage_encrypt_{part}'
    lines = open(path).read().split('\n')
    body = _block(lines, 'abbrev generatedRoot.body')
    vals = {}
    for line in body:
        m = re.match(r'let (w_\S+) : (.*?) := (.*)$', line)
        if m and not m.group(3).startswith('inputs') and not m.group(3).startswith('witness'):
            vals[m.group(1)] = m.group(3)
    _, names, atoms = parse(path, 'generatedRoot')
    outs = root_outputs(path)
    pairs = []
    for i in range(len(outs) - 1):
        a_, b_ = outs[i], outs[i + 1]
        if a_ in vals and b_ in vals and vals[a_].startswith('MxxRuntime.matrixAdd') and \
                vals[b_].startswith('MxxRuntime.matrixAdd') and (i == 0 or outs[i - 1] not in vals or
                                                                   not vals[outs[i - 1]].startswith('MxxRuntime.matrixAdd')):
            pairs.append((a_, b_))
    out = [f'''import TopLemmas
import Stage_encrypt_{part}

/-! The encryptions of the encrypt stage's root (written by `generator/gen_enc_root.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
''']
    for c0, c1 in pairs:
        try:
            out.append(one_enc(path, ns, names, atoms, vals, ab, c0, c1, c0[2:]))
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print('skip', c0, c1, ex)
    out.append(f'\nend RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_EncRoot.lean', 'w').write(''.join(out))
    print(part, pairs)
    break
