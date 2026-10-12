"""Write `Part_<c_j>_PreSwitch.lean`: the LUT switching ciphertexts the preprocessing computes:
`σ(s_p) B⁰_j + B¹_j = (C_σ Δ' (s_p, 1)_j) g + e_j`.

Usage: `python3 gen_pre_switch.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_part import parts_of
from patterns import destruct_inline, parse_inline
from segment import abbreviations, short

WIRE = re.compile(r'\bw_\d+_(?:\d+|concat_\d+)\b')


def switch(part, d, k, ab):
    ns = f'Stage_preprocess_{part}'
    path = f'{d}/generated/{ns}.lean'
    text = open(path).read()
    name = f'parallel_generatedRoot_{k}'
    i = text.index(f'def {name} ')
    body = text[i:text.index('\n\n', i)]
    lets = dict((m.group(1), short(m.group(3), ab)) for m in re.finditer(r'let (w_\S+) : (.*?) := (.*)', body))
    types = dict((m.group(1), short(m.group(2), ab)) for m in re.finditer(r'let (w_\S+) : (.*?) :=', body))
    for m in re.finditer(r'∃ \((w_\S+) : (.*?)\),', body):
        types[m.group(1)] = short(m.group(2), ab)
    _, n, a = parse_inline(path, name)
    atom = dict(zip(n, a))
    def producer(w):
        return next(x for x in n if x != '-' and re.search(rf'\b{w}$', atom[x]))
    def args(w, op):
        m = re.match(rf'MxxRuntime\.{op} (\S+) (\S+)$', lets[w])
        return m.group(1), m.group(2)
    f00, f01, f10, f11 = WIRE.findall(atom['hout'])
    x0, e0 = args(f10, 'matrixAdd')
    a0, b0 = args(x0, 'matrixAdd')
    neg0, _ = args(a0, 'matrixMulScalarLeft')
    sw = re.match(r'MxxRuntime\.matrixNeg (\S+)$', lets[neg0]).group(1)
    s0, g = args(b0, 'matrixMulScalarLeft')
    d0, spw = args(s0, 'matrixMulScalarLeft')
    dc, C = args(d0, 'matrixMulScalarLeft')
    ic, S = args(C, 'matrixMulScalarLeft')
    x1, e1 = args(f11, 'matrixAdd')
    a1, b1 = args(x1, 'matrixAdd')
    neg1, _ = args(a1, 'matrixMulScalarLeft')
    s1, _ = args(b1, 'matrixMulScalarLeft')
    d1, onew = args(s1, 'matrixMulScalarLeft')
    # The table terms.
    def terms(w):
        if lets[w].startswith('MxxRuntime.matrixAdd'):
            l, r = args(w, 'matrixAdd')
            return terms(l) + terms(r)
        t, y = args(w, 'matrixMulScalarLeft')
        return [(t, y)]
    tl = terms(S)
    W = len(tl)
    def entry(w):
        m = re.match(r'\(MxxRuntime\.rotationPolynomial \((\d+)\)', lets[w])
        if m:
            return f'Xe (n := N) {m.group(1)}'
        m = re.match(r'\(MxxRuntime\.matrixPolynomial \[\((\w+)\)\]', lets[w])
        return f'(({m.group(1)} : Int) : ErrorPoly N)'
    T = '![' + ', '.join(entry(t) for t, _ in tl) + ']'
    delta = re.match(r'\(MxxRuntime\.matrixPolynomial \[\((\w+)\)\]', lets[dc]).group(1)
    inv = re.match(r'\(MxxRuntime\.matrixPolynomial \[\((\w+)\)\]', lets[ic]).group(1)
    Q, M = re.match(r'ExactMatrix (\S+) \d+ 1 (\d+)', types[f10]).groups()
    sec = WIRE.findall(atom[producer(spw)])[0]
    Qt = re.match(r'ExactMatrix (\S+) ', types[sec]).group(1)
    one = WIRE.findall(atom[producer(onew)])[0]
    gad = WIRE.findall(atom[producer(g)])[0]
    gq, gw = re.match(r'ExactMatrix (\S+) \d+ 1 (\d+)', types[gad]).groups()
    sig, sodd, coef, swc = ('sigma2', 'sigma2_odd', 'lutCoefficient2', 'switch_coeff2') if W == 2 else \
        ('sigma', 'sigma_odd', 'lutCoefficient', 'switch_coeff4')
    dd = destruct_inline(path, ns, name, 'h')
    ys = []
    for t, y in tl:
        ysrc = WIRE.findall(atom[producer(y)])[0]
        ys.append((y, ysrc, producer(y)))
    yfacts = ''
    for y, ysrc, pr in ys:
        yfacts += f"""  obtain ⟨_, hy{y[2:]}⟩ := ringAutomorphismRuns_eq (by norm_num) hN {pr}
"""
    lets_all = []
    def collect(w):
        if w in lets and w not in lets_all and not lets[w].startswith('inputs'):
            lets_all.append(w)
            for z in WIRE.findall(lets[w]):
                collect(z)
    collect(C)
    for _y, ysrc, _p in ys:
        collect(ysrc)
    if gad == g:
        gbind = f"""    {{g : ExactMatrix {Q} N 1 {M}}} (hg : gadgetMatrixRuns Backend.backend 32768 {gw} g)"""
        gproof = f"""  have hgg := gadgetMatrixRuns_unique {producer(g)} hg
  subst hgg
"""
    else:
        gbind = f"""    {{g' : ExactMatrix {gq} N 1 {gw}}} {{g : ExactMatrix {Q} N 1 {M}}}
    (hg' : gadgetMatrixRuns Backend.backend 32768 {gw} g') (hg : centeredRebaseRuns g' g)"""
        gproof = f"""  rw [gadgetMatrixRuns_unique {producer(gad)} hg'] at {producer(g)}
  have hgg := centeredRebaseRuns_unique {producer(g)} hg
  subst hgg
"""
    sigs = 'sigmas2' if W == 2 else 'sigmas'
    return f"""
set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The switching ciphertexts of branch `i` the preprocessing computes in `{k}`. -/
theorem preSwitch{k} {{model : HashModel}} {{params : {ns}.Params}} (i : Fin {W}) {{x : _}} {{outputs : _}}
    (h : {ns}.{name} Backend.backend model params i x outputs)
    {{sp : ErrorPoly N}} (hsec : x.2.1 = fun _ _ ↦ reducePoly {Qt} N sp) (hone : x.2.2.1 = 1)
{gbind} :
    ∃ e : Fin 2 → ErrorMatrix N 1 {M}, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly {Q} N (substitution ({sig} i).toNat ({sodd} i) sp) •
          ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 0 j +
        ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 1 j =
        ({coef} {Q} {inv} (fun k ↦ reducePoly {Q} N ({T} k)) ({sig} i).toNat *
          reducePoly {Q} N ((({delta} : Int) : ErrorPoly N) * unitPair sp j)) • g +
          reduceMatrix {Q} N 1 {M} (e j) := by
{dd}{gproof}  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound {producer(e0)}
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound {producer(e1)}
  have hsp : {spw} = fun _ _ ↦ reducePoly {Q} N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN {producer(spw)} (fun _ _ ↦ sp)
      (by show x.2.1 = _; rw [hsec]; rfl)
  have hone' : {onew} = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN {producer(onew)} (fun _ _ ↦ 1) (by
      show x.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  obtain ⟨_, hsw⟩ := ringAutomorphismRuns_eq (by norm_num) hN {producer(sw)}
  have hsw' : {sw} = fun _ _ ↦ reducePoly {Q} N (substitution ({sig} i).toNat ({sodd} i) sp) := by
    rw [hsw, hsp]
    funext r c
    exact (reducePoly_substitution _ _ _).symm
{yfacts}  have hC : {C} 0 0 = {coef} {Q} {inv} (fun k ↦ reducePoly {Q} N ({T} k)) ({sig} i).toNat := by
    rw [{coef}, ← {swc} ({sig} i).toNat ({sodd} i)]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three,
      Matrix.head_cons, reducePoly_Xe, map_intCast]
    simp only [{', '.join(sorted(set(lets_all), key=lambda z: int(z.split('_')[1])))}, {', '.join(f'hy{y[2:]}' for y, _, _ in ys)},
      matrixMulScalarLeft, matrixAdd, matrixNeg, rotationPolynomial, Matrix.add_apply, Matrix.neg_apply]
    simp [matrixPolynomial_single, reducePoly_Xe, Fin.sum_univ_four, X, {sig}, {sigs}, map_ofNat, map_one, map_zero]
    try ring
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly {Q} N _ • {f00} + {f10} = _
    simp only [{f10}, {x0}, {a0}, {b0}, {s0}, {d0}, {dc}, {neg0}, hsw', hsp, hE0]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring
  · show reducePoly {Q} N _ • {f01} + {f11} = _
    simp only [{f11}, {x1}, {a1}, {b1}, {s1}, {d1}, {dc}, {neg1}, hsw', hone', hE1]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring
"""


d = sys.argv[1]
ab = abbreviations(open(f'{d}/TopSetup.lean').read())
for part in parts_of(d):
    text = open(f'{d}/generated/Stage_preprocess_{part}.lean').read()
    ks = [int(m[1]) for m in re.findall(r'\(∀ i : Fin (\d+), parallel_generatedRoot_(\d+) backend hashModel params i \((w_\d+_0), \((w_\d+_0), \((w_\d+_0), \(\)\)\)\) ', text)]
    out = [f'''import TopSwitch
import Stage_preprocess_{part}
import ExponentScale
import ExponentAutomorphism

/-! The LUT switching ciphertexts the preprocessing computes (written by
`generator/gen_pre_switch.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
''']
    for k in sorted(ks):
        out.append(switch(part, d, k, ab))
    out.append(f'\nend RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_PreSwitch.lean', 'w').write(''.join(out))
    print(part, sorted(ks))
    break
