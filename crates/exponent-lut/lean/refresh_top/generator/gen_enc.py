"""Write `Part_<c_j>_Enc.lean`: the encryptions of the encrypt stage's parallels: the label members
of every PRF and the LUT helpers at every level.

Usage: `python3 gen_enc.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_part import parts_of
from patterns import destruct, destruct_inline, parse, parse_inline
from segment import abbreviations, short


def info(path, name, ab):
    text = open(path).read()
    i = text.index(f'def {name} ')
    body = text[i:text.index('\n\n', i)]
    types = dict((m.group(1), short(m.group(2), ab)) for m in re.finditer(r'let (w_\S+) : (.*?) :=', body))
    consts = re.findall(r'matrixPolynomial \[\((\w+)\)\]', short(body, ab))
    return body, types, consts


def label(part, d, k, ab):
    ns = f'Stage_encrypt_{part}'
    path = f'{d}/generated/{ns}.lean'
    body, types, _ = info(path, f'parallel_generatedRoot_{k}', ab)
    Q, M = re.match(r'ExactMatrix (\S+) 8 1 (\d+)', types['w_0_0']).groups()
    dd = destruct(path, ns, f'parallel_generatedRoot_{k}', 'h') if 'abbrev parallel_generatedRoot_%d.body' % k in open(path).read() else destruct_inline(path, ns, f'parallel_generatedRoot_{k}', 'h')
    return f'''
set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Label member `o` of the encryption parallel `{k}`. -/
theorem encLabel{k} {{model : HashModel}} {{params : {ns}.Params}} {{o : Nat}}
    {{A00 A01 A10 A11 : ExactMatrix {Q} N 1 {M}}} {{smM oneM : ExactMatrix {Q} N 1 1}} {{c e : Int}} {{outputs}}
    (h : {ns}.parallel_generatedRoot_{k} Backend.backend model params o
      (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs)
    {{sm : ErrorPoly N}} (hsm : smM = fun _ _ ↦ reducePoly {Q} N sm) (hone : oneM = 1)
    {{G : ExactMatrix {Q} N 1 {M}}} (hG : gadgetMatrixRuns Backend.backend 32768 {M} G) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1) = (A00, A01, A10, A11) ∧
      Rel sm ![![A00, A01], ![A10, A11]]
        (fun j ↦ reducePoly {Q} N (message (n := N) c e * unitPair sm j) • G)
        ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] 26 ∧
      outputs.2.2.2.2.2.2.1 = fun _ _ ↦ reducePoly {Q} N (message (n := N) c e) := by
{dd}  have hg13 := gadgetMatrixRuns_unique h13_0 hG
  subst hg13
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h17_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h24_0
  subst hout
  refine ⟨rfl, ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩, ?_⟩
  · fin_cases j
    · show w_18_0 = _
      simp only [w_18_0, w_16_0, w_15_0, w_14_0, w_8_0, w_5_0, w_7_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_0_0, w_2_0, w_9_0, w_11_0, hsm, hone, he0, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    · show w_25_0 = _
      simp only [w_25_0, w_23_0, w_22_0, w_21_0, w_19_0, w_20_0, w_14_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_1_0, w_3_0, w_9_0, w_11_0, hsm, hone, he1, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
  · show w_12_0 = _
    simp only [w_12_0, w_10_0, w_9_0, w_11_0, multiplyMonomial_liftInteger]
'''


def helper(part, d, k, ab, low):
    """LUT helper `o` of the encryption parallel `k`: the top level of its PRF, or (`low`) a level
    below, reached by a modulus reduction of the secret."""
    ns = f'Stage_encrypt_{part}'
    path = f'{d}/generated/{ns}.lean'
    body, types, consts = info(path, f'parallel_generatedRoot_{k}', ab)
    if low:
        Q, M = re.match(r'ExactMatrix (\S+) 8 1 (\d+)', types['w_5_0']).groups()
        Qt = re.match(r'ExactMatrix (\S+) 8 1 1', types['w_0_0']).group(1)
        names = dict(sm='w_1_0', A00='w_4_0', A10='w_8_0', A01='w_20_0', A11='w_22_0', s='w_12_0',
                     s1='w_25_0', gad='h14_0', reb='h15_0', g='w_15_0', e0='h18_0', e1='h29_0',
                     c0='w_19_0', c1='w_30_0', sel='w_3_0', one='w_7_0')
        lets0 = 'w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0'
        lets1 = 'w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0'
    else:
        Q, M = re.match(r'ExactMatrix (\S+) 8 1 (\d+)', types['w_4_0']).groups()
        Qt = Q
        names = dict(sm='w_0_0', A00='w_3_0', A10='w_6_0', A01='w_18_0', A11='w_20_0', s='w_10_0',
                     s1='w_23_0', gad='h12_0', reb='h13_0', g='w_13_0', e0='h16_0', e1='h27_0',
                     c0='w_17_0', c1='w_28_0', sel='w_2_0', one='w_5_0')
        lets0 = 'w_17_0, w_15_0, w_14_0, w_11_0, w_8_0, w_4_0, w_7_0, w_9_0'
        lets1 = 'w_28_0, w_26_0, w_25_0, w_24_0, w_22_0, w_19_0, w_21_0, w_9_0'
    W = 2 if '[(1), (9)]' in body else 4
    sig, sodd = ('sigma2', 'sigma2_odd') if W == 2 else ('sigma', 'sigma_odd')
    p = consts[0]
    _, n, a = parse_inline(path, f'parallel_generatedRoot_{k}')
    atom = dict(zip(n, a))
    hA = next(x for x in n if x != '-' and re.search(rf'\b{names["A00"]}$', atom[x]))
    level = re.search(r'\[\.u64Le \((\d+)\), \.integer', atom[hA]).group(1)
    gw = re.search(r'gadgetMatrixRuns backend 32768 (\d+)', atom[names['gad']]).group(1)
    gadw = re.search(r'gadgetMatrixRuns backend 32768 \d+ (w_\d+_0)', atom[names['gad']]).group(1)
    gq = short(re.search(rf'∃ \({gadw} : Mxx\.Primitives\.ExactMatrix (\d+) 8 1', body).group(1), ab)
    dd = destruct_inline(path, ns, f'parallel_generatedRoot_{k}', 'h')
    hash_names = {0: names['A00'], 2: names['A10'], 1: names['A01'], 3: names['A11']}
    hashes = {c: next(x for x in n if x != '-' and re.search(rf'\b{w}$', atom[x])) for c, w in hash_names.items()}
    red = ''
    if low:
        red = f"""  have hsm' : {names['sm']} = fun _ _ ↦ reducePoly {Q} N sm :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : {names['one']} = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
"""
    else:
        red = f"""  have hsm' : {names['sm']} = fun _ _ ↦ reducePoly {Q} N sm := hsm
  have hone' : {names['one']} = 1 := hone
"""
    return f"""
set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `{k}`. -/
theorem encHelper{k} {{model : HashModel}} {{params : {ns}.Params}} (o : Fin {W})
    {{smM oneM : ExactMatrix {Qt} N 1 1}} {{key : ByteArray}} {{outputs}}
    (h : {ns}.parallel_generatedRoot_{k} Backend.backend model params o (smM, key, oneM, ()) outputs)
    {{sm : ErrorPoly N}} (hsm : smM = fun _ _ ↦ reducePoly {Qt} N sm) (hone : oneM = 1)
    {{g' : ExactMatrix {gq} N 1 {gw}}} {{g : ExactMatrix {Q} N 1 {M}}}
    (hg' : gadgetMatrixRuns Backend.backend 32768 {gw} g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key {Q} {M} {level} ({sig} o))
      (fun j ↦ reducePoly {Q} N ((({p} : Nat) : Int) *
        unitPair (substitution ({sig} o).toNat ({sodd} o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
{dd}{red}  rw [gadgetMatrixRuns_unique {names['gad']} hg'] at {names['reb']}
  have hgg := centeredRebaseRuns_unique {names['reb']} hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound {names['e0']}
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound {names['e1']}
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by norm_num) hN h{names['s'][2:]}
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by norm_num) hN h{names['s1'][2:]}
  have hsub' : {names['s']} = fun _ _ ↦ reducePoly {Q} N (substitution ({sig} o).toNat ({sodd} o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : {names['s1']} = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key {Q} {M} {level} ({sig} o) =
      ![![{names['A00']}, {names['A01']}], ![{names['A10']}, {names['A11']}]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq {hashes[0]}).symm
    · exact (hashSample_eq {hashes[1]}).symm
    · exact (hashSample_eq {hashes[2]}).symm
    · exact (hashSample_eq {hashes[3]}).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show {names['c0']} = _
    simp only [{lets0}, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show {names['c1']} = _
    simp only [{lets1}, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
"""


d = sys.argv[1]
ab = abbreviations(open(f'{d}/TopSetup.lean').read())
for part in parts_of(d):
    path = f'{d}/generated/Stage_encrypt_{part}.lean'
    text = open(path).read()
    ks = sorted({int(k) for k in re.findall(r'^def parallel_generatedRoot_(\d+) ', text, re.M)})
    groups = {}
    for k in ks:
        nm = f'parallel_generatedRoot_{k}'
        try:
            _, n, _ = parse(path, nm)
        except StopIteration:
            _, n, _ = parse_inline(path, nm)
        groups.setdefault(' '.join(x for x in n if x != '-'), []).append(k)
    labels = next(v for s, v in groups.items() if s.startswith('h13_0 h17_0'))
    out = [f'''import TopLemmas
import Stage_encrypt_{part}
import TopLut2

/-! The encryptions of the encrypt stage's parallels (written by `generator/gen_enc.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
''']
    for k in labels:
        out.append(label(part, d, k, ab))
    tops = next(v for s_, v in groups.items() if s_.startswith('h3_0 h6_0'))
    lows = next(v for s_, v in groups.items() if s_.startswith('h1_0 h4_0'))
    for k in tops:
        out.append(helper(part, d, k, ab, False))
    for k in lows:
        out.append(helper(part, d, k, ab, True))
    out.append(f'\nend RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_Enc.lean', 'w').write(''.join(out))
    print(part, labels)
    break
