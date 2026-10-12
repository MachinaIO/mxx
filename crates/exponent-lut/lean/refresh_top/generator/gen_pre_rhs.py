"""Write `Part_<c_j>_PreRhs.lean`: the RHS bottoms the preprocessing computes for every PRF cell:
`s_p top_j + bot_j = p ν (s_p, 1)_j h + e_j`.

Usage: `python3 gen_pre_rhs.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_part import parts_of
from patterns import destruct_inline, parse_inline
from segment import abbreviations, short

WIRE = re.compile(r'\bw_\d+_(?:\d+|concat_\d+)\b')


def rhs(part, d, k, ab):
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
    outs = WIRE.findall(atom['hout'])
    top0, top1, bot0, bot1 = outs[4], outs[5], outs[6], outs[7]
    def args(w, op):
        m = re.match(rf'MxxRuntime\.{op} (\S+) (\S+)$', lets[w])
        return m.group(1), m.group(2)
    def producer(w):
        return next(x for x in n if x != '-' and re.search(rf'\b{w}$', atom[x]))
    # bot0 = (sp h p ν - sp top0) + e0
    x0, e0 = args(bot0, 'matrixAdd')
    s0, t0 = args(x0, 'matrixSub')
    u0, nu = args(s0, 'matrixMulScalarRight')
    v0, pc = args(u0, 'matrixMulScalarRight')
    sp, hw = args(v0, 'matrixMulScalarLeft')
    p = re.search(r'matrixPolynomial \[\((\w+)\)\]', lets[pc]).group(1)
    x1, e1 = args(bot1, 'matrixAdd')
    s1, _ = args(x1, 'matrixSub')
    u1, _ = args(s1, 'matrixMulScalarRight')
    v1, _ = args(u1, 'matrixMulScalarRight')
    one, _ = args(v1, 'matrixMulScalarLeft')
    Q, M = re.match(r'ExactMatrix (\S+) \d+ 1 (\d+)', types[top0]).groups()
    sec = WIRE.findall(atom[producer(sp)])[0]
    onein = WIRE.findall(atom[producer(one)])[0]
    nusrc = WIRE.findall(atom[producer(nu)])[0]
    lc, ex = args(nusrc, 'multiplyMonomial')
    sI = re.match(r'MxxRuntime\.liftInteger (\S+)$', lets[lc]).group(1)
    Qt = re.match(r'ExactMatrix (\S+) ', types[sec]).group(1)
    gad = WIRE.findall(atom[producer(hw)])[0]
    gq, gw = re.match(r'ExactMatrix (\S+) \d+ 1 (\d+)', types[gad]).groups()
    # the tag: its prefix and the components before `member` (an instance's digests)
    mt = re.search(r'hashSample \(hashModel\) \((\[[^\]]*\])\) \(\[(.*?)\.bytes \[109, 101, 109, 98, 101, 114\]',
                   atom[producer(top0)])
    pre = mt.group(2).strip().rstrip(',').strip()
    tag = f'(completeHashTag {mt.group(1)} [{pre}])' if pre else mt.group(1)
    tag_rw = 'rw [rhsTop_tag]; ' if pre else ''
    dd = destruct_inline(path, ns, name, 'h')
    inputs = re.search(r'\(inputs : (.*?)\) \(outputs', text[i:i + 4000]).group(1)
    nin = inputs.count(' × ')
    # The input tuple: key, the four one-encoding keys (or member families), sec, s, e, one.
    return f"""
set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `{k}`. -/
theorem preRhs{k} {{model : HashModel}} {{params : {ns}.Params}} {{i : Nat}} {{x : _}} {{outputs : _}}
    (h : {ns}.{name} Backend.backend model params i x outputs)
    {{sp : ErrorPoly N}} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly {Qt} N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {{g' : ExactMatrix {gq} N 1 {gw}}} {{g : ExactMatrix {Q} N 1 {M}}}
    (hg' : gadgetMatrixRuns Backend.backend 32768 {gw} g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 {M}, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly {Q} N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly {Q} N ((({p} : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix {Q} N 1 {M} (eB j) := by
{dd}  rw [gadgetMatrixRuns_unique {producer(gad)} hg'] at {producer(hw)}
  have hgg := centeredRebaseRuns_unique {producer(hw)} hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound {producer(e0)}
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound {producer(e1)}
  have hsp : {sp} = fun _ _ ↦ reducePoly {Q} N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN {producer(sp)} (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : {one} = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN {producer(one)} (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : {nu} = fun _ _ ↦ reducePoly {Q} N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN {producer(nu)}
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [{nusrc}, {lc}, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly {Q} N sp • {top0} + {bot0} = _
    simp only [{bot0}, {x0}, {s0}, {u0}, {v0}, {t0}, {pc}, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly {Q} N sp • {top1} + {bot1} = _
    simp only [{bot1}, {x1}, {s1}, {u1}, {v1}, {args(x1, 'matrixSub')[1]}, {pc}, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `{k}`. -/
theorem preTop{k} {{model : HashModel}} {{params : {ns}.Params}} {{i : Nat}} {{x : _}} {{outputs : _}}
    (h : {ns}.{name} Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 {Q} {M} {tag} i := by
{dd}  subst hout
  funext j
  fin_cases j
  · {tag_rw}exact hashSample_eq {producer(top0)}
  · {tag_rw}exact hashSample_eq {producer(top1)}
"""


d = sys.argv[1]
ab = abbreviations(open(f'{d}/TopSetup.lean').read())
for part in parts_of(d):
    from gen_keys_cells import cell_pairs
    pairs = cell_pairs(d, part)
    out = [f'''import TopLemmas
import Stage_preprocess_{part}
import ExponentScale
import ExponentRuntime

/-! The RHS of the PRF cells the preprocessing computes (written by `generator/gen_pre_rhs.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
''']
    for _, pre, _, _ in pairs:
        out.append(rhs(part, d, pre, ab))
    out.append(f'\nend RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_PreRhs.lean', 'w').write(''.join(out))
    print(part, [p[1] for p in pairs])
    break
