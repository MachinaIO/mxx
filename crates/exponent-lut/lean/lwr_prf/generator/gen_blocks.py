"""LwrCells.lean and LwrBlock<b>.lean: every cell of every block, and the block sums."""
import os, re
from lean_patterns import proj

TEMPLATES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'average-n32-crt7x40')


def small(name):
    """A proof module of `average-n32-crt7x40`, whose cells, block sums, and readout LUT are the
    templates of every parameter set."""
    return open(os.path.join(TEMPLATES, f'{name}.lean')).read()


def cell_template(b):
    """The cell lemma of the small block `b` (0 or 1) with its parameters as placeholders."""
    if b == 0:
        s = small('LwrBlock0')
        s = s[s.index('/-- The error bound of a block-0 cell'):s.index('/-- The six blocks')]
        rep = [(r'\bq7_eq\b', '«QEB»'), (r'\bBackend\.layout6 layout7\b', '«LB»'),
               (r'\bBackend\.layout5 layout6\b', '«LN»'),
               (r'\bhQ7\b', '«HQB»'), (r'\bhQ6\b', '«HQN»'), (r'\bQ7\b', '«QB»'), (r'\bQ6\b', '«QN»'),
               (r'\bP7\b', '«PB»'), (r'\bP6\b', '«PR»'),
               (r'\b14\b', '«MA»'), (r'\b12\b', '«MS»'), (r'\b10\b', '«MR»'), (r'\b17\b', '«M»'),
               (r'\brhsPrefix0\b', '«RP»'), (r'\bcoords 0\b', '«CO»'), (r'\bcoords_lt 0\b', '«CL»'),
               (r'\bselectInt_coords0\b', '«SC»'), (r'\bcell0\b', '«CELL»'),
               (r'\bcellBound0\b', '«CB»'), (r'parallel_generatedRoot_23\b', '«PAR»'),
               (r'block `0`', 'block `«B»`'), (r'block-0', 'block-«B»')]
    else:
        s = small('LwrBlock1')
        s = s[s.index('/-- The error bound of a member of `a` aligned'):s.index('set_option maxHeartbeats 4000000 in\n/-- One iteration')]
        rep = [(r'\bq7_eq6\b', '«QEA»'), (r'\bq6_eq\b', '«QEB»'), (r'\bBackend\.layout6 layout7\b', '«LA»'),
               (r'\bBackend\.layout5 layout6\b', '«LB»'), (r'\bBackend\.layout4 layout5\b', '«LN»'),
               (r'\bhQ7\b', '«HQT»'), (r'\bhQ6\b', '«HQB»'), (r'\bhQ5\b', '«HQN»'), (r'\bQ7\b', '«QT»'),
               (r'\bQ6\b', '«QB»'), (r'\bQ5\b', '«QN»'), (r'\bA6\b', '«A»'), (r'\bP6\b', '«PB»'),
               (r'\bP5\b', '«PR»'),
               (r'\b14\b', '«MT»'), (r'\b12\b', '«MA»'), (r'\b10\b', '«MS»'), (r'\b8\b', '«MR»'),
               (r'\b17\b', '«M»'),
               (r'\brhsPrefix1\b', '«RP»'), (r'\bcoords 1\b', '«CO»'), (r'\bcoords_lt 1\b', '«CL»'),
               (r'\bselectInt_coords1\b', '«SC»'), (r'\bcell1\b', '«CELL»'), (r'\balignBound1\b', '«AB»'),
               (r'\bcellBound1\b', '«CB»'), (r'parallel_generatedRoot_45\b', '«PAR»'),
               (r'block `1`', 'block `«B»`'), (r'block-1', 'block-«B»')]
    rep = [(r'Fin 14\b', '«FINC»'), (r'\bfourteen\b', '«NC»')] + rep + \
        [(r'\b1048576\b', '«BASE»'), (r'\b524288\b', '«DIGIT»'), (r'\b26\b', '«CUT»')]
    for a, c in rep:
        s = re.sub(a, c, s)
    # The tight gains of monomial plaintexts.
    s = s.replace('N * 1 * B', '1 * B').replace('N * 1 * «CUT»', '1 * «CUT»')
    s = re.sub(r'\(hαμ : polyNorm \((α ⟨«CO» i, «CL» i⟩ \* μ)\) ≤ 1\)', r'(hαμ : SmulGain (\1) 1)', s)
    s = re.sub(r'\(hα : polyNorm \((α ⟨«CO» i, «CL» i⟩)\) ≤ 1\)', r'(hα : SmulGain (\1) 1)', s)
    assert s.count('SmulGain') == 2, s[:3000]
    old = '''  have hpayb := modulusReduce_poly «HQT» «HQB» hN (s := α ⟨«CO» i, «CL» i⟩)
    (by rw [← hpay]; exact h45_0)
'''
    if b == 1:
        assert old in s
        s = s.replace(old, '''  change modulusReduceRuns (apay ⟨«CO» i, «CL» i⟩) w_45_0 at h45_0
  rw [hpay ⟨«CO» i, «CL» i⟩] at h45_0
  have hpayb := modulusReduce_poly «HQT» «HQB» hN h45_0
''')
    s = s.replace('mul_public (q :=', 'mul_public_gain (q :=').replace('mul_private_down (q :=', 'mul_private_down_gain (q :=')
    return s


def fill(s, d):
    for a, c in d.items():
        s = s.replace(a, c)
    assert '«' not in s, re.findall(r'«\w+»', s)[:5]
    return s


def cells_module(I):
    C = I.C
    s = small('LwrBlock0')
    common = s[s.index('/-- The six blocks'):s.index('set_option maxHeartbeats 4000000 in\n/-- One iteration')]
    tail = s[s.index('/-- Cells `0` and `1` plus cells'):s.index('set_option maxHeartbeats 64000000 in\n/-- Block `0`')]
    common = common.replace('fourteen cells', f'{C} cells').replace('Fin 14', f'Fin {C}').replace('k < 14', f'k < {C}')
    tail = (tail.replace('cells `2, ..., 13` are all fourteen cells', f'cells `2, ..., {C - 1}` are all {C} cells')
            .replace('Fin 14', f'Fin {C}').replace('range 12', f'range {C - 2}').replace('range 14', f'range {C}')
            .replace('2 12', f'2 {C - 2}').replace('(2 + 12 : Nat) = 14', f'(2 + {C - 2} : Nat) = {C}'))
    # Leftover template counts, unless the counts are the template's.
    assert C == 14 or ('Fin 14' not in common + tail and not re.search(r'\b1[24]\b', tail)), tail
    return f'''import LwrSetup

/-!
# Cell families and their sums

The cells of a block form families of `{C}` matrices; the generated sum adds cells `0` and `1` in
the root and the others in a loop.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

''' + common + tail + 'end LwrProof\n'


def block_file(I, b):
    L, B, C, M, N = I.L, I.B, I.C, I.M, I.n
    qb, qn = L - b, L - b - 1
    par, seq = 23 + 22 * b, 42 + 22 * b
    d = {'«QB»': f'Q{qb}', '«QN»': f'Q{qn}', '«PB»': f'P{qb}', '«PR»': f'P{qn}', '«M»': str(M),
         '«MS»': str(I.m(qn)), '«MR»': str(I.m(qn - 1)), '«RP»': f'rhsPrefix{b}', '«CO»': f'coords {b}',
         '«CL»': f'coords_lt {b}', '«SC»': f'selectInt_coords{b}', '«CELL»': f'cell{b}',
         '«CB»': f'cellBound{b}', '«PAR»': f'parallel_generatedRoot_{par}', '«B»': str(b),
         '«FINC»': f'Fin {C}', '«NC»': str(C), '«BASE»': str(I.base), '«DIGIT»': str(I.digit), '«CUT»': str(I.cutoff),
         '«QEB»': f'q{qb}_eq', '«LB»': f'Backend.layout{qb - 1} layout{qb}',
         '«LN»': f'Backend.layout{qn - 1} layout{qn}', '«HQB»': f'hQ{qb}', '«HQN»': f'hQ{qn}'}
    if b == 0:
        d['«MA»'] = str(I.m(L))
        cell = fill(cell_template(0), d)
    else:
        d.update({'«QEA»': f'q{L}_eq{qb}', '«LA»': f'Backend.layout{L - 1} layout{L}', '«HQT»': f'hQ{L}',
                  '«QT»': f'Q{L}', '«A»': f'A{qb}', '«MT»': str(I.m(L)), '«MA»': str(I.m(qb)),
                  '«AB»': f'alignBound{b}'})
        cell = fill(cell_template(1), d)
    # The block sum loop.
    s = small('LwrBlock0')
    sums = s[s.index('set_option maxHeartbeats 4000000 in\n/-- One iteration of the block-0 sum'):s.index('/-- Cells `0` and `1` plus cells')]
    sums = sums.replace('sequential_generatedRoot_42', f'sequential_generatedRoot_{seq}')
    sums = re.sub(r'\bsum0_step\b', f'sum{b}_step', sums)
    sums = re.sub(r'\bsum0\b', f'sum{b}', sums).replace('block-0', f'block-{b}')
    sums = sums.replace('Q6 N 1 10', f'Q{qn} N 1 {I.m(qn - 1)}').replace('SumState Q6 10', f'SumState Q{qn} {I.m(qn - 1)}')
    sums = sums.replace('Fin 14', f'Fin {C}')
    return cell, sums, block_lemma(I, b)


def block_lemma(I, b):
    L, B, C, M, N, W = I.L, I.B, I.C, I.M, I.n, I.W
    qb, qn = L - b, L - b - 1
    par, seq = 23 + 22 * b, 42 + 22 * b
    prev = 42 + 22 * (b - 1)
    path = f'{I.d}/generated/Stage_online_eval.lean'
    init = [(24 + 22 * b + 3 * k, 25 + 22 * b + 3 * k) for k in range(6)]
    names = [f'h{par}_5'] + [f'h{x}_0' for pr in init for x in pr] + [f'h{seq}_5']
    if b == 0:
        names = ['h14_0', 'h15_0', 'h18_0', 'h19_0'] + names
    haves = proj(path, 'generatedRoot', names)
    fam = lambda k: f'w.w_{par}_{k}'
    eqs = ''
    for k, (x, y) in enumerate(init):
        eqs += f'''  obtain ⟨_, e{x}⟩ := familyGetStatic_eq (k := 0) h{x}_0
  replace e{x} : w.w_{x}_0 = {fam(k)} ⟨0, by decide⟩ := e{x}
  obtain ⟨_, e{y}⟩ := familyGetStatic_eq (k := 1) h{y}_0
  replace e{y} : w.w_{y}_0 = {fam(k)} ⟨1, by decide⟩ := e{y}
'''
    ms = ''.join(f'  have m{k} : w.w_{seq}_{k} = ∑ i : Fin {C}, {fam(k)} i := by rw [l{k}, e{x}, e{y}]; exact sum_cells _\n'
                 for k, (x, y) in enumerate(init))
    initt = ',\n      '.join(f'w.w_{x}_0 + w.w_{y}_0' for x, y in init)
    fams_ = [f'{{r{j}0 r{j}1 : Fin {C} → ExactMatrix Q{L - 1 - j} N 1 {I.m(L - 2 - j)}}}' for j in range(B)]
    rfams = '\n    '.join(' '.join(fams_[k:k + 2]) for k in range(0, B, 2))
    ins_ = [f'r{j}0, r{j}1' for j in range(B)]
    rins = ',\n        '.join(', '.join(ins_[k:k + 4]) for k in range(0, B, 4))
    mL, mL1, m1 = I.m(L), I.m(L - 1), I.m(1)
    ms_, mr_ = I.m(qn), I.m(qn - 1)
    if b == 0:
        hs = f'''(hs : Rel sm (stateKey model key)
      (fun j ↦ reducePoly Q{L} N (((P{L} : Nat) : Int) * μ * unitPair sp j) • g) ![cs0, cs1] {I.cutoff})'''
        statekey = '''  -- The state key.
  have hkey : stateKey model key = ![![w.w_14_0, w.w_18_0], ![w.w_15_0, w.w_19_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h14_0).symm
    · exact (hashSample_eq h18_0).symm
    · exact (hashSample_eq h15_0).symm
    · exact (hashSample_eq h19_0).symm
  rw [hkey] at hs
'''
        bound = f'cellBound0 {I.cutoff}'
        bs = ''
    else:
        hs = f'''(hs : Rel sm ![![w.w_{prev}_0, w.w_{prev}_1], ![w.w_{prev}_2, w.w_{prev}_3]]
      (fun j ↦ reducePoly Q{qb} N (((P{qb} : Nat) : Int) * μ * unitPair sp j) • g)
      ![w.w_{prev}_4, w.w_{prev}_5] Bs)'''
        statekey = ''
        bound = f'cellBound{b} Bs'
        bs = '{Bs : Nat} '
    return f'''
set_option maxHeartbeats 64000000 in
/-- Block `{b}`: the sum of its {C} cells, an encoding of `P{qn} Σ_i α_(c_i) μ ν_i` at `Q{qn}`. -/
theorem block{b} {{model : HashModel}} {{key : ByteArray}}
    {{s0 s1 s2 s3 : Fin {W} → ExactMatrix Q1 N 1 {m1}}} {{k0 k1 : Fin {W} → ExactMatrix Q2 N 1 {m1}}}
    {{ca0 ca1 : Fin {M} → ExactMatrix Q{L} N 1 {mL}}} {{cs0 cs1 : ExactMatrix Q{L} N 1 {mL1}}}
    {{apay : Fin {M} → ExactMatrix Q{L} N 1 1}}
    {rfams}
    {{outputs}} {{w : Stage_online_eval.generatedRoot.Witness}}
    (hb : Stage_online_eval.generatedRoot.body Backend.backend model tape path GeneratedClaim.stage_4_params
      (s0, s1, s2, s3, k0, key, k1, ca0, ca1, cs0, apay, cs1, {rins}, ()) outputs w)
    {{sm sp μ : ErrorPoly N}} {{α : Fin {M} → ErrorPoly N}} {{ν : Fin {C} → ErrorPoly N}}
    (hsm : polyNorm sm ≤ 1)
    (hαμ : ∀ i : Fin {C}, SmulGain (α ⟨coords {b} i, coords_lt {b} i⟩ * μ) 1)
    (hα : ∀ k, SmulGain (α k) 1)
    {{G : ExactMatrix Q{L} N 1 {mL}}} (hG : gadgetMatrixRuns Backend.backend {I.base} {mL} G)
    {{g' : ExactMatrix Q{qn} N 1 {ms_}}} {{g : ExactMatrix Q{qb} N 1 {ms_}}}
    (hg' : gadgetMatrixRuns Backend.backend {I.base} {ms_} g') (hg : centeredRebaseRuns g' g)
    {{h : ExactMatrix Q{qn} N 1 {mr_}}}
    (ha : ∀ k : Fin {M}, Rel sm (memberKey model key k)
      (fun j ↦ reducePoly Q{L} N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] {I.cutoff})
    (hpay : ∀ k : Fin {M}, apay k = fun _ _ ↦ reducePoly Q{L} N (α k))
    {bs}{hs}
    (hrhs : ∀ i : Fin {C}, ∃ e : Fin 2 → ErrorMatrix N 1 {mr_}, (∀ j, CoeffBound (e j) {I.cutoff}) ∧ ∀ j,
      reducePoly Q{qn} N sp • rhsTop model key Q{qn} {mr_} rhsPrefix{b} i j + ![r{b}0 i, r{b}1 i] j =
        reducePoly Q{qn} N (((P{qn} : Nat) : Int) * ν i * unitPair sp j) • h +
          reduceMatrix Q{qn} N 1 {mr_} (e j)) :
    Rel sm ![![w.w_{seq}_0, w.w_{seq}_1], ![w.w_{seq}_2, w.w_{seq}_3]]
      (∑ i : Fin {C}, fun j ↦ reducePoly Q{qn} N (((P{qn} : Nat) : Int) *
        (α ⟨coords {b} i, coords_lt {b} i⟩ * μ * ν i) * unitPair sp j) • h)
      ![w.w_{seq}_4, w.w_{seq}_5] (∑ _i : Fin {C}, {bound}) := by
  unfold Stage_online_eval.generatedRoot.body at hb
  extract_lets at hb
{haves}{statekey}  -- Every cell, then their sum.
  have hcells : ∀ i : Fin {C}, Rel sm ![![{fam(0)} i, {fam(1)} i], ![{fam(2)} i, {fam(3)} i]]
      (fun j ↦ reducePoly Q{qn} N (((P{qn} : Nat) : Int) * (α ⟨coords {b} i, coords_lt {b} i⟩ * μ * ν i) *
        unitPair sp j) • h) ![{fam(4)} i, {fam(5)} i] ({bound}) := by
    intro i
    obtain ⟨e, he, hr⟩ := hrhs i
    exact cell{b} (h{par}_5 i) hsm (hαμ i) (hα _) hG hg' hg ha hpay hs hr he
  have hsum := Rel.sum Finset.univ (fun i _ ↦ hcells i)
  -- The generated sum is the sum of the {C} cells.
{eqs}  have hloop := sum{b} (count := {C - 2}) (init := (
      {initt}, ()))
    (final := (w.w_{seq}_0, w.w_{seq}_1, w.w_{seq}_2, w.w_{seq}_3, w.w_{seq}_4, w.w_{seq}_5, ())) h{seq}_5
  simp only [addCells, Prod.mk.injEq] at hloop
  obtain ⟨l0, l1, l2, l3, l4, l5, -⟩ := hloop
{ms}  have hA : (∑ i : Fin {C}, ![![{fam(0)} i, {fam(1)} i], ![{fam(2)} i, {fam(3)} i]]) =
      ![![w.w_{seq}_0, w.w_{seq}_1], ![w.w_{seq}_2, w.w_{seq}_3]] := by
    funext k j
    fin_cases k <;> fin_cases j <;> simp [Finset.sum_apply, m0, m1, m2, m3]
  have hc : (∑ i : Fin {C}, ![{fam(4)} i, {fam(5)} i]) = ![w.w_{seq}_4, w.w_{seq}_5] := by
    funext j
    fin_cases j <;> simp [Finset.sum_apply, m4, m5]
  rw [hA, hc] at hsum
  exact hsum
'''


def block_module(I, b):
    L = I.L
    cell, sums, blk = block_file(I, b)
    qb, qn = L - b, L - b - 1
    what = (f'multiplies the state encoding at `Q{qb}` by member `coords {b} i` of `a` (a public\n'
            f'multiplication at the native gadget of `Q{qb}`)') if b == 0 else \
        (f'aligns member `coords {b} i` of `a` from `Q{L}` to `Q{qb}`, multiplies the state\n'
         f'encoding at `Q{qb}` by it')
    return f'''import LwrCells

/-!
# Online evaluation, block {b}

Cell `i` of block `{b}` {what}, then by the selector of the cell (a private multiplication that
drops `P{qb}`); the block sums its {I.C} cells.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxRecDepth 100000

variable {{tape : SampleTape}} {{path loopPath cellPath : List Nat}} {{loopPaths : Nat → List Nat}}

''' + cell + sums + blk + '\nend LwrProof\n'
