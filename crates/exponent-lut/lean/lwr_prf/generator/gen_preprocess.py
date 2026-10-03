"""LwrPreprocess.lean: the selector RHS of every block and the readout LUT's switching ciphertexts."""
from lean_patterns import pattern, proj, atom_kinds, destruct
from gen_encrypt import split_top


def tup(k):
    return 'outputs' + '.2' * k + '.1'


NAMES11 = dict(top0='h17_0', top1='h54_0', pay='h32_0', gad='h33_0', reb='h34_0', msg='h42_0', e0='h46_0',
               one='h59_0', e1='h65_0', w0='w_17_0', w1='w_54_0')
NAMES27 = dict(top0='h30_0', top1='h67_0', pay='h45_0', gad='h46_0', reb='h47_0', msg='h55_0', e0='h59_0',
               one='h72_0', e1='h78_0', w0='w_30_0', w1='w_67_0')


def preprocess(I):
    L, B, C, W, N = I.L, I.B, I.C, I.W, I.n
    path = f'{I.d}/generated/Stage_preprocess.lean'
    pre = I.pre
    root = pre[pre.index('def generatedRoot '):]
    outs = split_top(root[root.index('(outputs : ') + 11:root.index(') : Prop')])
    ins = ', '.join(['key', 'payload'] + [f'b{j}, x{j}' for j in range(B)])
    binders = ' '.join(f'b{j} x{j}' for j in range(B))
    parts = [f'''import LwrSetup

/-!
# The preprocessing stage

The selector of cell `i` of block `j` is preprocessed into the RHS ciphertext
`s top_j + bot_j = (P ν r_j) g + e_j` at the modulus of the block's private multiplication, and
the readout LUT's switching ciphertexts at `Q1` satisfy
`σ_i(s) B_{{0j}} + B_{{1j}} = (C_i ⌊Q1/2⌋ r_j) · 1 + e_j`.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxRecDepth 100000

variable {{tape : SampleTape}} {{path : List Nat}}
''']
    pat11 = pattern(path, 'parallel_generatedRoot_11')
    pat27 = pattern(path, 'parallel_generatedRoot_27')
    rp = pattern(path, 'generatedRoot')[1:-1].split(', ')
    for j in range(B):
        par = 11 + 16 * j
        q, qg, p, m = L - 1 - j, L - 2 - j, L - 1 - j, I.m(L - 2 - j)
        pat = pat11 if j == 0 else pat27
        nm = NAMES11 if j == 0 else NAMES27
        assert pat == pattern(path, f'parallel_generatedRoot_{par}'), par
        haves = proj(path, 'generatedRoot', [f'h{par}_7', 'hout'])
        haves = haves.replace(f'have h{par}_7 :=', 'have hpar :=')
        want = f'(Fin {C} → Mxx.Primitives.ExactMatrix {I.Q[q]} {N} 1 {m})'
        pos = [i for i, x in enumerate(outs) if x.strip() == want]
        assert len(pos) == 2 and pos[1] == pos[0] + 1, (j, pos)
        parts.append(f'''
set_option maxHeartbeats 8000000 in
/-- The selector RHS of member `i` of block `{j}` at `Q{q}`: `s top_j + bot_j = (P{p} ν r_j) g + e_j`. -/
theorem preprocess_selector{j} {{model : HashModel}} {{key : ByteArray}} {{payload : ExactMatrix Q{L} N 1 1}}
    {{{binders} : Fin {C} → Int}} {{outputs}}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape path GeneratedClaim.stage_2_params
      ({ins}, ()) outputs)
    {{sp : ErrorPoly N}} (hpayload : payload = fun _ _ ↦ reducePoly Q{L} N sp)
    {{g' : ExactMatrix Q{qg} N 1 {I.m(qg)}}} {{g : ExactMatrix Q{q} N 1 {I.m(qg)}}}
    (hg' : gadgetMatrixRuns Backend.backend {I.base} {I.m(qg)} g') (hg : centeredRebaseRuns g' g) (i : Fin {C}) :
    ∃ e : Fin 2 → ErrorMatrix N 1 {m}, (∀ jj, CoeffBound (e jj) {I.cutoff}) ∧ ∀ jj,
      reducePoly Q{q} N sp • rhsTop model key Q{q} {m} rhsPrefix{j} i jj + ![{tup(pos[0])} i, {tup(pos[1])} i] jj =
        reducePoly Q{q} N (((P{p} : Nat) : Int) * message (n := N) (b{j} i) (x{j} i) * unitPair sp jj) • g +
          reduceMatrix Q{q} N 1 {m} (e jj) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
{haves}  subst hout hpayload
  obtain {pat} := hpar i
  have hpay := modulusReduce_poly hQ{L} hQ{q} hN {nm['pay']}
  have hone := modulusReduce_one hQ{L} hQ{q} hN {nm['one']}
  rw [multiplyMonomial_liftInteger] at {nm['msg']}
  have hnu := modulusReduce_poly hQ{L} hQ{q} hN {nm['msg']}
  have hgg' := gadgetMatrixRuns_unique {nm['gad']} hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique {nm['reb']} hg
  subst hgg
  obtain ⟨e0', hb0, he0⟩ := gaussian_bound_at hN {nm['e0']}
  obtain ⟨e1', hb1, he1⟩ := gaussian_bound_at hN {nm['e1']}
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, -, -, -, -, ho6, ho7, -⟩ := hout
  refine ⟨![e0', e1'], fun jj ↦ by fin_cases jj <;> assumption, fun jj ↦ ?_⟩
  have htop : ∀ jj, rhsTop model key Q{q} {m} rhsPrefix{j} i jj = ![{nm['w0']}, {nm['w1']}] jj := by
    intro jj
    fin_cases jj
    · exact (hashSample_eq {nm['top0']}).symm
    · exact (hashSample_eq {nm['top1']}).symm
  simp only [htop]
  fin_cases jj <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp (config := {{ zetaDelta := true }}) [ho6, ho7, matrixAdd, matrixSub, matrixMulScalarRight_smul,
      Matrix.mul_apply, unitPair, hb0, hb1, hpay, hone, hnu, reduceMatrix, matrixPolynomial_single,
      map_ofNat] <;>
    ring
''')
    # The switching ciphertexts.
    kinds = atom_kinds(path, 'parallel_generatedRoot_3')
    autos = [n for n, k in kinds.items() if k == 'MxxRuntime.ringAutomorphismRuns']
    gauss = [n for n, k in kinds.items() if k == 'MxxRuntime.gaussianSampleAt']
    reds = [n for n, k in kinds.items() if k == 'MxxRuntime.modulusReduceRuns']
    assert autos[0] == 'h5_0' and len(autos) == W + 1 and reds[0] == 'h4_0'
    keep = ['h4_0', 'h5_0'] + gauss + [reds[1]] + autos[1:] + ['hout']
    lines = destruct(path, 'parallel_generatedRoot_3', 'hcell', keep)
    haves = proj(path, 'generatedRoot', ['h3_3', 'hout'])
    want = f'(Fin {W} → Mxx.Primitives.ExactMatrix {I.Q[1]} {N} 1 1)'
    pos = [i for i, x in enumerate(outs) if x.strip() == want]
    assert pos == [0, 1, 2, 3], pos
    aut_lines = '\n'.join(f'  obtain ⟨_, a{k}⟩ := ringAutomorphismRuns_eq hQ1 hN {n}'
                          for k, n in enumerate(autos[1:]))
    a_names = ', '.join(f'a{k}' for k in range(W))
    a_names_sp = ' '.join(f'a{k}' for k in range(W))
    sig = ', '.join(f'({x})' for x in I.sigmas)
    parts.append(f'''
set_option maxHeartbeats 64000000 in
set_option linter.unusedSimpArgs false in
/-- The readout LUT's switching ciphertexts at `Q1`: `σ_i(s) B_{{0j}} + B_{{1j}} = (C_i ⌊Q1/2⌋ r_j) · 1 + e_j`. -/
theorem preprocess_switch {{model : HashModel}} {{key : ByteArray}} {{payload : ExactMatrix Q{L} N 1 1}}
    {{{binders} : Fin {C} → Int}} {{outputs}}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape path GeneratedClaim.stage_2_params
      ({ins}, ()) outputs)
    {{sp : ErrorPoly N}} (hpayload : payload = fun _ _ ↦ reducePoly Q{L} N sp)
    (i : Fin {W}) :
    ∃ e : Fin 2 → ErrorMatrix N 1 1, (∀ j, CoeffBound (e j) {I.cutoff}) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![outputs.1 i, outputs.2.1 i] j + ![outputs.2.2.1 i, outputs.2.2.2.1 i] j =
        (lutCoefficient Q1 inverse1 (sigma i).toNat *
          reducePoly Q1 N ((((Q1 / 2 : Nat) : Int) : ErrorPoly N) * unitPair sp j)) •
          (1 : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 (e j) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
{haves}  subst hout hpayload
  have hcell := h3_3 i
  unfold Stage_preprocess.parallel_generatedRoot_3 at hcell
{lines}  obtain ⟨_, hw5⟩ := ringAutomorphismRuns_eq hQ1 hN h5_0
  have hw4 := modulusReduce_poly hQ{L} hQ1 hN h4_0
  obtain ⟨e0', hb0, he0⟩ := gaussian_bound_at hN {gauss[0]}
  obtain ⟨e1', hb1, he1⟩ := gaussian_bound_at hN {gauss[1]}
  have hw{reds[1][1:-2]} := modulusReduce_one hQ{L} hQ1 hN {reds[1]}
{aut_lines}
  have hsig : MxxIR.selectInt (Int.ofNat ↑i) [{sig}] = sigma ↑i := rfl
  simp only [hsig] at hw5 {a_names_sp}
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, ho2, ho3, -⟩ := hout
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  rw [lutCoefficient_range (sigma_odd i)]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    obtain rfl : c = 0 := Subsingleton.elim _ _ <;>
    simp (config := {{ zetaDelta := true }}) only [reducePoly_substitution, Nat.succ_eq_add_one,
      Nat.reduceAdd, ho0, ho1, Fin.zero_eta, Fin.mk_one, Fin.isValue, Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.cons_val_fin_one, ho2, matrixAdd, matrixMulScalarLeft_one, matrixNeg, hw5,
      hw4, neg_mul, rotationPolynomial, Int.toNat_zero, pow_zero, map_one, Matrix.neg_apply,
      Int.reduceToNat, map_neg, substitution_root_pow, {a_names}, mul_one, hb0, ho3, hw{reds[1][1:-2]}, hb1,
      Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, Matrix.mul_apply, Finset.univ_unique,
      Fin.default_eq_zero, Finset.sum_singleton, matrixPolynomial_single, Int.cast_ofNat, Int.cast_one,
      Finset.sum_const, Finset.card_singleton, one_smul, mul_neg, one_mul, Finset.sum_neg_distrib,
      reduceMatrix, Nat.cast_ofNat, MxxExponentLutLwr.readout, Nat.cast_add, Nat.cast_one, mul_ite,
      Finset.sum_range_succ, Finset.range_one, ↓reduceIte, mul_zero, Nat.zero_div, Nat.cast_zero, zero_add,
      one_ne_zero, Nat.reduceDiv, Nat.reduceSub, OfNat.ofNat_ne_zero, Nat.reduceMul, Nat.ofNat_pos,
      Nat.div_self, Int.natCast_ediv, Int.reduceDiv, unitPair, reducePoly_mul, map_ofNat,
      Matrix.one_apply_eq] <;>
    ring

end LwrProof
''')
    return ''.join(parts)
