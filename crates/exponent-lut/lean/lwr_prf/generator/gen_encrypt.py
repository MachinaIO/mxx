"""LwrEncrypt.lean: the state input, the members of `a`, and the readout LUT helpers."""
import re
from lean_patterns import pattern


def split_top(s):
    """Split a product type at its top-level `×`."""
    parts, depth, cur = [], 0, ''
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == '×' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    parts.append(cur)
    return parts


def tup(k):
    return 'outputs' + '.2' * k + '.1'


def encrypt(I):
    L, M, W, B = I.L, I.M, I.W, I.B
    mL, mL1, m1, base = I.m(L), I.m(L - 1), I.m(1), I.base
    rp = pattern(f'{I.d}/generated/Stage_encrypt.lean', 'generatedRoot')[1:-1].split(', ')
    assert rp[:8] == ['h2_0', 'h5_0', 'h9_0', 'h10_0', 'h20_0', 'h22_0', 'h24_0', 'h31_0'], rp[:8]
    assert rp[9] == 'h35_2' and rp[-1] == 'hout'
    k = len(rp)
    state_pat = ', '.join(['h2', 'h5', 'h9', 'h10', 'h20', 'h22', 'h24', 'h31'] + ['-'] * (k - 9) + ['hout'])
    member_pat = ', '.join(['-'] * 9 + ['hloop'] + ['-'] * (k - 11) + ['hout'])
    # The level-B helpers: the encryption loop and the output pair at `Q2` with width `m1`.
    enc = I.enc
    root = enc[enc.index('def generatedRoot '):]
    outs = root[root.index('(outputs : ') + 11:root.index(') : Prop')]
    parts = split_top(outs)
    want = f'(Fin {W} → Mxx.Primitives.ExactMatrix {I.Q[2]} {I.n} 1 {m1})'
    pos = [i for i, x in enumerate(parts) if x.strip() == want]
    assert len(pos) == 2 and pos[1] == pos[0] + 1, pos
    hk = pos[0]
    loops = [f'parallel_generatedRoot_{x}' for x in range(36, 36 + B + 1)]
    last = next(l for l in loops if re.search(rf'def {l} .*\(outputs : Mxx\.Primitives\.ExactMatrix {I.Q[2]} ', enc))
    li = rp.index(f'h{last.rsplit("_", 1)[1]}_1')
    helper_pat = ', '.join(['-'] * li + ['hloop'] + ['-'] * (k - li - 2) + ['hout'])
    hp = pattern(f'{I.d}/generated/Stage_encrypt.lean', last)
    assert hp == ('⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, '
                  'w_29_0, h1_0, -, -, h4_0, h7_0, h8_0, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, '
                  '-, -, h25_0, h29_0, hout⟩'), hp
    return f'''import LwrSetup

/-!
# The encryption stage

The state input is encrypted at `Q{L}` with scale `P{L}` against the gadget of `Q{L - 1}`; every member of
the public vector `a` is a public encoding at `Q{L}` with the native gadget, whose payload is the
mask, together with its public payload; and the readout LUT's helpers are encryptions at `Q2` of
`P2 σ_i(s)` against the gadget of `Q1`.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxRecDepth 100000

variable {{tape : SampleTape}} {{path : List Nat}}

set_option maxHeartbeats 4000000 in
/-- The state input is an encoding of `c X^e` at `Q{L}` with scale `P{L}`. -/
theorem encrypt_state {{model : HashModel}} {{mask payload : ExactMatrix Q{L} N 1 1}} {{key : ByteArray}}
    {{coefficient exponent : Int}} {{ac ae : Fin {M} → Int}} {{outputs}}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_3_params
      (mask, key, payload, coefficient, exponent, ac, ae, ()) outputs)
    {{sm sp : ErrorPoly N}} (hmask : mask = fun _ _ ↦ reducePoly Q{L} N sm)
    (hpayload : payload = fun _ _ ↦ reducePoly Q{L} N sp)
    {{g' : ExactMatrix Q{L - 1} N 1 {mL1}}} {{g : ExactMatrix Q{L} N 1 {mL1}}}
    (hg' : gadgetMatrixRuns Backend.backend {base} {mL1} g') (hg : centeredRebaseRuns g' g) :
    Rel sm (stateKey model key)
      (fun j ↦ reducePoly Q{L} N (((P{L} : Nat) : Int) * message (n := N) coefficient exponent * unitPair sp j) • g)
      ![outputs.1, outputs.2.1] {I.cutoff} := by
  obtain ⟨w, {state_pat}⟩ := h
  have hgg' := gadgetMatrixRuns_unique h9 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h10 hg
  subst hgg
  obtain ⟨e0, hb0, he0⟩ := gaussian_bound_at hN h20
  obtain ⟨e1, hb1, he1⟩ := gaussian_bound_at hN h31
  subst hout hmask hpayload
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, stateKey model key k j = ![![w.w_2_0, w.w_22_0], ![w.w_5_0, w.w_24_0]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [stateKey, statePrefix, componentBytes, hashSample_eq h2, hashSample_eq h5,
        hashSample_eq h22, hashSample_eq h24]
  simp only [hkey]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [matrixAdd, matrixSub, matrixMulScalarRight_smul, Matrix.mul_apply,
      multiplyMonomial_liftInteger, unitPair, hb0, hb1, reduceMatrix, matrixPolynomial_single,
      map_ofNat]
     try ring)

set_option maxHeartbeats 4000000 in
/-- Member `i` of `a` is a public encoding of `c_i X^(e_i)` at `Q{L}` with the native gadget, and
its public payload is that plaintext. -/
theorem encrypt_member {{model : HashModel}} {{mask payload : ExactMatrix Q{L} N 1 1}} {{key : ByteArray}}
    {{coefficient exponent : Int}} {{ac ae : Fin {M} → Int}} {{outputs}}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_3_params
      (mask, key, payload, coefficient, exponent, ac, ae, ()) outputs)
    {{sm : ErrorPoly N}} (hmask : mask = fun _ _ ↦ reducePoly Q{L} N sm)
    {{G : ExactMatrix Q{L} N 1 {mL}}} (hG : gadgetMatrixRuns Backend.backend {base} {mL} G)
    (i : Fin {M}) :
    Rel sm (memberKey model key i)
      (fun j ↦ reducePoly Q{L} N (message (n := N) (ac i) (ae i) * unitPair sm j) • G)
      ![outputs.2.2.1 i, outputs.2.2.2.1 i] {I.cutoff} ∧
      outputs.2.2.2.2.1 i = fun _ _ ↦ reducePoly Q{L} N (message (n := N) (ac i) (ae i)) := by
  obtain ⟨w, {member_pat}⟩ := h
  subst hout hmask
  obtain ⟨w2, w5, w12, w16, w18, w20, w25, h2, h5, h12, h16, h18, h20, h25, hout⟩ := hloop i
  have hgg := gadgetMatrixRuns_unique h12 hG
  subst hgg
  obtain ⟨e0, hb0, he0⟩ := gaussian_bound_at hN h16
  obtain ⟨e1, hb1, he1⟩ := gaussian_bound_at hN h25
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, ho2, -⟩ := hout
  refine ⟨⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩, ?_⟩
  · have hkey : ∀ k j, memberKey model key i k j = ![![w2, w18], ![w5, w20]] k j := by
      intro k j
      fin_cases k <;> fin_cases j <;>
        simp [memberKey, memberPrefix, memberBytes, componentBytes, hashSample_eq h2,
          hashSample_eq h5, hashSample_eq h18, hashSample_eq h20]
    simp only [hkey]
    fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
      (simp [matrixAdd, matrixSub, Matrix.mul_apply, multiplyMonomial_liftInteger, unitPair,
        hb0, hb1, reduceMatrix, ho0, ho1]
       try ring)
  · simp [ho2, multiplyMonomial_liftInteger]

set_option maxHeartbeats 4000000 in
set_option linter.unusedSimpArgs false in
/-- The level-{B} helpers: encryptions at `Q2` of `P2 σ_i(s)` against the gadget of `Q1`. -/
theorem encrypt_helper {{model : HashModel}} {{mask payload : ExactMatrix Q{L} N 1 1}}
    {{key : ByteArray}} {{coefficient exponent : Int}} {{ac ae : Fin {M} → Int}} {{outputs}}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_3_params
      (mask, key, payload, coefficient, exponent, ac, ae, ()) outputs)
    {{sm : ErrorPoly N}} (hmask : mask = fun _ _ ↦ reducePoly Q{L} N sm)
    {{g' : ExactMatrix Q1 N 1 {m1}}} {{g : ExactMatrix Q2 N 1 {m1}}}
    (hg' : gadgetMatrixRuns Backend.backend {base} {m1} g') (hg : centeredRebaseRuns g' g)
    (i : Fin {W}) :
    Rel sm (lutA model key Q2 {m1} {B} (sigma i))
      (fun j ↦ reducePoly Q2 N (((P2 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g)
      ![{tup(hk)} i, {tup(hk + 1)} i] {I.cutoff} := by
  obtain ⟨w, {helper_pat}⟩ := h
  subst hout hmask
  obtain ⟨w1, w4, w7, w8, w12, w14, w15, w18, w20, w22, w25, w29, h1, -, -, h4, h7, h8, -, -, h12,
    h14, h15, h18, h20, h22, -, -, h25, h29, hout⟩ := hloop i
  have hw1 := modulusReduce_poly hQ{L} hQ2 hN h1
  have hw7 := modulusReduce_one hQ{L} hQ2 hN h7
  subst hw1 hw7
  have hgg' := gadgetMatrixRuns_unique h14 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h15 hg
  subst hgg
  obtain ⟨e0, hb0, he0⟩ := gaussian_bound_at hN h18
  obtain ⟨e1, hb1, he1⟩ := gaussian_bound_at hN h29
  obtain ⟨_, hw12⟩ := ringAutomorphismRuns_eq hQ2 hN h12
  obtain ⟨_, hw25⟩ := ringAutomorphismRuns_eq hQ2 hN h25
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, -⟩ := hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : lutA model key Q2 {m1} {B} (sigma i) = ![![w4, w20], ![w8, w22]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4).symm
    · exact (hashSample_eq h20).symm
    · exact (hashSample_eq h8).symm
    · exact (hashSample_eq h22).symm
  simp only [hkey]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp only [ho0, matrixAdd, matrixSub, matrixMulScalarLeft_one, Matrix.one_mul, hw12,
      hb0, ho1, hw25, hb1, Fin.zero_eta, Fin.mk_one, Fin.isValue, Matrix.cons_val_zero, Matrix.add_apply,
      Matrix.sub_apply, Matrix.mul_apply, Finset.univ_unique, Fin.default_eq_zero, Finset.sum_singleton,
      matrixPolynomial_single, Int.cast_ofNat, Finset.sum_const, Finset.card_singleton, one_smul, reduceMatrix,
      Nat.succ_eq_add_one, Nat.reduceAdd, Matrix.cons_val', Matrix.cons_val_fin_one, Matrix.cons_val_one,
      Nat.cast_ofNat, unitPair, reducePoly_mul, map_ofNat, reducePoly_substitution, Matrix.smul_apply,
      smul_eq_mul, Matrix.one_apply_eq, map_one, mul_one]

end LwrProof
'''
