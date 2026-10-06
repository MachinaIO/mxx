"""LwrCertificate.lean: the average-case correctness certificate."""
from info import wrap
from lean_patterns import proj


def cert(I):
    L, B, C, M, N, W, s = I.L, I.B, I.C, I.M, I.n, I.W, I.stride
    pp = I.pprime
    T = pp * (W - 1) // W + 1
    TT = 4 * pp
    cut = I.cutoff
    m = I.m
    sels = ' '.join(f's{b}' for b in range(B))
    imports = ''.join(f'import LwrBlock{b}\n' for b in range(B))
    o = [imports + f'''import LwrLut
import LwrEncrypt
import LwrPreprocess
import LwrInputs

/-!
# The average-case correctness certificate

Every execution on every sample tape satisfying the generated claim's run relation decodes the
half-integer readout of its selected coordinates modulo `{TT}`. The requirement makes the plaintext of member
`k` of `a` the monomial `X^({s} a_k)` and every block's selectors one-hot; the {B} blocks then
accumulate `X^({s}z)` for the sum `z` of the selected coordinates, the readout LUT gives
`(-1)^⌊z/{W}⌋ (2 τ(z mod {W}) - 1)` at the terminal modulus, and decryption returns it modulo `{TT}`,
which is the ideal output. The plaintexts of the members of `a` and of the state are monomials, so multiplying an
error by them keeps its bound (`SmulGain _ 1`).
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxRecDepth 100000

variable {{tape : SampleTape}} {{path : List Nat}}

/-! ## Plaintexts of the run -/

/-- A one-hot family selects one cell. -/
theorem one_hot {{s : Fin {C} → Int}} (h01 : ∀ i, 0 ≤ s i ∧ s i ≤ 1) (hsum : ∑ i, s i = 1) :
    ∃ sel : Fin {C}, ∀ i, s i = if i = sel then 1 else 0 := by
  have hex : ∃ sel, s sel = 1 := by
    by_contra hne
    push_neg at hne
    have hzero : ∀ i, s i = 0 := fun i ↦ by have := h01 i; have := hne i; omega
    simp [hzero] at hsum
  obtain ⟨sel, hsel⟩ := hex
  refine ⟨sel, fun i ↦ ?_⟩
  split
  · rename_i hi
    rw [hi, hsel]
  · rename_i hi
    by_contra hne
    have hi1 : s i = 1 := by have := h01 i; omega
    have hpair : ∑ j ∈ ({{i, sel}} : Finset (Fin {C})), s j ≤ ∑ j, s j :=
      Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
        (fun j _ _ ↦ (h01 j).1)
    rw [Finset.sum_pair hi, hi1, hsel, hsum] at hpair
    omega

/-- The block sum with one-hot selectors is the selected cell. -/
theorem collapse {{q m p : Nat}} (h : ExactMatrix q N 1 m) (sp μ : ErrorPoly N)
    (α ν : Fin {C} → ErrorPoly N) (s : Fin {C} → Int) (sel : Fin {C})
    (hs : ∀ i, s i = if i = sel then 1 else 0) (hν : ∀ i, ν i = (s i : ErrorPoly N)) :
    (∑ i : Fin {C}, fun j ↦ reducePoly q N (((p : Nat) : Int) * (α i * μ * ν i) * unitPair sp j) • h) =
      fun j ↦ reducePoly q N (((p : Nat) : Int) * (α sel * μ) * unitPair sp j) • h := by
  funext j
  rw [Finset.sum_apply, Finset.sum_eq_single sel]
  · simp [hν, hs]
  · intro i _ hi
    simp [hν, hs, hi]
  · simp

/-- A one-hot block selects one coordinate. -/
theorem one_hot_sum (a : Fin {M} → Int) (b : Fin {B}) {{s : Fin {C} → Int}} {{sel : Fin {C}}}
    (hs : ∀ i, s i = if i = sel then 1 else 0) :
    ∑ i : Fin {C}, s i * a ⟨coords b i, coords_lt b i⟩ = a ⟨coords b sel, coords_lt b sel⟩ := by
  rw [Finset.sum_eq_single sel]
  · simp [hs]
  · intro i _ hi
    simp [hs, hi]
  · simp

/-- The plaintext of a member of `a` accepted by the requirement is `X^({s}a)`. -/
theorem message_member {{c e a : Int}} (hc : c * c = 1) (hrel : e * 2 + (1 - c) * {N} = a * {2 * s})
    (he : 0 ≤ e ∧ e ≤ {N - 1}) : message (n := N) c e = Xe ({s} * a.toNat) := by
  have hc' : c = 1 ∨ c = -1 := Int.eq_one_or_neg_one_of_mul_eq_one hc
  rcases hc' with rfl | rfl
  · exact message_one rfl (by omega) (by simp only [N]; omega)
  · rw [message, show ((2 * N : Nat) : Int) = {2 * N} from rfl, Int.emod_eq_of_lt (by omega) (by omega)]
    have hsplit : {s} * a.toNat = e.toNat + {N} := by omega
    rw [Xe, hsplit, pow_add, Negacyclic.root_pow_n]
    push_cast
    ring

/-- A selector plaintext with exponent zero is the bit. -/
theorem message_selector {{s x : Int}} (hx : x = 0) : message (n := N) s x = (s : ErrorPoly N) := by
  subst hx
  simp [message]

theorem Xe_mul (a b : Nat) : Xe (n := N) a * Xe b = Xe (a + b) := by
  rw [Xe, Xe, Xe, pow_add]

theorem intCast_coeff_zero (x : Int) : ((x : ErrorPoly N)).coeff ⟨0, hN⟩ = x := by
  have h := intCast_mul_coeff x (1 : ErrorPoly N) ⟨0, hN⟩
  rw [mul_one] at h
  have h1 := Negacyclic.coeff_root_pow (R := Int) (n := N) hN ⟨0, hN⟩ ⟨0, hN⟩
  rw [show ((⟨0, hN⟩ : Fin N) : Nat) = 0 from rfl, pow_zero, if_pos rfl] at h1
  rw [h, h1, mul_one]

/-- The half-integer readout modulo `{TT}` is the ideal value: `m` for an even `⌊z / {W}⌋` and
`{TT} - m` for an odd one, with `m = 2 τ(z mod {W}) - 1`. -/
theorem readout_mod{TT} (z : Nat) :
    ((lutValue z).coeff ⟨0, hN⟩) % {TT} =
      ((((z : Int) % {W}) * {pp} / {W} + 1) * 2 - 1) +
        (z : Int) % {2 * W} / {W} * ({TT} - ((((z : Int) % {W}) * {pp} / {W} + 1) * 2 - 1) * 2) := by
  rw [lutValue, intCast_coeff_zero, MxxExponentLutLwr.halfIntegerReadout, MxxExponentLutLwr.readout]
  have hτ : ((2 * ({pp} * (z % {W}) / {W} + 1) - 1 : Nat) : Int) =
      (((z : Int) % {W}) * {pp} / {W} + 1) * 2 - 1 := by omega
  rw [hτ]
  have hb : (z : Int) % {2 * W} / {W} = ((z / {W} : Nat) : Int) % 2 := by omega
  rw [hb]
  rcases Nat.even_or_odd (z / {W}) with h | h
  · rw [h.neg_one_pow, one_mul, show ((z / {W} : Nat) : Int) % 2 = 0 by
      obtain ⟨k, hk⟩ := h; omega]
    omega
  · rw [h.neg_one_pow, neg_one_mul, show ((z / {W} : Nat) : Int) % 2 = 1 by
      obtain ⟨k, hk⟩ := h; omega]
    omega

theorem lutValue_coeff_bound (z : Nat) : |(lutValue z).coeff ⟨0, hN⟩| ≤ {2 * pp - 1} := by
  rw [lutValue, intCast_coeff_zero, MxxExponentLutLwr.halfIntegerReadout, MxxExponentLutLwr.readout]
  have hτ : 2 * ({pp} * (z % {W}) / {W} + 1) - 1 ≤ {2 * pp - 1} := by omega
  rcases Nat.even_or_odd (z / {W}) with h | h
  · rw [h.neg_one_pow, one_mul, abs_le]; omega
  · rw [h.neg_one_pow, neg_one_mul, abs_le]; omega

theorem secret_ternary {{out : ExactMatrix Q{L} N 1 1}}
    (h : Stage_payload.generatedRoot tape path GeneratedClaim.stage_0_params () out) :
    ∃ s : ErrorPoly N, out = (fun _ _ ↦ reducePoly Q{L} N s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly_at hN hw
  exact ⟨s, hout.trans hs, hn⟩

theorem mask_ternary {{out : ExactMatrix Q{L} N 1 1}}
    (h : Stage_mask.generatedRoot tape path GeneratedClaim.stage_1_params () out) :
    ∃ s : ErrorPoly N, out = (fun _ _ ↦ reducePoly Q{L} N s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly_at hN hw
  exact ⟨s, hout.trans hs, hn⟩
''']
    if I.h2:
        # The certificate goes through H2 (`LwrH2Cert`); this module keeps the lemmas on the plaintexts.
        o.append('\nend LwrProof\n')
        return ''.join(o)
    # The correctness of every run.
    valid = wrap(['-', '⟨hc1, hc2⟩', '⟨he1, he2⟩', 'ha', 'hac', 'hae'] +
                 [f'hs{b}, hx{b}' for b in range(B)], ', ', 4, 10)
    sums = wrap([f'hsum{b}' for b in range(B)], ', ', 4, 16)
    sets = '\n'.join([f'  set a := external.input_3'] +
                     [f'  set s{b} := external.input_{6 + 2 * b}' for b in range(B)] +
                     [f'  set x{b} := external.input_{7 + 2 * b}' for b in range(B)])
    onehots = '\n'.join(f'  obtain ⟨sel{b}, hsel{b}⟩ := one_hot hs{b} hsum{b}' for b in range(B))
    nus = '\n'.join(f'''  have hν{b} : ∀ i, message (n := N) (s{b} i) (x{b} i) = (s{b} i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx{b} i; omega)''' for b in range(B))
    gadgets = '\n'.join(f'  obtain ⟨g{m(k)}, hg{m(k)}⟩ := gadget_exists (m := {m(k)}) layout{k} rfl rfl rfl'
                        for k in range(L, 0, -1))
    rebases = '\n'.join(f'  have hg{m(k)}u : centeredRebaseRuns g{m(k)} (centeredRebase g{m(k)} : ExactMatrix Q{k + 1} N 1 {m(k)}) := ⟨hQ{k + 1}, rfl⟩'
                        for k in range(L - 1, 0, -1))
    rhs = '\n'.join(f'  have hrhs{b} := preprocess_selector{b} hpre hsp hg{m(L - 2 - b)} hg{m(L - 2 - b)}u' for b in range(B))
    blocks = []
    for b in range(B):
        qn = L - 1 - b
        mu = 'Xe 0' if b == 0 else f'Xe ({s} * z{b})'
        hs = 'hstate' if b == 0 else f'hin{b}'
        prev = '0' if b == 0 else f'{s} * z{b}'
        prevz = '0' if b == 0 else f'z{b}'
        blocks.append(f'''  -- Block {b}.
  have hb{b} := block{b} hb hsmn (μ := {mu}) (α := α)
    (ν := fun i ↦ message (n := N) (s{b} i) (x{b} i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg{m(qn)} hg{m(qn)}u hmemα hpayα {hs} hrhs{b}
  have hin{b + 1} := hb{b}.target_eq (collapse _ _ _ _ _ _ sel{b} hsel{b} hν{b})
  rw [Xe_mul] at hin{b + 1}
  obtain ⟨z{b + 1}, hz{b + 1}⟩ : ∃ z : Nat,
      z = (a ⟨coords {b} sel{b}, coords_lt {b} sel{b}⟩).toNat + {prevz} := ⟨_, rfl⟩
  rw [show {s} * (a ⟨coords {b} sel{b}, coords_lt {b} sel{b}⟩).toNat + {prev} = {s} * z{b + 1} by
    rw [hz{b + 1}]; ring] at hin{b + 1}
''')
    hout = proj(f'{I.d}/generated/Stage_online_eval.lean', 'generatedRoot', ['hout'], 'hb2').replace('have hout', 'have houtOnlineEval')
    bound = f'cellBound0 {cut}'
    for b in range(1, B):
        bound = f'cellBound{b} (∑ _i : Fin {C}, {bound})'
    bound = f'lutBound (∑ _i : Fin {C}, {bound})'
    hz = wrap([f'one_hot_sum a {b} hsel{b}' for b in range(B)] + [f'hz{b}' for b in range(B, 0, -1)],
              ', ', 6, 22)
    hzs = None
    toNats = wrap(['Int.toNat_of_nonneg (ha _).1'] * B, ', ', 6, 9)
    ideal_args = wrap(sels.split(' '), ' ', 6, 30)
    o.append(f'''
set_option maxHeartbeats 0 in
/-- Every run decodes the parity of the readout of its selected coordinates. -/
theorem correct {{model : HashModel}} {{external : GeneratedClaim.ExternalInputs}}
    {{execution : GeneratedClaim.Execution}}
    (hruns : GeneratedClaim.Runs model external tape execution) :
    execution.«stage_5» = execution.«ideal» := by
  obtain ⟨hvalid, hpay, hmask, hpre, henc, honlineEval, hdec, hreq, hideal, hreqtrue⟩ := hruns
  obtain ⟨sp, hsp, hspn⟩ := secret_ternary hpay
  obtain ⟨sm, hsm, hsmn⟩ := mask_ternary hmask
  obtain ⟨{valid}⟩ := hvalid
  obtain ⟨hmem, {sums}⟩ := requirement_holds hreq hreqtrue
  have hid := ideal_value hideal
{sets}
  -- The selected cells and the plaintexts.
{onehots}
  let α : Fin {M} → ErrorPoly N := fun k ↦ Xe ({s} * (a k).toNat)
  have hα : ∀ k, message (n := N) (external.input_4 k) (external.input_5 k) = α k := fun k ↦
    message_member (hmem k).1 (hmem k).2 (hae k)
{nus}
  have hμ0 : message (n := N) external.input_1 external.input_2 = Xe 0 :=
    message_one (k := 0) (by omega) (by simp; omega) (by simp only [N]; omega)
  -- The gadget matrices.
{gadgets}
{rebases}
  have hG := hg{m(L)}
  -- Encryption and preprocessing.
  have hstate := encrypt_state henc hsm hsp hg{m(L - 1)} hg{m(L - 1)}u
  rw [hμ0] at hstate
  have hmembers := encrypt_member henc hsm hG
  have hmemα : ∀ k : Fin {M}, Rel sm (memberKey model external.input_0 k)
      (fun j ↦ reducePoly Q{L} N (α k * unitPair sm j) • g{m(L)})
      ![execution.«stage_3».2.2.1 k, execution.«stage_3».2.2.2.1 k] {cut} := fun k ↦ by
    rw [← hα k]; exact (hmembers k).1
  have hpayα : ∀ k : Fin {M}, execution.«stage_3».2.2.2.2.1 k = fun _ _ ↦ reducePoly Q{L} N (α k) :=
    fun k ↦ by rw [← hα k]; exact (hmembers k).2
  have hhelp := encrypt_helper henc hsm hg{m(1)} hg{m(1)}u
{rhs}
  have hsw := preprocess_switch hpre hsp
  -- The {B} blocks accumulate the monomial of the sum of the selected coordinates.
  obtain ⟨w, hb⟩ := honlineEval
{''.join(blocks)}  -- The readout LUT and the decryption.
  have hlut := lutRound hb hsmn hg{m(1)} hg{m(1)}u hin{B} hhelp hsw
  have hb2 := hb
  unfold Stage_online_eval.generatedRoot.body at hb2
  extract_lets at hb2
{hout}  obtain ⟨wd, hd⟩ := hdec
  unfold Stage_decode.generatedRoot.body at hd
  dsimp only at hd
  obtain ⟨hm1, hm2, -, hdec1, hout⟩ := hd
  rw [hsm] at hm1
  have hw2 := modulusReduce_poly hQ{L} hQ1 hN hm1
  have hw6 := modulusReduce_one hQ{L} hQ1 hN hm2
  have hbound : 4 * {bound} + 2 * {T} + 12 < Q1 := by
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
    decide
  have hval := decode_terminal_parity_bounded hQ1 (by norm_num) hN hlut hbound (k := ⟨0, hN⟩)
    (lutValue_coeff_bound z{B}) (by
      rw [houtOnlineEval, hw2, hw6]
      funext r c
      obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _
      simp only [matrixSub, matrixAdd, matrixMulScalarLeft_eq, Matrix.sub_apply, Matrix.add_apply,
        Matrix.smul_apply, smul_eq_mul, Matrix.one_apply_eq, one_mul]
      rfl) hdec1
  rw [hout, hval, readout_parity, hid]
  have hz : selectedSum a {ideal_args} = (z{B} : Int) := by
    rw [selectedSum, {hz}]
    push_cast
    rw [{toNats}]
    ring
  rw [hz]

/-- The average-case correctness claim of `{I.row["parameter_id"]}`: the failure set is empty, so its
tape measure is below every bound. -/
theorem certificate : GeneratedClaim.CorrectnessClaim := by
  intro model external
  have hempty : {{tape | ∃ execution, GeneratedClaim.Runs model external tape execution ∧
      ¬ execution.«stage_5» = execution.«ideal»}} = ∅ := by
    ext tape
    simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
    rintro ⟨execution, hruns, hne⟩
    exact hne (correct hruns)
  rw [hempty, MeasureTheory.measure_empty]
  exact zero_le _

/-- info: 'LwrProof.certificate' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms certificate

end LwrProof
''')
    return ''.join(o)
