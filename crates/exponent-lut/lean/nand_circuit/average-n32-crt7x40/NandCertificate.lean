import NandOnline3

/-!
# The average-case correctness certificate

Every execution on every sample tape satisfying the generated claim's run relation decodes the NAND
of its two input bits, so the failure set of the claim is empty. The tape samplers record their
cutoffs (`ExponentSampling`); the payload and mask stages give ternary secrets; the encryption and
preprocessing stages give the encodings, helpers, RHS ciphertexts, and switching ciphertexts of `NandEncrypt` and
`NandPreprocess`; the four online rounds carry the state monomial `X^(roundArgument b r)` through
the LUT tables; and the decode stage reads coefficient `1` of the terminal plaintext
`X^(outputExponent b)`, which is `1 - b₀ b₁`.
-/

namespace NandProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxExponentLutNand

variable {tape : SampleTape} {path loopPath : List Nat}

/-- The gadget matrix of a regular layout with base `2^20`. -/
theorem gadget_exists {q m : Nat} {layout : RegularLayout q}
    (hlayout : Backend.backend.regularLayout q N = some layout)
    (hbase : (1048576 : Int) = layout.base) (hdigits : (m : Int) = layout.digitCount)
    (hwidth : m = 1 * layout.digitCount) :
    ∃ g : ExactMatrix q N 1 m, gadgetMatrixRuns Backend.backend 1048576 m g :=
  ⟨_, layout, hlayout, hbase, hdigits, hwidth, rfl⟩

/-- A plaintext `1 · X^e` with `0 ≤ e < 64` is the monomial `X^e`. -/
theorem message_one {c e : Int} {k : Nat} (hc : c = 1) (he : e = k) (hk : k < 64) :
    message c e = Xe k := by
  subst hc he
  rw [message, Int.cast_one, one_mul, Int.emod_eq_of_lt (by omega) (by omega), Int.toNat_natCast]

/-- The requirement stage accepts exactly the exponents `8 b`. -/
theorem requirement_holds {a b c d : Int} {out : Bool}
    (h : Requirement_0.generatedRoot tape path GeneratedClaim.requirement_0_params (a, b, c, d, ()) out)
    (htrue : out = true) : a = b * 8 ∧ c = d * 8 := by
  obtain ⟨_, h⟩ := h
  simp only [Requirement_0.generatedRoot.body, Requirement_0.generatedRoot.constraints_0] at h
  subst htrue
  by_cases h1 : a = b * 8 <;> by_cases h2 : c = d * 8
  · exact ⟨h1, h2⟩
  all_goals simp [h1, h2] at h

/-- The ideal stage computes `1 - b₀ b₁`. -/
theorem ideal_value {a b out : Int}
    (h : Ideal.generatedRoot tape path GeneratedClaim.ideal_params (a, b, ()) out) : out = 1 - a * b := by
  obtain ⟨_, h⟩ := h
  simpa [Ideal.generatedRoot.body, Ideal.generatedRoot.constraints_0] using h

/-- A ternary secret sampled by the payload or mask stage. -/
theorem secret_ternary {out : ExactMatrix Q7 N 1 1}
    (h : Stage_payload.generatedRoot tape path GeneratedClaim.stage_0_params () out) :
    ∃ s : ErrorPoly N, out = (fun _ _ ↦ reducePoly Q7 N s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly hw
  exact ⟨s, hout.trans hs, hn⟩

theorem mask_ternary {out : ExactMatrix Q7 N 1 1}
    (h : Stage_mask.generatedRoot tape path GeneratedClaim.stage_1_params () out) :
    ∃ s : ErrorPoly N, out = (fun _ _ ↦ reducePoly Q7 N s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly hw
  exact ⟨s, hout.trans hs, hn⟩

set_option maxHeartbeats 64000000 in
/-- Every run on every tape decodes the ideal output. -/
theorem correct {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (hruns : GeneratedClaim.Runs model external tape execution) :
    execution.«stage_5» = execution.«ideal» := by
  obtain ⟨hvalid, hpay, hmask, hpre, henc, honline, hdec, hreq, hideal, hreqtrue⟩ := hruns
  obtain ⟨sp, hsp, hspn⟩ := secret_ternary hpay
  obtain ⟨sm, hsm, hsmn⟩ := mask_ternary hmask
  -- The external inputs: coefficient `1`, start exponent `2`, bits `b₀ b₁`, exponents `8 bᵢ`.
  obtain ⟨-, ⟨hc1, hc2⟩, ⟨he1, he2⟩, ⟨hb0, hb0'⟩, -, ⟨hb1, hb1'⟩, -⟩ := hvalid
  obtain ⟨hx0, hx1⟩ := requirement_holds hreq hreqtrue
  have hideal' := ideal_value hideal
  obtain ⟨n0, hn0⟩ : ∃ n : Nat, external.input_3 = n := ⟨external.input_3.toNat, by omega⟩
  obtain ⟨n1, hn1⟩ : ∃ n : Nat, external.input_5 = n := ⟨external.input_5.toNat, by omega⟩
  have hn0' : n0 ≤ 1 := by omega
  have hn1' : n1 ≤ 1 := by omega
  have hmsg0 : message external.input_1 external.input_2 = Xe 2 :=
    message_one (by omega) (by omega) (by norm_num)
  have hmsg4 : message external.input_1 external.input_4 = Xe (8 * n0) :=
    message_one (by omega) (by omega) (by omega)
  have hmsg6 : message external.input_1 external.input_6 = Xe (8 * n1) :=
    message_one (by omega) (by omega) (by omega)
  -- The gadget matrices of levels `0` through `7`.
  obtain ⟨g14, hg14⟩ := gadget_exists (m := 14) layout7 rfl rfl rfl
  obtain ⟨g12, hg12⟩ := gadget_exists (m := 12) layout6 rfl rfl rfl
  obtain ⟨g10, hg10⟩ := gadget_exists (m := 10) layout5 rfl rfl rfl
  obtain ⟨g8, hg8⟩ := gadget_exists (m := 8) layout4 rfl rfl rfl
  obtain ⟨g6, hg6⟩ := gadget_exists (m := 6) layout3 rfl rfl rfl
  have hg12up : centeredRebaseRuns g12 (centeredRebase g12 : ExactMatrix Q7 N 1 12) := ⟨hQ7, rfl⟩
  have hg10up : centeredRebaseRuns g10 (centeredRebase g10 : ExactMatrix Q6 N 1 10) := ⟨hQ6, rfl⟩
  have hg8up : centeredRebaseRuns g8 (centeredRebase g8 : ExactMatrix Q5 N 1 8) := ⟨hQ5, rfl⟩
  have hg6up : centeredRebaseRuns g6 (centeredRebase g6 : ExactMatrix Q4 N 1 6) := ⟨hQ4, rfl⟩
  -- Encryption and preprocessing.
  have hinput := encrypt_input henc hsm hsp hg14 (centeredRebaseRuns_self hQ7 hN g14)
  have hhelp1 := encrypt_helper1 henc hsm hg12 hg12up
  have hhelp3 := encrypt_helper3 henc hsm hg10 hg10up
  have hhelp5 := encrypt_helper5 henc hsm hg8 hg8up
  have hhelp7 := encrypt_helper7 henc hsm hg6 hg6up
  have hrhs0 := preprocess_rhs0 hpre hsp hg12 hg12up
  have hrhs1 := preprocess_rhs1 hpre hsp hg10 hg10up
  have hrhs2 := preprocess_rhs2 hpre hsp hg8 hg8up
  have hrhs3 := preprocess_rhs3 hpre hsp hg6 hg6up
  have hsw0 := preprocess_switch0 hpre hsp hg12 (centeredRebaseRuns_self hQ6 hN g12)
  have hsw1 := preprocess_switch1 hpre hsp hg10 (centeredRebaseRuns_self hQ5 hN g10)
  have hsw2 := preprocess_switch2 hpre hsp hg8 (centeredRebaseRuns_self hQ4 hN g8)
  have hsw3 := preprocess_switch3 hpre hsp
  rw [hmsg0] at hinput
  rw [hmsg4] at hrhs0 hrhs2
  rw [hmsg6] at hrhs1 hrhs3
  -- The four online rounds on the arguments `roundArgument b r`.
  have hrounds := rounds_nand ![n0, n1] (fun i ↦ by fin_cases i <;> simpa)
  obtain ⟨w, hb⟩ := honline
  have hr0 := round0 hb hsmn (polyNorm_Xe 2) hg14 (centeredRebaseRuns_self hQ7 hN g14) hg12 hg12up
    hinput hrhs0 hhelp1 hsw0 (arg := ⟨roundArgument ![n0, n1] 0, hrounds.1 0⟩) (by
      rw [Xe, Xe, Xe, ← pow_add]
      congr 1
      have e : roundArgument ![n0, n1] 0 = 2 + 8 * n0 := by
        simp [roundArgument, roundInput, initialExponent]
      exact e.symm)
  have hr1 := round1 hb hsmn (polyNorm_Xe _) hg12 hg10 hg10up hr0 hrhs1 hhelp3 hsw1
    (arg := ⟨roundArgument ![n0, n1] 1, hrounds.1 1⟩) (by
      rw [Xe, Xe, Xe, ← pow_add]
      congr 1
      have e : roundArgument ![n0, n1] 1 = lookup 0 (roundArgument ![n0, n1] 0) + 8 * n1 := by
        simp [roundArgument, roundInput]
      simp only [e, lookup, dif_pos (hrounds.1 0)])
  have hr2 := round2 hb hsmn (polyNorm_Xe _) hg10 hg8 hg8up hr1 hrhs2 hhelp5 hsw2
    (arg := ⟨roundArgument ![n0, n1] 2, hrounds.1 2⟩) (by
      rw [Xe, Xe, Xe, ← pow_add]
      congr 1
      have e : roundArgument ![n0, n1] 2 = lookup 1 (roundArgument ![n0, n1] 1) + 8 * n0 := by
        simp [roundArgument, roundInput]
      simp only [e, lookup, dif_pos (hrounds.1 1)])
  have hr3 := round3 hb hsmn (polyNorm_Xe _) hg8 hg6 hg6up hr2 hrhs3 hhelp7 hsw3
    (arg := ⟨roundArgument ![n0, n1] 3, hrounds.1 3⟩) (by
      rw [Xe, Xe, Xe, ← pow_add]
      congr 1
      have e : roundArgument ![n0, n1] 3 = lookup 2 (roundArgument ![n0, n1] 2) + 8 * n1 := by
        simp [roundArgument, roundInput]
      simp only [e, lookup, dif_pos (hrounds.1 2)])
  -- The online outputs are the terminal encoding.
  have hb2 := hb
  unfold Stage_online.generatedRoot.body at hb2
  extract_lets at hb2
  have houtOnline := hb2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  -- Decryption reads coefficient `1` of `X^(outputExponent b)`.
  obtain ⟨wd, hd⟩ := hdec
  unfold Stage_decode.generatedRoot.body at hd
  dsimp only at hd
  obtain ⟨hm1, hm2, -, -, hdec1, hout⟩ := hd
  rw [hsm] at hm1
  have hw2 := modulusReduce_poly hQ7 hQ3 hm1
  have hw6 := modulusReduce_one hQ7 hQ3 hm2
  have hbound : 4 * roundBound3 + 4 < Q3 := by decide
  have ht : table 3 ⟨roundArgument ![n0, n1] 3, hrounds.1 3⟩ = 1 - n0 * n1 := by
    have h := hrounds.2
    simp only [outputExponent, lookup, dif_pos (hrounds.1 3)] at h
    exact h
  have ht1 : table 3 ⟨roundArgument ![n0, n1] 3, hrounds.1 3⟩ < N := by
    rw [ht]
    exact lt_of_le_of_lt (Nat.sub_le 1 _) (by decide)
  have hcoeff : (Xe (table 3 ⟨roundArgument ![n0, n1] 3, hrounds.1 3⟩)).coeff ⟨1, by decide⟩ =
      if 1 - n0 * n1 = 1 then 1 else 0 := by
    have h := Negacyclic.coeff_root_pow (R := Int) (n := N) hN
      ⟨table 3 ⟨roundArgument ![n0, n1] 3, hrounds.1 3⟩, ht1⟩ ⟨1, by decide⟩
    rw [Xe, show table 3 ⟨roundArgument ![n0, n1] 3, hrounds.1 3⟩ =
      ((⟨table 3 ⟨roundArgument ![n0, n1] 3, hrounds.1 3⟩, ht1⟩ : Fin N) : Nat) from rfl, h]
    simp [Fin.ext_iff, ht]
  have hval := decode_terminal hQ3 (by norm_num) hN hr3 hbound (k := ⟨1, by decide⟩)
    (by rw [hcoeff]; split <;> simp) (by
      rw [houtOnline, hw2, hw6]
      funext r c
      obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _
      simp only [matrixSub, matrixAdd, matrixMulScalarLeft_eq, Matrix.sub_apply, Matrix.add_apply,
        Matrix.smul_apply, smul_eq_mul, Matrix.one_apply_eq, one_mul]
      rfl) hdec1
  rw [hout, hval, hcoeff, hideal', hn0, hn1]
  rcases (by omega : n0 = 0 ∨ n0 = 1) with rfl | rfl <;>
    rcases (by omega : n1 = 0 ∨ n1 = 1) with rfl | rfl <;> rfl

/-- The average-case correctness claim of `average-n32-crt7x40`: the failure set is empty, so its
tape measure is below every bound. -/
theorem certificate : GeneratedClaim.CorrectnessClaim := by
  intro model external
  have hempty : {tape | ∃ execution, GeneratedClaim.Runs model external tape execution ∧
      ¬ execution.«stage_5» = execution.«ideal»} = ∅ := by
    ext tape
    simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
    rintro ⟨execution, hruns, hne⟩
    exact hne (correct hruns)
  rw [hempty, MeasureTheory.measure_empty]
  exact zero_le _

/-- info: 'NandProof.certificate' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms certificate

end NandProof
