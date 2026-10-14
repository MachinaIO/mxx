import RingGswBggStep

/-!
# The average-case correctness certificate

Every execution on every sample tape satisfying the generated claim's run relation decrypts the
ideal output bit, so the failure set of the claim is empty. The tape samplers record their cutoffs,
so every fresh encoding error is at most the cutoff; the evaluation and the ideal circuit read the
same gate records, so `layer_step` carries the encodings of the ideal bits through every layer; and
the decoding action and the rounding of `decode_output` return the output bit.
-/

namespace RingGswBggProof

open Mxx.Primitives MxxRuntime RingGswBgg

set_option maxHeartbeats 4000000

/-- The fresh error cutoff and the error bound after `l` layers. -/
abbrev CUTOFF : Nat := 20
def bound (l : Nat) : Nat := (nextBound M N (BASE.toNat / 2))^[l] CUTOFF

theorem bound_succ (l : Nat) : bound (l + 1) = nextBound M N (BASE.toNat / 2) (bound l) :=
  Function.iterate_succ_apply' _ _ _

theorem cutoff_le_bound (l : Nat) : CUTOFF ≤ bound l := by
  induction l with
  | zero => exact le_rfl
  | succ l ih =>
    rw [bound_succ]
    refine ih.trans ?_
    unfold nextBound
    rw [Nat.add_mul]
    omega

set_option maxRecDepth 100000 in
/-- The decryption condition at the claim's depth: four times the output error bound, plus four,
is below the modulus. -/
theorem decryption_bound :
    4 * (M * bound GeneratedClaim.stage_2_params.«depth».toNat * (BASE.toNat / 2)) + 4 < Q := by
  decide +kernel

theorem backend_exact :
    ∀ layout, Backend.backend.regularLayout Q N = some layout → layout.droppedModuli = 0 := by
  intro layout h
  have hl : Backend.backend.regularLayout Q N = some Backend.layout0 := by
    simp [Backend.backend]
  rw [hl] at h
  injection h with h
  subst h
  rfl

set_option maxRecDepth 100000 in
theorem run_correct {hashModel external tape execution}
    (h : GeneratedClaim.Runs hashModel external tape execution) :
    execution.«stage_3» = execution.«ideal» := by
  have hq : 1 < Q := by norm_num
  have hn : 0 < N := by norm_num
  have hodd : Q % 2 = 1 := by norm_num
  obtain ⟨hvalid, -, hencrypt, hevaluate, hdecrypt, -, hideal, -⟩ := h
  obtain ⟨⟨w2, w4, w5, w7, w8, w10, w10', w11, w12, w17, w19, w21, w23⟩, he⟩ := hencrypt
  dsimp only [Stage_encrypt.generatedRoot.body, Stage_encrypt.generatedRoot.constraints_0] at he
  obtain ⟨hcat, hhash, -, hslices, -, hlift, hG, -, hencs, hhash1, -, -, -, -, -, -, hslice1, hG',
    htensor, hgauss1, -, -, -, -, -, -, hslice23, hout1⟩ := he
  obtain ⟨⟨v2, v3, v5, v6, v6', v6'', v12, v13, v15, v15', v15'', v17, v18, v21, v22, v24⟩, hv⟩ :=
    hevaluate
  dsimp only [Stage_evaluate.generatedRoot.body, Stage_evaluate.generatedRoot.constraints_0] at hv
  obtain ⟨hvhash, -, hvslices, -, hvlift, -, hzip, hvhash1, -, -, -, -, -, -, hvslice1, -, hloop,
    hosrc, -, -, hpk, -, hcatR, hdecR, -, -, hvec, hout2⟩ := hv
  obtain ⟨⟨i1, i6, i8, i9⟩, hid⟩ := hideal
  dsimp only [Ideal.generatedRoot.body, Ideal.generatedRoot.constraints_0] at hid
  obtain ⟨-, hibits, -, hiloop, hosrc', -, -, hibit, hout3⟩ := hid
  obtain ⟨⟨d3, d7⟩, hdx⟩ := hdecrypt
  dsimp only [Stage_decrypt.generatedRoot.body, Stage_decrypt.generatedRoot.constraints_0] at hdx
  obtain ⟨hcat', -, hthr, hout4⟩ := hdx
  have hsG : matrixMul w2 w8 = w2 * w8 := rfl
  -- The constant-one encoding.
  have hone : Enc w2 (matrixMul w2 w8) w23 w12 1 CUTOFF := by
    have hG'' : w17 = w8 := gadgetMatrixRuns_unique hG' hG
    subst hG''
    obtain ⟨e, he, hb⟩ := gaussian_error hn hgauss1
    have h19 : w19 = matrixMul w2 w17 := by
      ext i j
      fin_cases i
      exact tensorRuns_one htensor j
    refine ⟨e, ?_, hb⟩
    rw [sliceMatrix_full hslice23, h19, he]
    ext i j
    simp [matrixAdd, matrixSub, matrixMul]
  -- The input encodings.
  have hinputs : ∀ p, Enc w2 (matrixMul w2 w8) (w10 p) (w5 p) (w7 p) CUTOFF ∧
      w7 p = bitMatrix (decide (external.input_1 p = 1)) := by
    intro p
    have hl := hlift p
    dsimp only [Stage_encrypt.parallel_generatedRoot_7,
      Stage_encrypt.parallel_generatedRoot_7.constraints_0] at hl
    have hp := hencs p
    dsimp only [Stage_encrypt.parallel_generatedRoot_10] at hp
    obtain ⟨g, hp⟩ := hp
    dsimp only [Stage_encrypt.parallel_generatedRoot_10.constraints_0] at hp
    obtain ⟨hg, hpo⟩ := hp
    simp only [Prod.mk.injEq] at hpo
    obtain ⟨hw10, -, -⟩ := hpo
    obtain ⟨e, he, hb⟩ := gaussian_error hn hg
    refine ⟨⟨e, by rw [hw10, he]; rfl, hb⟩, ?_⟩
    rw [hl]
    exact liftInteger_bit (hvalid.2.1 p)
  -- The evaluation reads the encryption's public matrices and plaintexts.
  have hv2 : v2 = w4 := hashSample_unique hvhash hhash
  subst hv2
  have hkeys : ∀ p, v3 p = w5 p := by
    intro p
    have ha := hvslices p
    have hb := hslices p
    dsimp only [Stage_evaluate.parallel_generatedRoot_3, Stage_encrypt.parallel_generatedRoot_5] at ha hb
    obtain ⟨a, ha⟩ := ha
    obtain ⟨b, hb⟩ := hb
    dsimp only [Stage_evaluate.parallel_generatedRoot_3.constraints_0,
      Stage_encrypt.parallel_generatedRoot_5.constraints_0] at ha hb
    obtain ⟨-, -, -, -, -, -, ha, rfl⟩ := ha
    obtain ⟨-, -, -, -, -, -, hb, rfl⟩ := hb
    exact sliceMatrix_unique ha hb
  have hplain : ∀ p, v5 p = w7 p := by
    intro p
    have ha := hvlift p
    have hb := hlift p
    dsimp only [Stage_evaluate.parallel_generatedRoot_5, Stage_encrypt.parallel_generatedRoot_7,
      Stage_evaluate.parallel_generatedRoot_5.constraints_0,
      Stage_encrypt.parallel_generatedRoot_7.constraints_0] at ha hb
    rw [ha, hb]
  have hv12 : v12 = w11 := hashSample_unique hvhash1 hhash1
  subst hv12
  have hv13 : v13 = w12 := sliceMatrix_unique hvslice1 hslice1
  subst hv13
  rw [hout1] at hloop hzip
  -- The initial layer encodes the input bits.
  have hinit : ∀ p, Enc w2 (matrixMul w2 w8) ((v6, v6', v6'', ()).1 p) ((v6, v6', v6'', ()).2.1 p)
      ((v6, v6', v6'', ()).2.2.1 p) (bound 0) ∧ (v6, v6', v6'', ()).2.2.1 p = bitMatrix (i1 p) := by
    intro p
    have hz := hzip p
    dsimp only [Stage_evaluate.parallel_generatedRoot_6,
      Stage_evaluate.parallel_generatedRoot_6.constraints_0] at hz
    simp only [Prod.mk.injEq] at hz
    obtain ⟨hz0, hz1, hz2, -⟩ := hz
    have hb := hibits p
    dsimp only [Ideal.parallel_generatedRoot_1, Ideal.parallel_generatedRoot_1.constraints_0] at hb
    obtain ⟨henc, hbit⟩ := hinputs p
    simp only at hz0 hz1 hz2 ⊢
    rw [hz0, hz1, hz2, hkeys, hplain]
    exact ⟨henc, by rw [hbit, hb]⟩
  -- Every layer encodes the ideal layer.
  have hcount : GeneratedClaim.ideal_params.«depth».toNat =
      GeneratedClaim.stage_2_params.«depth».toNat := rfl
  rw [hcount] at hiloop
  have hfinal := iterRuns_invariant₂
    (Inv := fun l st bs ↦ ∀ p, Enc w2 (matrixMul w2 w8) (st.1 p) (st.2.1 p) (st.2.2.1 p) (bound l) ∧
      st.2.2.1 p = bitMatrix (bs p)) hloop hiloop hinit
    (fun l x y x' y' hinv hx hy ↦ by
      have := layer_step hq hn backend_exact hG hsG (cutoff_le_bound l) hone rfl rfl hinv hx hy
      beta_reduce
      rw [bound_succ]
      exact this)
  -- The output slot.
  have hsrc : i8 = v17 := familyGetDynamic_eq hosrc' hosrc
  subst hsrc
  obtain ⟨pos, hpos, rfl⟩ := hvec
  obtain ⟨pos₁, hpos₁, rfl⟩ := hpk
  obtain ⟨pos₂, hpos₂, rfl⟩ := hibit
  have e₁ : pos = pos₁ := Fin.ext (by omega)
  have e₂ : pos = pos₂ := Fin.ext (by omega)
  subst e₁ e₂
  obtain ⟨henc, hbit⟩ := hfinal pos
  -- Decryption.
  have hd3 : d3 = w2 := concatColumns_unique hcat' hcat
  subst hd3
  obtain ⟨hGD, -⟩ := gadget_exact hq hn hG hdecR backend_exact
  have htarget : ∀ r c (k : Fin N), k.val ≠ 0 → (v21 r c).coeff k = 0 := by
    intro r c k hk
    rw [hcatR r c]
    split_ifs
    · simp
    · rw [MxxRuntime.matrixPolynomial_single, ← reducePoly_intCast, reducePoly_coeff hq hn,
        MxxExponentLut.intCast_coeff_ne_zero hn _ hk, Int.cast_zero]
  obtain ⟨d, hD, hdb, hconst⟩ :=
    MxxExponentLut.gadgetDecomposeRuns_constant hq hn hdecR backend_exact htarget
  have hsT : d3 * v21 = liftInteger (((Q / 2 : Nat) : Int)) := by
    rw [secret_mul_target hcat hcatR]
    ext i j
    fin_cases i; fin_cases j
    simp only [MxxRuntime.matrixPolynomial_single, liftInteger]
    norm_num [Int.fdiv]
  have hB := decryption_bound
  rw [hout2] at hthr
  have hdecode := decode_output hq hodd hn hsG henc hbit hGD hD hdb hconst hsT hB hthr
  rw [hout4, hout3, hdecode]

theorem certificate : GeneratedClaim.CorrectnessClaim := by
  intro model external
  have hempty : {tape | ∃ execution, GeneratedClaim.Runs model external tape execution ∧
      ¬ execution.«stage_3» = execution.«ideal»} = ∅ := by
    ext tape
    simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
    rintro ⟨execution, hruns, hne⟩
    exact hne (run_correct hruns)
  rw [hempty, MeasureTheory.measure_empty]
  exact zero_le _

/-- info: 'RingGswBggProof.certificate' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms certificate

end RingGswBggProof
