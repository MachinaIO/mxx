import PublicIpDecode
import PublicIpInputs

/-!
# The average-case correctness certificate

Every execution on every sample tape satisfying the generated claim's run relation checks its output
with a zero residual count, so the failure set of the claim is empty. The requirement makes the
plaintext of member `k` the monomial `X^(2 a_k)`; the tree multiplies members `0`, `1`, and `3`
into a public encoding of `X^(2 (a₀ + a₁ + a₃))`, which is the claimed plaintext
`X^(2 ((a₀ + a₁ + a₃) mod 32))` since `X^64 = 1`; and the decode stage decodes every residual
coefficient of that encoding to zero.
-/

namespace PublicIpProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxExponentLutPublicIp

variable {tape : SampleTape} {path : List Nat}

theorem mask_ternary {out : ExactMatrix Q N 1 1}
    (h : Stage_mask.generatedRoot tape path GeneratedClaim.stage_0_params () out) :
    ∃ s : ErrorPoly N, out = (fun _ _ ↦ reducePoly Q N s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly_at hN hw
  exact ⟨s, hout.trans hs, hn⟩

/-- The terminal error bound fits a quarter of `Q`. -/
theorem bound_fits : 4 * productBound (productBound 26 26) 26 + 4 < Q := by
  simp only [productBound]
  decide

set_option maxHeartbeats 64000000 in
/-- Every run decodes a zero residual count, the ideal output. -/
theorem correct {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (hruns : GeneratedClaim.Runs model external tape execution) :
    execution.«stage_3» = execution.«ideal» := by
  obtain ⟨hvalid, hmask, henc, honline, hdec, hreq, hideal, hreqtrue⟩ := hruns
  obtain ⟨sm, hsm, hsmn⟩ := mask_ternary hmask
  obtain ⟨-, ha, -, hae, -, hee⟩ := hvalid
  obtain ⟨hmem, -, hc, hrel⟩ := requirement_holds hreq hreqtrue
  rw [ideal_value hideal]
  have hN32 : ((N : Nat) : Int) = 32 := rfl
  have hN32' : N = 32 := rfl
  set a := external.input_1
  -- The plaintexts of the members and the claimed plaintext.
  let α : Fin 5 → ErrorPoly N := fun k ↦ Xe (a k * 2).toNat
  have hα : ∀ k, message (n := N) (external.input_2 k) (external.input_3 k) = α k := fun k ↦
    message_eq_Xe (hmem k).1 (by have := (hmem k).2; push_cast; linarith)
      (by have := hae k; omega)
  have hμ : message (n := N) external.input_4 external.input_5 = α 0 * α 1 * α 3 := by
    rw [message_eq_Xe (x := (a 0 + a 1 + a 3) % 32 * 2) hc (by push_cast; linarith) (by omega),
      Xe_mul, Xe_mul]
    have h0 := (ha 0).1
    have h1 := (ha 1).1
    have h3 := (ha 3).1
    rw [show (a 0 * 2).toNat + (a 1 * 2).toNat + (a 3 * 2).toNat =
      ((a 0 + a 1 + a 3) % 32 * 2).toNat + 2 * N * ((a 0 + a 1 + a 3) / 32).toNat by
        rw [hN32']; omega,
      Xe_period]
  -- Encryption.
  obtain ⟨G, hG⟩ := gadget_exists (m := 4) layoutQ rfl rfl rfl
  have hmembers := encrypt_member henc hsm hG
  have hmemα : ∀ k : Fin 5, Rel sm (memberKey model external.input_0 k)
      (fun j ↦ reducePoly Q N (α k * unitPair sm j) • G)
      ![execution.«stage_1».1 k, execution.«stage_1».2.1 k] 26 := fun k ↦ by
    rw [← hα k]; exact (hmembers k).1
  have hpayα : ∀ k : Fin 5, execution.«stage_1».2.2.1 k = fun _ _ ↦ reducePoly Q N (α k) :=
    fun k ↦ by rw [← hα k]; exact (hmembers k).2
  -- Online evaluation and the check of its output.
  have hout := root honline (fun k ↦ polyNorm_Xe hN1 _)
    (by simp only [Xe_mul]; exact polyNorm_Xe hN1 _) hG hmemα hpayα
  rw [← hμ] at hout
  exact decode_value hdec hsm hG hout bound_fits

/-- The average-case correctness claim of `average-n32-crt2x40`: the failure set is empty, so its
tape measure is below every bound. -/
theorem certificate : GeneratedClaim.CorrectnessClaim := by
  intro model external
  have hempty : {tape | ∃ execution, GeneratedClaim.Runs model external tape execution ∧
      ¬ execution.«stage_3» = execution.«ideal»} = ∅ := by
    ext tape
    simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
    rintro ⟨execution, hruns, hne⟩
    exact hne (correct hruns)
  rw [hempty, MeasureTheory.measure_empty]
  exact zero_le _

/-- info: 'PublicIpProof.certificate' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms certificate

end PublicIpProof
