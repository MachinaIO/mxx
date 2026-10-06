import LwrChainAll
import LwrModelLut

/-!
# The finalized readout of a run is its model

The readout LUT and its finalization turn the state of the model at the LUT input into the model's
finalized error.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {tape : SampleTape}

set_option maxRecDepth 100000 in
set_option maxHeartbeats 64000000 in
/-- The finalized readout of a run: an encoding of the half-integer readout with scale `k`, whose
error is the model's finalized error, and the numerators of every ModDown of the model are
divisible by their dropped factors. -/
theorem chainFinal {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (hruns : GeneratedClaim.Runs model external tape execution) :
    ∃ w, Stage_online_eval.generatedRoot.body Backend.backend model tape [4]
        GeneratedClaim.stage_4_params
        (execution.«stage_2».1, execution.«stage_2».2.1, execution.«stage_2».2.2.1,
          execution.«stage_2».2.2.2.1, execution.«stage_3».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          external.input_0, execution.«stage_3».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_3».2.2.1, execution.«stage_3».2.2.2.1, execution.«stage_3».1,
          execution.«stage_3».2.2.2.2.1, execution.«stage_3».2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, ()) execution.«stage_4» w ∧
      div0 model external.input_0 external tape ∧ div1 model external.input_0 external tape ∧
      div2 model external.input_0 external tape ∧ div3 model external.input_0 external tape ∧
      div4 model external.input_0 external tape ∧ divLut model external.input_0 external tape ∧
      (∃ d : ErrorMatrix N 2 1, finalDigits = d ∧ CoeffBound d 524288 ∧
        (∀ r c (i : Fin N), i.val ≠ 0 → (d r c).coeff i = 0) ∧
        ∃ G : ExactMatrix Q1 N 1 2, gadgetMatrixRuns Backend.backend 1048576 2 G ∧
          G * reduceMatrix Q1 N 2 1 d = (matrixPolynomial [outScale] : ExactMatrix Q1 N 1 1)) ∧
      RelE (maskM tape) ![![matrixMul w.w_165_0 w.w_168_0, matrixMul w.w_184_0 w.w_168_0],
          ![matrixMul w.w_188_0 w.w_168_0, matrixMul w.w_192_0 w.w_168_0]]
        (fun j ↦ reducePoly Q1 N ((outScale : ErrorPoly N) * lutValue (zOf external 5) *
          unitPair (payloadM tape) j) • (1 : ExactMatrix Q1 N 1 1))
        ![matrixMul w.w_198_0 w.w_168_0, matrixMul w.w_204_0 w.w_168_0]
        (finalErr model external.input_0 external tape) := by
  obtain ⟨w, hb, hdiv0, hdiv1, hdiv2, hdiv3, hdiv4, hk, hrel⟩ := chain hruns
  obtain ⟨hvalid, hpay, hmask, hpre, henc, -, -, hreq, -, hreqtrue⟩ := hruns
  obtain ⟨-, -, -, ha, -, -, hs0, -, hs1, -, hs2, -, hs3, -, hs4, -⟩ := hvalid
  obtain ⟨-, -, -, -, -, hsum4⟩ := requirement_holds hreq hreqtrue
  obtain ⟨sel4, hsel4⟩ := one_hot hs4 hsum4
  have hspM := payload_model hpay
  have hsmM := mask_model hmask
  obtain ⟨g2, hg2⟩ := gadget_exists (m := 2) layout1 rfl rfl rfl
  have hg2u : centeredRebaseRuns g2 (centeredRebase g2 : ExactMatrix Q2 N 1 2) := ⟨hQ2, rfl⟩
  -- The block-4 target collapses to the LUT input target.
  have hnn : ∀ b' i, 0 ≤ selOf external b' i := by
    intro b' i
    fin_cases b'
    exacts [(hs0 i).1, (hs1 i).1, (hs2 i).1, (hs3 i).1, (hs4 i).1]
  have hstep4 : muOf external 5 = alphaOf external ⟨coords 4 sel4, coords_lt 4 sel4⟩ * muOf external 4 :=
    muOf_step external 4 (sel := sel4) hsel4 hnn (fun k ↦ (ha k).1)
  have hin := hrel.target_eq (collapse _ _ _ (fun i ↦ alphaOf external ⟨coords 4 i, coords_lt 4 i⟩)
    (nuOf external 4) _ sel4 hsel4 (fun i ↦ rfl))
  beta_reduce at hin
  rw [← hstep4, gReb_eq hg2 hg2u] at hin
  rw [muOf] at hin
  have hhelp := encrypt_helper5E henc hsmM hg2 hg2u
  have hsw := fun i ↦ (preprocess_switchE hpre hspM hg2 i).2
  obtain ⟨⟨dfin, hfd1, hfd2, hfd3, hfd4⟩, Rsum, hsum, v, hvp, hout⟩ := lutRoundE hb hg2 hg2u hin hhelp hsw
  have hfd := (⟨dfin, hfd1, hfd2, hfd3, g2, hg2, hfd4⟩ : ∃ d : ErrorMatrix N 2 1, finalDigits = d ∧
    CoeffBound d 524288 ∧ (∀ r c (i : Fin N), i.val ≠ 0 → (d r c).coeff i = 0) ∧
    ∃ G : ExactMatrix Q1 N 1 2, gadgetMatrixRuns Backend.backend 1048576 2 G ∧
      G * reduceMatrix Q1 N 2 1 d = (matrixPolynomial [outScale] : ExactMatrix Q1 N 1 1))
  -- The switching ciphertexts and the input key are the model's.
  have hfam : (fun k j ↦ ![![execution.«stage_2».1, execution.«stage_2».2.1],
      ![execution.«stage_2».2.2.1, execution.«stage_2».2.2.2.1]] k j) =
      lutFam model external.input_0 tape := by
    funext k j i
    have htop := congrFun (preprocess_switchTop hpre i) j
    fin_cases k
    · fin_cases j <;> exact htop
    · have h := bot_of_rhs ((preprocess_switchE hpre hspM hg2 i).2 j)
      fin_cases j
      · simp only [lutFam, lutSwitchBot, gadgetOf_eq hg2]
        rw [← htop]
        exact h
      · simp only [lutFam, lutSwitchBot, gadgetOf_eq hg2]
        rw [← htop]
        exact h
  rw [hk, hfam] at hsum hvp
  have hR : Rsum = lutRow model external.input_0 external tape := hsum.eq_encRow
  have hv : ∀ j, v j = lutV model external.input_0 external tape j := by
    intro j
    have h := fun row column ↦ hvp j row column
    rw [hR] at h
    exact div_of_identity (by norm_num) h
  have hdivLut : divLut model external.input_0 external tape := by
    intro j row column
    have h := hvp j row column
    rw [hR, hv j] at h
    exact h
  refine ⟨w, hb, hdiv0, hdiv1, hdiv2, hdiv3, hdiv4, hdivLut, hfd, ?_⟩
  have herr : (fun j ↦ (v j - ∑ i : Fin 16, substitution (sigma i).toNat (sigma_odd i)
      (Xe (n := N) (2 * zOf external 5)) • switchError tape [2] i j) * finalDigits) =
      finalErr model external.input_0 external tape := by
    funext j
    rw [hv j]
    rfl
  rw [← herr]
  exact hout

end LwrProof
