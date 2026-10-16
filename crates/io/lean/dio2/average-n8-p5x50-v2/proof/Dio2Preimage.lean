import W_Stage_anchors_Facts
import W_Stage_preimages_0_0_Facts
import W_Stage_preimages_1_1_Facts
import W_Claim
import GenKey

/-! Generated sampled-anchor and bounded exact-preimage contracts.
Members stay quantified; only the exported prime families are specialized. -/

set_option maxHeartbeats 4000000
set_option maxRecDepth 16384
set_option linter.unusedVariables false

attribute [local irreducible] Mxx.Primitives.negacyclicModulus

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut

def SolvedPreimage {q n rows inner columns : Nat}
    (A : ExactMatrix q n rows inner) (target : ExactMatrix q n rows columns)
    (P : ExactMatrix q n inner columns) (cut : Nat) : Prop :=
  A * P = target ∧ ∃ error : ErrorMatrix n inner columns,
    P = reduceMatrix q n inner columns error ∧ CoeffBound error cut

theorem solvedPreimage_of_sampled {q n rows inner columns : Nat} {Token : Type}
    {backend : BackendContext} {sigma : Rat} {base digits cutoff : Int}
    {A : ExactMatrix q n rows inner} {T : TrapdoorValue (ExactMatrix q n rows inner) Token}
    (hT : trapdoorSample backend sigma base digits cutoff A T)
    {target : ExactMatrix q n rows columns} {cut : Nat} {P : ExactMatrix q n inner columns}
    (hP : preimageRunsDispatched backend (trapdoorPublic T) T target cut P) :
    SolvedPreimage A target P cut := by
  have hA : trapdoorPublic T = A := trapdoorSample_public hT
  rw [hA] at hP
  exact RefreshTopProof.Gen.preimage_sampled_gen hT hP

theorem anchors_trapdoor {backend : BackendContext} {params : W_Stage_anchors.Params}
    {u : Unit} {outputs : _} (h : W_Stage_anchors.generatedRoot backend params u outputs) :
    trapdoorSample backend (4/1) 4096 25 22011706 outputs.1 outputs.2.1 ∧
    trapdoorSample backend (4/1) 4096 25 22011706 outputs.2.2.1 outputs.2.2.2.1 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : W_Stage_anchors.generatedRoot.Ctx backend params := ⟨u, witness⟩
  have F := W_Stage_anchors.generatedRoot.facts ctx hbody
  rw [F.c0.hout]
  exact ⟨F.c0.h0_1, F.c0.h1_1⟩

open W_Stage_preimages_0_0 in
theorem preimages_0_0_spec {backend : BackendContext} {params : W_Stage_preimages_0_0.Params}
    {inputs : _} {outputs : _} (h : W_Stage_preimages_0_0.generatedRoot backend params inputs outputs)
    {sigma : Rat} {base digits cutoff : Int}
    {A : ExactMatrix W_Stage_preimages_0_0.modulus_0 8 2 54}
    (hT : trapdoorSample backend sigma base digits cutoff A inputs.1) :
    (∀ member : Fin 36, SolvedPreimage A (inputs.2.1 member) (outputs.1 member) 22011706) ∧
    (∀ member : Fin 36, SolvedPreimage A (inputs.2.2.1 member) (outputs.2.1 member) 22011706) ∧
    (∀ member : Fin 36, SolvedPreimage A (inputs.2.2.2.1 member) (outputs.2.2.1 member) 22011706) ∧
    (∀ member : Fin 36, SolvedPreimage A (inputs.2.2.2.2.1 member) (outputs.2.2.2.1 member) 22011706) ∧
    (∀ member : Fin 36, SolvedPreimage A (inputs.2.2.2.2.2.1 member) (outputs.2.2.2.2.1 member) 22011706) := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : W_Stage_preimages_0_0.generatedRoot.Ctx backend params := ⟨inputs, witness⟩
  have F := W_Stage_preimages_0_0.generatedRoot.facts ctx hbody
  rw [F.c0.hout]
  refine ⟨fun member ↦ ?_, fun member ↦ ?_, fun member ↦ ?_, fun member ↦ ?_, fun member ↦ ?_⟩
  · obtain ⟨sampled, _, hsample, hout⟩ := F.c0.h3_0 member
    change SolvedPreimage A _ (W_Stage_preimages_0_0.generatedRoot.w_3_0 ctx member) _
    rw [hout]
    exact solvedPreimage_of_sampled hT hsample
  · obtain ⟨sampled, _, hsample, hout⟩ := F.c0.h5_0 member
    change SolvedPreimage A _ (W_Stage_preimages_0_0.generatedRoot.w_5_0 ctx member) _
    rw [hout]
    exact solvedPreimage_of_sampled hT hsample
  · obtain ⟨sampled, _, hsample, hout⟩ := F.c0.h7_0 member
    change SolvedPreimage A _ (W_Stage_preimages_0_0.generatedRoot.w_7_0 ctx member) _
    rw [hout]
    exact solvedPreimage_of_sampled hT hsample
  · obtain ⟨sampled, _, hsample, hout⟩ := F.c0.h9_0 member
    change SolvedPreimage A _ (W_Stage_preimages_0_0.generatedRoot.w_9_0 ctx member) _
    rw [hout]
    exact solvedPreimage_of_sampled hT hsample
  · obtain ⟨sampled, _, hsample, hout⟩ := F.c0.h11_0 member
    change SolvedPreimage A _ (W_Stage_preimages_0_0.generatedRoot.w_11_0 ctx member) _
    rw [hout]
    exact solvedPreimage_of_sampled hT hsample

open W_Stage_preimages_1_1 in
theorem preimages_1_1_spec {backend : BackendContext} {params : W_Stage_preimages_1_1.Params}
    {inputs : _} {outputs : _} (h : W_Stage_preimages_1_1.generatedRoot backend params inputs outputs)
    {sigma : Rat} {base digits cutoff : Int}
    {A : ExactMatrix W_Stage_preimages_1_1.modulus_0 8 2 54}
    (hT : trapdoorSample backend sigma base digits cutoff A inputs.1) :
    SolvedPreimage A inputs.2.1 outputs 22011706 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : W_Stage_preimages_1_1.generatedRoot.Ctx backend params := ⟨inputs, witness⟩
  have F := W_Stage_preimages_1_1.generatedRoot.facts ctx hbody
  rw [F.c0.hout]
  exact solvedPreimage_of_sampled hT F.c0.h3_0

theorem claim_preimages {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs hashModel external execution) :
    ((∀ member : Fin 36, SolvedPreimage execution.«stage_1».1 (execution.«stage_8».1 member) (execution.«stage_9».1 member) 22011706) ∧
      (∀ member : Fin 36, SolvedPreimage execution.«stage_1».1 (execution.«stage_8».2.1 member) (execution.«stage_9».2.1 member) 22011706) ∧
      (∀ member : Fin 36, SolvedPreimage execution.«stage_1».1 (execution.«stage_8».2.2.1 member) (execution.«stage_9».2.2.1 member) 22011706) ∧
      (∀ member : Fin 36, SolvedPreimage execution.«stage_1».1 (execution.«stage_8».2.2.2.1 member) (execution.«stage_9».2.2.2.1 member) 22011706) ∧
      (∀ member : Fin 36, SolvedPreimage execution.«stage_1».1 (execution.«stage_8».2.2.2.2.1 member) (execution.«stage_9».2.2.2.2.1 member) 22011706)) ∧
    SolvedPreimage execution.«stage_1».2.2.1 execution.«stage_14».1 execution.«stage_15» 22011706 := by
  have hanchors := anchors_trapdoor (h.2.2.1)
  exact ⟨preimages_0_0_spec (h.2.2.2.2.2.2.2.2.2.2.1) hanchors.1,
    preimages_1_1_spec (h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) hanchors.2⟩

end Dio2Proof
