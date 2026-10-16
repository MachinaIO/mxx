import W_Stage_anchors_Facts
import W_Stage_preimages_0_0_Facts
import W_Stage_preimages_1_1_Facts
import W_Claim
import GenKey

/-!
# The decoder preimages

Stage `anchors` samples the step anchors `B_0`, `B_1` (`2 × 54` over `Q_0`) with their trapdoors
(`anchors_trapdoor`). Stage `preimages_0_0` samples, under the trapdoor of `B_0`, one preimage
`P_{p,j}` per prime `p` (five families, its inputs `2..6`) and per member `j` of the decoder key
family (`36` members) with `B_0 · P_{p,j} = K_{p,j}`; stage `preimages_1_1` samples, under the
trapdoor of `B_1`, the output preimage `P` with `B_1 · P = K` (`54 × 1`). Every preimage is the
reduction of an integer matrix with coefficients at most the graph's cutoff `22011706`
(`preimages_0_0_spec`, `preimages_1_1_spec`). `claim_preimages` wires the anchors stage into both
preimage stages through `W_GeneratedClaim.Runs`.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime

/-- `P` solves `A · P = target` exactly and is the reduction of an integer matrix with every
coefficient at most `cut`. -/
def SolvedPreimage {q n rows inner cols : Nat} (A : ExactMatrix q n rows inner)
    (target : ExactMatrix q n rows cols) (P : ExactMatrix q n inner cols) (cut : Nat) : Prop :=
  A * P = target ∧ ∃ d : ErrorMatrix n inner cols, P = reduceMatrix q n inner cols d ∧
    CoeffBound d cut

/-- A dispatched preimage under the public matrix of a sampled trapdoor `T` of `A`. -/
theorem solvedPreimage_of_sampled {q n rows inner cols : Nat} {Token : Type}
    {backend : BackendContext} {sigma : Rat} {base digits cutoff : Int}
    {A : ExactMatrix q n rows inner} {T : TrapdoorValue (ExactMatrix q n rows inner) Token}
    (hT : trapdoorSample backend sigma base digits cutoff A T)
    {target : ExactMatrix q n rows cols} {cut : Nat} {P : ExactMatrix q n inner cols}
    (hP : preimageRunsDispatched backend (trapdoorPublic T) T target cut P) :
    SolvedPreimage A target P cut := by
  have hA : trapdoorPublic T = A := trapdoorSample_public hT
  rw [hA] at hP
  exact RefreshTopProof.Gen.preimage_sampled_gen hT hP

/-- The anchors stage: both anchors are the public matrices of their sampled trapdoors. -/
theorem anchors_trapdoor {backend : BackendContext} {params : W_Stage_anchors.Params} {u : Unit}
    {outputs : _} (h : W_Stage_anchors.generatedRoot backend params u outputs) :
    trapdoorSample backend (4/1) 4096 25 22011706 outputs.1 outputs.2.1 ∧
      trapdoorSample backend (4/1) 4096 25 22011706 outputs.2.2.1 outputs.2.2.2.1 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : W_Stage_anchors.generatedRoot.Ctx backend params := ⟨u, witness⟩
  have F := W_Stage_anchors.generatedRoot.facts ctx hbody
  rw [F.c0.hout]
  exact ⟨F.c0.h0_1, F.c0.h1_1⟩

section Preimages00

open W_Stage_preimages_0_0

/-- The step-0 preimages: for a trapdoor (the stage's first input) sampled for `A`, every member
`j` of each of the five per-prime output families solves `A · P = K` for the matching member of
the matching target family (the stage's inputs `2..6`), within the cutoff `22011706`. -/
theorem preimages_0_0_spec {backend : BackendContext} {params : Params} {inputs : _}
    {outputs : _} (h : generatedRoot backend params inputs outputs)
    {sigma : Rat} {base digits cutoff : Int} {A : ExactMatrix modulus_0 8 2 54}
    (hT : trapdoorSample backend sigma base digits cutoff A inputs.1) :
    (∀ j : Fin 36, SolvedPreimage A (inputs.2.1 j) (outputs.1 j) 22011706) ∧
      (∀ j : Fin 36, SolvedPreimage A (inputs.2.2.1 j) (outputs.2.1 j) 22011706) ∧
      (∀ j : Fin 36, SolvedPreimage A (inputs.2.2.2.1 j) (outputs.2.2.1 j) 22011706) ∧
      (∀ j : Fin 36, SolvedPreimage A (inputs.2.2.2.2.1 j) (outputs.2.2.2.1 j) 22011706) ∧
      (∀ j : Fin 36, SolvedPreimage A (inputs.2.2.2.2.2.1 j) (outputs.2.2.2.2.1 j) 22011706) := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c0.hout]
  refine ⟨fun j ↦ ?_, fun j ↦ ?_, fun j ↦ ?_, fun j ↦ ?_, fun j ↦ ?_⟩
  · obtain ⟨w, -, hP, hw⟩ := F.c0.h3_0 j
    show SolvedPreimage A _ (generatedRoot.w_3_0 ctx j) _
    rw [hw]; exact solvedPreimage_of_sampled hT hP
  · obtain ⟨w, -, hP, hw⟩ := F.c0.h5_0 j
    show SolvedPreimage A _ (generatedRoot.w_5_0 ctx j) _
    rw [hw]; exact solvedPreimage_of_sampled hT hP
  · obtain ⟨w, -, hP, hw⟩ := F.c0.h7_0 j
    show SolvedPreimage A _ (generatedRoot.w_7_0 ctx j) _
    rw [hw]; exact solvedPreimage_of_sampled hT hP
  · obtain ⟨w, -, hP, hw⟩ := F.c0.h9_0 j
    show SolvedPreimage A _ (generatedRoot.w_9_0 ctx j) _
    rw [hw]; exact solvedPreimage_of_sampled hT hP
  · obtain ⟨w, -, hP, hw⟩ := F.c0.h11_0 j
    show SolvedPreimage A _ (generatedRoot.w_11_0 ctx j) _
    rw [hw]; exact solvedPreimage_of_sampled hT hP

end Preimages00

section Preimages11

open W_Stage_preimages_1_1

/-- The output preimage of the last step: for a trapdoor (the stage's first input) sampled for
`A`, the output solves `A · P = K` for the stage's second input, within the cutoff `22011706`. -/
theorem preimages_1_1_spec {backend : BackendContext} {params : Params} {inputs : _}
    {outputs : _} (h : generatedRoot backend params inputs outputs)
    {sigma : Rat} {base digits cutoff : Int} {A : ExactMatrix modulus_0 8 2 54}
    (hT : trapdoorSample backend sigma base digits cutoff A inputs.1) :
    SolvedPreimage A inputs.2.1 outputs 22011706 := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : generatedRoot.Ctx backend params := ⟨inputs, witness⟩
  have F := generatedRoot.facts ctx hbody
  rw [F.c0.hout]
  exact solvedPreimage_of_sampled hT F.c0.h3_0

end Preimages11

/-- The claim-level wiring: the step-0 preimages solve against the anchor `B_0` (output 0 of the
anchors stage) and the decoder key families of `keys_online_0_0` (stage 8), and the output
preimage solves against `B_1` (output 2 of the anchors stage) and the key of `keys_online_1_1`
(stage 14), all within the cutoff `22011706`. -/
theorem claim_preimages {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution) :
    ((∀ j : Fin 36, SolvedPreimage execution.«stage_1».1 (execution.«stage_8».1 j)
        (execution.«stage_9».1 j) 22011706) ∧
      (∀ j : Fin 36, SolvedPreimage execution.«stage_1».1 (execution.«stage_8».2.1 j)
        (execution.«stage_9».2.1 j) 22011706) ∧
      (∀ j : Fin 36, SolvedPreimage execution.«stage_1».1 (execution.«stage_8».2.2.1 j)
        (execution.«stage_9».2.2.1 j) 22011706) ∧
      (∀ j : Fin 36, SolvedPreimage execution.«stage_1».1 (execution.«stage_8».2.2.2.1 j)
        (execution.«stage_9».2.2.2.1 j) 22011706) ∧
      (∀ j : Fin 36, SolvedPreimage execution.«stage_1».1 (execution.«stage_8».2.2.2.2.1 j)
        (execution.«stage_9».2.2.2.2.1 j) 22011706)) ∧
    SolvedPreimage execution.«stage_1».2.2.1 execution.«stage_14».1 execution.«stage_15»
      22011706 := by
  obtain ⟨-, -, h1, -, -, -, -, -, -, -, h9, -, -, -, -, -, h15, -⟩ := h
  have hA := anchors_trapdoor h1
  exact ⟨preimages_0_0_spec h9 hA.1, preimages_1_1_spec h15 hA.2⟩

end Dio2Proof
