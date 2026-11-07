import PeKeygenScatter
import Stage_keygen_helpers_Facts

set_option maxRecDepth 200000
set_option maxHeartbeats 2000000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

abbrev KeygenOutputs := (Fin 256 → ExactMatrix Q N 295 1) ×
  (Fin 256 → ExactMatrix Q N 30 1) × (Fin 256 → ExactMatrix Q N 1 1) ×
  ExactMatrix Q N 60 1 × Unit

def anchorOut (outputs : KeygenOutputs) : Fin 256 → ExactMatrix Q N 295 1 := outputs.1
def vCOut (outputs : KeygenOutputs) : Fin 256 → ExactMatrix Q N 30 1 := outputs.2.1
def vIOut (outputs : KeygenOutputs) : Fin 256 → ExactMatrix Q N 1 1 := outputs.2.2.1
def vFOut (outputs : KeygenOutputs) : ExactMatrix Q N 60 1 := outputs.2.2.2.1

noncomputable def keySample {rows : Nat} (tape : SampleTape) (site : List Nat) : ErrorMatrix N rows 1 :=
  tapeWitness tape site keyLaw

theorem key_sample {rows : Nat} {tape : SampleTape} {site : List Nat}
    {output : ExactMatrix Q N rows 1}
    (sample : gaussianSampleAt tape site
      (2007780228562889017327695763426836351403150354206288662473911661538034308791619296312261592598031117 /
        5000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000)
      2610114297 output) :
    output = reduceMatrix Q N rows 1 (keySample tape site) ∧
      CoeffBound (keySample (rows := rows) tape site) keyCutoff :=
  gaussianSampleAt_explicit hN sample

noncomputable def helperSample (tape : SampleTape) (path : List Nat) (pair : Fin 11520) :
    ErrorMatrix N 60 1 := fun entry _ ↦
  (tapeWitness tape (path ++ [0, pair.val / 45] ++ [0]) keyLaw : ErrorMatrix N 60 45)
    entry ⟨pair % 45, by omega⟩

theorem helpers_sample {tape : SampleTape} {path : List Nat}
    {helpers : Fin 256 → ExactMatrix Q N 60 45}
    (h : Stage_keygen_helpers.generatedRoot tape path GeneratedClaim.stage_6_params () helpers) :
    ∀ pair, packedHelpers helpers pair = reduceMatrix Q N 60 1 (helperSample tape path pair) ∧
      packedHelpers helpers pair = packedHelpers helpers pair ∧
      CoeffBound (helperSample tape path pair) keyCutoff := by
  obtain ⟨witness, body⟩ := h
  have facts := Stage_keygen_helpers.generatedRoot.facts ⟨(), witness⟩ body
  rw [facts.c0.hout]
  intro pair
  obtain ⟨sample, sampled, output⟩ := facts.c0.h0_0 ⟨pair / 45, by omega⟩
  obtain ⟨equal, bound⟩ := gaussianSampleAt_explicit hN sampled
  refine ⟨?_, rfl, ?_⟩
  · funext entry column
    rw [Subsingleton.elim column 0]
    change (Stage_keygen_helpers.generatedRoot.w_0_0 ⟨(), witness⟩
      ⟨pair.val / 45, by omega⟩) entry ⟨pair.val % 45, by omega⟩ = _
    rw [output, equal]
    rfl
  · intro entry column coefficient
    exact bound entry ⟨pair % 45, by omega⟩ coefficient

def anchorW {model : HashModel} {tape : SampleTape} {path : List Nat}
    (ctx : Stage_keygen.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_7_params) :
    Fin 256 → ExactMatrix Q N 295 1 := Stage_keygen.generatedRoot.w_28_0 ctx

theorem keygen_samples {model : HashModel} {tape : SampleTape} {path : List Nat}
    {ctx : Stage_keygen.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_7_params}
    {outputs : KeygenOutputs} (facts : Stage_keygen.generatedRoot.Facts ctx outputs) :
    (∀ row : Fin 256, Stage_keygen.generatedRoot.w_26_0 ctx row =
      reduceMatrix Q N 30 1 (keySample tape (path ++ [26, row.val] ++ [0])) ∧
      Stage_keygen.generatedRoot.w_26_0 ctx row = Stage_keygen.generatedRoot.w_26_0 ctx row ∧
      CoeffBound (keySample (rows := 30) tape (path ++ [26, row.val] ++ [0])) keyCutoff) ∧
    (∀ row : Fin 256, Stage_keygen.generatedRoot.w_26_1 ctx row =
      reduceMatrix Q N 1 1 (keySample tape (path ++ [26, row.val] ++ [1])) ∧
      Stage_keygen.generatedRoot.w_26_1 ctx row = Stage_keygen.generatedRoot.w_26_1 ctx row ∧
      CoeffBound (keySample (rows := 1) tape (path ++ [26, row.val] ++ [1])) keyCutoff) := by
  constructor <;> intro row
  all_goals
    obtain ⟨compact, identity, routed, hc, hi, -, -, -, output⟩ := facts.c0.h26_2 row
    obtain ⟨ec, bc⟩ := key_sample hc
    obtain ⟨ei, bi⟩ := key_sample hi
    simp only [Prod.mk.injEq] at output
    obtain ⟨oc, oi, -⟩ := output
  · exact ⟨oc.trans ec, rfl, bc⟩
  · exact ⟨oi.trans ei, rfl, bi⟩

theorem keygen_anchors {model : HashModel} {tape : SampleTape} {path : List Nat}
    {td : TrapdoorValue (ExactMatrix Q N 1 295) Unit} (kind : td.kind = .sampledSecret)
    {key : ByteArray} {K0 K1 : ExactMatrix Q N 1 30}
    {helpers : Fin 256 → ExactMatrix Q N 60 45} {BQ : ExactMatrix PQ N 1 295}
    {witness : Stage_keygen.generatedRoot.Witness} {outputs : KeygenOutputs}
    (facts : Stage_keygen.generatedRoot.Facts (⟨(td, key, K0, K1, helpers, BQ, ()), witness⟩ :
      Stage_keygen.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_7_params) outputs)
    (row : Fin 256) :
    let ctx : Stage_keygen.generatedRoot.Ctx Backend.backend model tape path
      GeneratedClaim.stage_7_params := ⟨(td, key, K0, K1, helpers, BQ, ()), witness⟩
    trapdoorPublic td * anchorW ctx row = Stage_keygen.generatedRoot.w_27_0 ctx row ∧
      ∃ sample : ErrorMatrix N 295 1, anchorW ctx row = reduceMatrix Q N 295 1 sample ∧
        CoeffBound sample keyCutoff := by
  dsimp only
  obtain ⟨preimage, compact, -, preimage_run, compact_run, output⟩ := facts.c0.h28_0 row
  have compact_eq : compact = preimage :=
    centeredRebaseRuns_unique compact_run.2.2.1 (centeredRebaseRuns_self hQ hN preimage)
  rw [anchorW, output, compact_eq]
  refine ⟨preimageRunsDispatched_equation hQ hN preimage_run, ?_⟩
  rcases preimage_run.2 with sampled | imported
  · exact preimageRuns_bounded sampled
  · exact absurd (kind.symm.trans imported.1) (by decide)

theorem family_sum {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_keygen.Params} {family : Fin 256 → ExactMatrix Q N 1 256}
    {first second final : ExactMatrix Q N 1 256}
    (first_get : familyGetStatic family 0 first) (second_get : familyGetStatic family 1 second)
    (loop : MxxIR.IterRuns (fun index current next ↦
      Stage_keygen.sequential_generatedRoot_18 Backend.backend model tape (path ++ [18, index])
        params index (current, family, ()) next) 254 (first + second) final) :
    final = ∑ row : Fin 256, family row := by
  have step : ∀ index current next (bound : index < 254),
      Stage_keygen.sequential_generatedRoot_18 Backend.backend model tape (path ++ [18, index])
        params index (current, family, ()) next →
      next = current + family ⟨index + 2, by omega⟩ := by
    intro index current next bound body
    obtain ⟨value, -, -, get, output⟩ := body
    obtain ⟨_, equal⟩ := familyGet_eq get
    rw [output, equal]
    rfl
  have sum := stream_sum (fun index : Fin 254 ↦ family ⟨index.val + 2, by omega⟩) step loop
  obtain ⟨_, first_eq⟩ := familyGet_eq first_get
  obtain ⟨_, second_eq⟩ := familyGet_eq second_get
  rw [first_eq, second_eq] at sum
  rw [sum]
  rw [Fin.sum_univ_succ (f := family)]
  rw [Fin.sum_univ_succ (f := fun index : Fin 255 ↦ family index.succ)]
  abel

set_option maxHeartbeats 16000000 in
theorem keygen_helper_sum {model : HashModel} {tape : SampleTape} {path : List Nat}
    {ctx : Stage_keygen.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_7_params}
    {outputs : KeygenOutputs} (facts : Stage_keygen.generatedRoot.Facts ctx outputs)
    (target : Fin 256) :
    Stage_keygen.generatedRoot.w_18_0 ctx 0 target =
      (∑ lane : Fin 45, helperTerm model (Stage_keygen.generatedRoot.w_2_0 ctx)
        (packedHelpers (Stage_keygen.generatedRoot.w_10_0 ctx)) target lane) 0 0 := by
  have gadget := gadgetMatrixRuns_unique facts.c0.h11_0 g0_runs
  have joined := facts.c0.h13_0
  rw [gadget] at joined
  have zero : Stage_keygen.generatedRoot.w_13_0 ctx = gZero :=
    concat_joinRow (r := ![g0, 0]) joined
  have rows := facts.c0.h14_0
  rw [zero] at rows
  have row_sums : Stage_keygen.generatedRoot.w_14_0 ctx =
      fun row ↦ ∑ lane : Fin 45, scatterTerm model (Stage_keygen.generatedRoot.w_2_0 ctx)
        (Stage_keygen.generatedRoot.w_10_0 ctx row) row lane := by
    funext row
    exact helperScatterRow row.isLt (rows row)
  have total := family_sum facts.c0.h15_0 facts.c0.h16_0 facts.c0.h18_0
  rw [total, row_sums]
  exact scatter_gather _ _ _ target

set_option maxHeartbeats 16000000 in
theorem keygen_targets {model : HashModel} {tape : SampleTape} {path : List Nat}
    {td : TrapdoorValue (ExactMatrix Q N 1 295) Unit} {key : ByteArray}
    {K0 K1 : ExactMatrix Q N 1 30} {helpers : Fin 256 → ExactMatrix Q N 60 45}
    {BQ : ExactMatrix PQ N 1 295} {witness : Stage_keygen.generatedRoot.Witness}
    {outputs : KeygenOutputs}
    (facts : Stage_keygen.generatedRoot.Facts (⟨(td, key, K0, K1, helpers, BQ, ()), witness⟩ :
      Stage_keygen.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_7_params) outputs)
    (row : Nat) (bound : row < 256) :
    let ctx : Stage_keygen.generatedRoot.Ctx Backend.backend model tape path
      GeneratedClaim.stage_7_params := ⟨(td, key, K0, K1, helpers, BQ, ()), witness⟩
    Stage_keygen.generatedRoot.w_27_0 ctx ⟨row, bound⟩ =
      substituteMatrix (slotInv row) (slotInv_odd ⟨row, bound⟩)
        (slotResidual model key
          (messageRow model key 0 - joinRow ![K0, K1] *
            reduceMatrix Q N 60 1 (keySample tape (path ++ [7])))
          (packedHelpers helpers) (expand (fun _ _ ↦ BQ 0 0))
          (Stage_keygen.generatedRoot.w_26_0 ctx ⟨row, bound⟩)
          (Stage_keygen.generatedRoot.w_26_1 ctx ⟨row, bound⟩) ⟨row, bound⟩) := by
  dsimp only
  let ctx : Stage_keygen.generatedRoot.Ctx Backend.backend model tape path
    GeneratedClaim.stage_7_params := ⟨(td, key, K0, K1, helpers, BQ, ()), witness⟩
  have key_row : Stage_keygen.generatedRoot.w_3_0 ctx = messageRow model key 0 :=
    hashSample_eq facts.c0.h3_0
  have joined : Stage_keygen.generatedRoot.w_6_0 ctx = joinRow ![K0, K1] :=
    concat_joinRow (r := ![K0, K1]) facts.c0.h6_0
  obtain ⟨sample, -⟩ := key_sample facts.c0.h7_0
  have y : Stage_keygen.generatedRoot.w_9_0 ctx = messageRow model key 0 -
      joinRow ![K0, K1] * reduceMatrix Q N 60 1 (keySample tape (path ++ [7])) := by
    show matrixSub _ (matrixMul _ _) = _
    rw [key_row, joined, sample]
    rfl
  obtain ⟨_, slice_entry⟩ := sliceMatrix_column facts.c0.h20_0
  have first : Stage_keygen.generatedRoot.w_20_0 ctx = fun _ _ ↦ BQ 0 0 := by
    funext entry column
    rw [Subsingleton.elim entry 0, Subsingleton.elim column 0, slice_entry]
    rfl
  have expanded : Stage_keygen.generatedRoot.w_25_0 ctx = expand (fun _ _ ↦ BQ 0 0) := by
    have expand_eq := expand_read facts.c0.h21_0 facts.c0.h22_0 facts.c0.h24_0
    show Stage_keygen.generatedRoot.w_24_0 ctx = _
    rw [expand_eq, first]
  obtain ⟨compact, identity, routed, -, -, -, -, route, compact_output⟩ :=
    facts.c0.h26_2 ⟨row, bound⟩
  have route_eq := routing_slot bound route
  rw [expanded] at route_eq
  simp only [Prod.mk.injEq] at compact_output
  obtain ⟨compact_eq, identity_eq, residual_eq, -⟩ := compact_output
  obtain ⟨selected, target, -, -, -, -, -, -, -, -, slice_run, -, -, target_run, output⟩ :=
    facts.c0.h27_0 ⟨row, bound⟩
  have target_eq := routing_slotInv bound target_run
  rw [output, target_eq]
  congr 1
  have selected_eq : selected 0 0 = Stage_keygen.generatedRoot.w_18_0 ctx 0 ⟨row, bound⟩ :=
    slice_run.2.2.2.2.2.2.2.2 0 0 (by norm_num) (by simpa using bound)
  have helpers_eq : Stage_keygen.generatedRoot.w_18_0 ctx 0 ⟨row, bound⟩ =
      (∑ lane : Fin 45, helperTerm model key (packedHelpers helpers) ⟨row, bound⟩ lane) 0 0 :=
    keygen_helper_sum facts ⟨row, bound⟩
  funext entry column
  rw [Subsingleton.elim entry 0, Subsingleton.elim column 0]
  change (matrixSub (matrixMulScalarLeft (liftInteger
    (MxxIR.selectInt (Int.ofNat row) deltaList)) (Stage_keygen.generatedRoot.w_9_0 ctx))
    (matrixAdd selected (Stage_keygen.generatedRoot.w_26_2 ctx ⟨row, bound⟩))) 0 0 = _
  rw [deltaList_eq row bound, y, residual_eq]
  simp only [matrixSub, matrixAdd, matrixMul, matrixMulScalarLeft,
    liftInteger, Matrix.add_apply, Matrix.sub_apply, Matrix.one_apply_eq, one_mul,
    route_eq, ← compact_eq, ← identity_eq, selected_eq, helpers_eq, slotResidual]
  by_cases equal : row = 0 <;> simp only [equal, if_true, if_false, Int.cast_one, Int.cast_zero,
    one_mul, zero_mul, Matrix.sub_apply, Matrix.zero_apply] <;> abel

end PeProof
