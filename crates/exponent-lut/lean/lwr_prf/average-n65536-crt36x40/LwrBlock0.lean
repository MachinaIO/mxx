import LwrCells

/-!
# Online evaluation, block 0

Cell `i` of block `0` multiplies the state encoding at `Q36` by member `coords 0 i` of `a` (a public
multiplication at the native gadget of `Q36`), then by the selector of the cell (a private multiplication that
drops `P36`); the block sums its 113 cells.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxRecDepth 100000

variable {tape : SampleTape} {path loopPath cellPath : List Nat} {loopPaths : Nat → List Nat}

/-- The error bound of a block-0 cell for a state of error `B`. -/
abbrev cellBound0 (B : Nat) : Nat :=
  ((140 + 140) * N * ((144 + 144) * N * 26 * 512 + 1 * B) * 512 + (N + 2) * (P36 / 2)) / P36 +
    1 * 26

set_option maxHeartbeats 16000000 in
/-- Cell `i` of block `0`: an encoding of `P35 (α_c μ ν_i)` at `Q35`. -/
theorem cell0 {model : HashModel} {key : ByteArray} {ca0 ca1 : Fin 1025 → ExactMatrix Q36 N 1 144}
    {apay : Fin 1025 → ExactMatrix Q36 N 1 1} {As : Fin 2 → Fin 2 → ExactMatrix Q36 N 1 140}
    {cs : Fin 2 → ExactMatrix Q36 N 1 140} {bot0 bot1 : Fin 113 → ExactMatrix Q35 N 1 136} {i : Fin 113}
    {outputs}
    (hcell : Stage_online_eval.parallel_generatedRoot_23 Backend.backend model tape cellPath
      GeneratedClaim.stage_4_params i
      (ca0, key, ca1, As 0 0, As 1 0, cs 0, apay, As 0 1, As 1 1, cs 1, bot0 i, bot1 i, ()) outputs)
    {sm sp μ ν : ErrorPoly N} {α : Fin 1025 → ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    (hαμ : SmulGain (α ⟨coords 0 i, coords_lt 0 i⟩ * μ) 1) (hα : SmulGain (α ⟨coords 0 i, coords_lt 0 i⟩) 1)
    {G : ExactMatrix Q36 N 1 144} (hG : gadgetMatrixRuns Backend.backend 1024 144 G)
    {g' : ExactMatrix Q35 N 1 140} {g : ExactMatrix Q36 N 1 140}
    (hg' : gadgetMatrixRuns Backend.backend 1024 140 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q35 N 1 136}
    (ha : ∀ k : Fin 1025, Rel sm (memberKey model key k)
      (fun j ↦ reducePoly Q36 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] 26)
    (hpay : ∀ k : Fin 1025, apay k = fun _ _ ↦ reducePoly Q36 N (α k))
    {Bs : Nat} (hs : Rel sm As (fun j ↦ reducePoly Q36 N (((P36 : Nat) : Int) * μ * unitPair sp j) • g)
      cs Bs)
    {eB : Fin 2 → ErrorMatrix N 1 136}
    (hrhs : ∀ j, reducePoly Q35 N sp • rhsTop model key Q35 136 rhsPrefix0 i j + ![bot0 i, bot1 i] j =
      reducePoly Q35 N (((P35 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q35 N 1 136 (eB j))
    (heB : ∀ j, CoeffBound (eB j) 26) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q35 N (((P35 : Nat) : Int) * (α ⟨coords 0 i, coords_lt 0 i⟩ * μ * ν) *
        unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] (cellBound0 Bs) := by
  unfold Stage_online_eval.parallel_generatedRoot_23 at hcell
  try extract_lets at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_5_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_6_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_6_concat_1, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_8_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_10_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_11_concat_1, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_12_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_15_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_16_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_17_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_19_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_26_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_27_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_27_concat_1, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_30_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_31_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_32_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_34_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_38_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_39_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_40_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_40_concat_1, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_41_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_45_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_46_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_48_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_50_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_53_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_54_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_58_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_59_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_60_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_65_0, hcell⟩ := hcell
  try extract_lets at hcell
  have h2_0 := hcell.2.2.2.2.1
  have h4_0 := hcell.2.2.2.2.2.2.2.1
  have h5_0 := hcell.2.2.2.2.2.2.2.2.2.2.1
  have h4_0_1 := hcell.2.2.2.2.2.2.2.2.2.2.2.1
  have h6_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h8_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h9_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h10_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h9_0_1 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h11_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h12_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h15_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h16_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h17_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h19_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h22_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h25_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h26_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h25_0_1 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h27_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h30_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h31_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h32_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h34_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h38_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h39_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h38_0_1 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h40_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h41_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h42_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h44_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h45_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h46_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h48_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h49_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h50_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h52_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h53_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h54_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h56_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h57_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h58_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h59_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h60_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h61_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h62_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h63_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h64_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h65_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hout := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2

  -- The coordinate, the member of `a`, and its public key.
  have hc := selectInt_coords0 i
  rw [hc] at h4_0 h5_0 h9_0 h10_0
  obtain ⟨_, hw2⟩ := familyGet_eq (k := coords 0 i) (by rw [← hc]; exact h2_0)
  obtain ⟨_, hw8⟩ := familyGet_eq (k := coords 0 i) (by rw [← hc]; exact h8_0)
  obtain ⟨_, hw22⟩ := familyGet_eq (k := coords 0 i) (by rw [← hc]; exact h22_0)
  subst hw2 hw8 hw22
  have hkey : memberKey model key (coords 0 i) = ![![w_4_0, w_9_0], ![w_5_0, w_10_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h9_0).symm
    · exact (hashSample_eq h5_0).symm
    · exact (hashSample_eq h10_0).symm
  have ha' := ha ⟨coords 0 i, coords_lt 0 i⟩
  rw [hkey] at ha'
  -- The public multiplication at `Q36`.
  have hpub := mul_public_gain (q := Q36) (m := 144) (m' := 140) (bound := 512) hN hQ36
    Backend.layout35 layout36 rfl hG hα ha' hs h4_0_1 h6_0 h9_0_1 h11_0 h12_0
    (Xs := ![w_15_0, w_30_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h30_0])
    (D' := ![w_16_0, w_31_0]) (fun j ↦ by fin_cases j; exacts [h16_0, h31_0])
    (D := ![w_17_0, w_32_0]) (fun j ↦ by fin_cases j; exacts [h17_0, h32_0])
    (hpay _) (Y := ![w_19_0, w_34_0]) (A' := ![![w_25_0, w_38_0], ![w_26_0, w_39_0]])
    (fun j ↦ by fin_cases j; exacts [h19_0, h34_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h38_0])
    (fun j ↦ by fin_cases j; exacts [h26_0, h39_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q36 N (((P36 : Nat) : Int) *
      (α ⟨coords 0 i, coords_lt 0 i⟩ * μ) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `P36`.
  have htop : rhsTop model key Q35 136 rhsPrefix0 i = ![w_42_0, w_50_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h42_0).symm
    · exact (hashSample_eq h50_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down_gain (q := Q36) (q' := Q35) (p := P36) (m := 140) (m' := 136) (bound := 512)
    hN q36_eq hQ36 hQ35 (by norm_num) Backend.layout34 layout35 rfl hg' hg hsm hpub'
    h25_0_1 h27_0 h38_0_1 h40_0 h41_0 (top := ![w_42_0, w_50_0]) (bot := ![bot0 i, bot1 i])
    (Xb := ![w_44_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h52_0])
    (D' := ![w_45_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h53_0])
    (D := ![w_46_0, w_54_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h54_0])
    hrhs' heB hαμ (X := ![w_62_0, w_64_0]) (M := ![![w_48_0, w_56_0], ![w_58_0, w_60_0]])
    (fun j ↦ by fin_cases j; exacts [h62_0, h64_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h56_0])
    (fun j ↦ by fin_cases j; exacts [h58_0, h60_0])
    (c' := ![w_63_0, w_65_0]) (A' := ![![w_49_0, w_57_0], ![w_59_0, w_61_0]])
    (fun j ↦ by fin_cases j; exacts [h63_0, h65_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h49_0, h57_0, h59_0, h61_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring

set_option maxHeartbeats 4000000 in
/-- One iteration of the block-0 sum adds cell `2 + i` to every block. -/
theorem sum0_step {model : HashModel} {f0 f1 f2 f3 f4 f5 : Fin 113 → ExactMatrix Q35 N 1 136} {i : Nat}
    {current next : SumState Q35 136}
    (hbody : Stage_online_eval.sequential_generatedRoot_42 Backend.backend model tape loopPath
      GeneratedClaim.stage_4_params i
      (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
        current.2.2.2.2.2.1, f0, f1, f2, f3, f4, f5, ()) next) :
    next = (current.1 + cellAt f0 (2 + i), current.2.1 + cellAt f1 (2 + i),
      current.2.2.1 + cellAt f2 (2 + i), current.2.2.2.1 + cellAt f3 (2 + i),
      current.2.2.2.2.1 + cellAt f4 (2 + i), current.2.2.2.2.2.1 + cellAt f5 (2 + i), ()) := by
  unfold Stage_online_eval.sequential_generatedRoot_42 at hbody
  try extract_lets at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_15_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_19_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  try extract_lets at hbody
  have h3_0 := hbody.2.2.1
  have h7_0 := hbody.2.2.2.2.2.1
  have h11_0 := hbody.2.2.2.2.2.2.2.2.1
  have h15_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.1
  have h19_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h23_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hout := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2

  have e0 := familyGet_cell h3_0
  have e1 := familyGet_cell h7_0
  have e2 := familyGet_cell h11_0
  have e3 := familyGet_cell h15_0
  have e4 := familyGet_cell h19_0
  have e5 := familyGet_cell h23_0
  subst e0 e1 e2 e3 e4 e5
  rw [hout]
  rfl

/-- The generated block-0 sum adds cells `2, ..., count + 1` to its initial state. -/
theorem sum0 {model : HashModel} {f0 f1 f2 f3 f4 f5 : Fin 113 → ExactMatrix Q35 N 1 136} {count : Nat}
    {init final : SumState Q35 136}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : SumState Q35 136) ↦
      Stage_online_eval.sequential_generatedRoot_42 Backend.backend model tape (loopPaths i) GeneratedClaim.stage_4_params i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, f0, f1, f2, f3, f4, f5, ()) next) count init final) :
    final = addCells init f0 f1 f2 f3 f4 f5 count := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : SumState Q35 136) ↦ s = addCells init f0 f1 f2 f3 f4 f5 k) ?_ ?_ hruns
  · simp [addCells]
  · intro i current next hcur hbody
    rw [sum0_step hbody, hcur]
    simp only [addCells, Finset.sum_range_succ, add_assoc]


set_option maxHeartbeats 64000000 in
/-- Block `0`: the sum of its 113 cells, an encoding of `P35 Σ_i α_(c_i) μ ν_i` at `Q35`. -/
theorem block0 {model : HashModel} {key : ByteArray}
    {s0 s1 s2 s3 : Fin 128 → ExactMatrix Q1 N 1 4} {k0 k1 : Fin 128 → ExactMatrix Q2 N 1 4}
    {ca0 ca1 : Fin 1025 → ExactMatrix Q36 N 1 144} {cs0 cs1 : ExactMatrix Q36 N 1 140}
    {apay : Fin 1025 → ExactMatrix Q36 N 1 1}
    {r00 r01 : Fin 113 → ExactMatrix Q35 N 1 136} {r10 r11 : Fin 113 → ExactMatrix Q34 N 1 132}
    {r20 r21 : Fin 113 → ExactMatrix Q33 N 1 128} {r30 r31 : Fin 113 → ExactMatrix Q32 N 1 124}
    {r40 r41 : Fin 113 → ExactMatrix Q31 N 1 120} {r50 r51 : Fin 113 → ExactMatrix Q30 N 1 116}
    {r60 r61 : Fin 113 → ExactMatrix Q29 N 1 112} {r70 r71 : Fin 113 → ExactMatrix Q28 N 1 108}
    {r80 r81 : Fin 113 → ExactMatrix Q27 N 1 104} {r90 r91 : Fin 113 → ExactMatrix Q26 N 1 100}
    {r100 r101 : Fin 113 → ExactMatrix Q25 N 1 96} {r110 r111 : Fin 113 → ExactMatrix Q24 N 1 92}
    {r120 r121 : Fin 113 → ExactMatrix Q23 N 1 88} {r130 r131 : Fin 113 → ExactMatrix Q22 N 1 84}
    {r140 r141 : Fin 113 → ExactMatrix Q21 N 1 80} {r150 r151 : Fin 113 → ExactMatrix Q20 N 1 76}
    {r160 r161 : Fin 113 → ExactMatrix Q19 N 1 72} {r170 r171 : Fin 113 → ExactMatrix Q18 N 1 68}
    {r180 r181 : Fin 113 → ExactMatrix Q17 N 1 64} {r190 r191 : Fin 113 → ExactMatrix Q16 N 1 60}
    {r200 r201 : Fin 113 → ExactMatrix Q15 N 1 56} {r210 r211 : Fin 113 → ExactMatrix Q14 N 1 52}
    {r220 r221 : Fin 113 → ExactMatrix Q13 N 1 48} {r230 r231 : Fin 113 → ExactMatrix Q12 N 1 44}
    {r240 r241 : Fin 113 → ExactMatrix Q11 N 1 40} {r250 r251 : Fin 113 → ExactMatrix Q10 N 1 36}
    {r260 r261 : Fin 113 → ExactMatrix Q9 N 1 32} {r270 r271 : Fin 113 → ExactMatrix Q8 N 1 28}
    {r280 r281 : Fin 113 → ExactMatrix Q7 N 1 24} {r290 r291 : Fin 113 → ExactMatrix Q6 N 1 20}
    {r300 r301 : Fin 113 → ExactMatrix Q5 N 1 16} {r310 r311 : Fin 113 → ExactMatrix Q4 N 1 12}
    {r320 r321 : Fin 113 → ExactMatrix Q3 N 1 8} {r330 r331 : Fin 113 → ExactMatrix Q2 N 1 4}
    {outputs} {w : Stage_online_eval.generatedRoot.Witness}
    (hb : Stage_online_eval.generatedRoot.body Backend.backend model tape path GeneratedClaim.stage_4_params
      (s0, s1, s2, s3, k0, key, k1, ca0, ca1, cs0, apay, cs1, r00, r01, r10, r11, r20, r21, r30, r31,
        r40, r41, r50, r51, r60, r61, r70, r71,
        r80, r81, r90, r91, r100, r101, r110, r111,
        r120, r121, r130, r131, r140, r141, r150, r151,
        r160, r161, r170, r171, r180, r181, r190, r191,
        r200, r201, r210, r211, r220, r221, r230, r231,
        r240, r241, r250, r251, r260, r261, r270, r271,
        r280, r281, r290, r291, r300, r301, r310, r311,
        r320, r321, r330, r331, ()) outputs w)
    {sm sp μ : ErrorPoly N} {α : Fin 1025 → ErrorPoly N} {ν : Fin 113 → ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    (hαμ : ∀ i : Fin 113, SmulGain (α ⟨coords 0 i, coords_lt 0 i⟩ * μ) 1)
    (hα : ∀ k, SmulGain (α k) 1)
    {G : ExactMatrix Q36 N 1 144} (hG : gadgetMatrixRuns Backend.backend 1024 144 G)
    {g' : ExactMatrix Q35 N 1 140} {g : ExactMatrix Q36 N 1 140}
    (hg' : gadgetMatrixRuns Backend.backend 1024 140 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q35 N 1 136}
    (ha : ∀ k : Fin 1025, Rel sm (memberKey model key k)
      (fun j ↦ reducePoly Q36 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] 26)
    (hpay : ∀ k : Fin 1025, apay k = fun _ _ ↦ reducePoly Q36 N (α k))
    (hs : Rel sm (stateKey model key)
      (fun j ↦ reducePoly Q36 N (((P36 : Nat) : Int) * μ * unitPair sp j) • g) ![cs0, cs1] 26)
    (hrhs : ∀ i : Fin 113, ∃ e : Fin 2 → ErrorMatrix N 1 136, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q35 N sp • rhsTop model key Q35 136 rhsPrefix0 i j + ![r00 i, r01 i] j =
        reducePoly Q35 N (((P35 : Nat) : Int) * ν i * unitPair sp j) • h +
          reduceMatrix Q35 N 1 136 (e j)) :
    Rel sm ![![w.w_42_0, w.w_42_1], ![w.w_42_2, w.w_42_3]]
      (∑ i : Fin 113, fun j ↦ reducePoly Q35 N (((P35 : Nat) : Int) *
        (α ⟨coords 0 i, coords_lt 0 i⟩ * μ * ν i) * unitPair sp j) • h)
      ![w.w_42_4, w.w_42_5] (∑ _i : Fin 113, cellBound0 26) := by
  unfold Stage_online_eval.generatedRoot.body at hb
  extract_lets at hb
  have h14_0 := hb.2.2.2.2.1
  have h15_0 := hb.2.2.2.2.2.1
  have h18_0 := hb.2.2.2.2.2.2.1
  have h19_0 := hb.2.2.2.2.2.2.2.1
  have h23_5 := hb.2.2.2.2.2.2.2.2.2.1
  have h24_0 := hb.2.2.2.2.2.2.2.2.2.2.1
  have h25_0 := hb.2.2.2.2.2.2.2.2.2.2.2.1
  have h27_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h28_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h30_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h31_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h33_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h34_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h36_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h37_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h39_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h40_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h42_5 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  -- The state key.
  have hkey : stateKey model key = ![![w.w_14_0, w.w_18_0], ![w.w_15_0, w.w_19_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h14_0).symm
    · exact (hashSample_eq h18_0).symm
    · exact (hashSample_eq h15_0).symm
    · exact (hashSample_eq h19_0).symm
  rw [hkey] at hs
  -- Every cell, then their sum.
  have hcells : ∀ i : Fin 113, Rel sm ![![w.w_23_0 i, w.w_23_1 i], ![w.w_23_2 i, w.w_23_3 i]]
      (fun j ↦ reducePoly Q35 N (((P35 : Nat) : Int) * (α ⟨coords 0 i, coords_lt 0 i⟩ * μ * ν i) *
        unitPair sp j) • h) ![w.w_23_4 i, w.w_23_5 i] (cellBound0 26) := by
    intro i
    obtain ⟨e, he, hr⟩ := hrhs i
    exact cell0 (h23_5 i) hsm (hαμ i) (hα _) hG hg' hg ha hpay hs hr he
  have hsum := Rel.sum Finset.univ (fun i _ ↦ hcells i)
  -- The generated sum is the sum of the 113 cells.
  obtain ⟨_, e24⟩ := familyGetStatic_eq (k := 0) h24_0
  replace e24 : w.w_24_0 = w.w_23_0 ⟨0, by decide⟩ := e24
  obtain ⟨_, e25⟩ := familyGetStatic_eq (k := 1) h25_0
  replace e25 : w.w_25_0 = w.w_23_0 ⟨1, by decide⟩ := e25
  obtain ⟨_, e27⟩ := familyGetStatic_eq (k := 0) h27_0
  replace e27 : w.w_27_0 = w.w_23_1 ⟨0, by decide⟩ := e27
  obtain ⟨_, e28⟩ := familyGetStatic_eq (k := 1) h28_0
  replace e28 : w.w_28_0 = w.w_23_1 ⟨1, by decide⟩ := e28
  obtain ⟨_, e30⟩ := familyGetStatic_eq (k := 0) h30_0
  replace e30 : w.w_30_0 = w.w_23_2 ⟨0, by decide⟩ := e30
  obtain ⟨_, e31⟩ := familyGetStatic_eq (k := 1) h31_0
  replace e31 : w.w_31_0 = w.w_23_2 ⟨1, by decide⟩ := e31
  obtain ⟨_, e33⟩ := familyGetStatic_eq (k := 0) h33_0
  replace e33 : w.w_33_0 = w.w_23_3 ⟨0, by decide⟩ := e33
  obtain ⟨_, e34⟩ := familyGetStatic_eq (k := 1) h34_0
  replace e34 : w.w_34_0 = w.w_23_3 ⟨1, by decide⟩ := e34
  obtain ⟨_, e36⟩ := familyGetStatic_eq (k := 0) h36_0
  replace e36 : w.w_36_0 = w.w_23_4 ⟨0, by decide⟩ := e36
  obtain ⟨_, e37⟩ := familyGetStatic_eq (k := 1) h37_0
  replace e37 : w.w_37_0 = w.w_23_4 ⟨1, by decide⟩ := e37
  obtain ⟨_, e39⟩ := familyGetStatic_eq (k := 0) h39_0
  replace e39 : w.w_39_0 = w.w_23_5 ⟨0, by decide⟩ := e39
  obtain ⟨_, e40⟩ := familyGetStatic_eq (k := 1) h40_0
  replace e40 : w.w_40_0 = w.w_23_5 ⟨1, by decide⟩ := e40
  have hloop := sum0 (count := 111) (init := (
      w.w_24_0 + w.w_25_0,
      w.w_27_0 + w.w_28_0,
      w.w_30_0 + w.w_31_0,
      w.w_33_0 + w.w_34_0,
      w.w_36_0 + w.w_37_0,
      w.w_39_0 + w.w_40_0, ()))
    (final := (w.w_42_0, w.w_42_1, w.w_42_2, w.w_42_3, w.w_42_4, w.w_42_5, ())) h42_5
  simp only [addCells, Prod.mk.injEq] at hloop
  obtain ⟨l0, l1, l2, l3, l4, l5, -⟩ := hloop
  have m0 : w.w_42_0 = ∑ i : Fin 113, w.w_23_0 i := by rw [l0, e24, e25]; exact sum_cells _
  have m1 : w.w_42_1 = ∑ i : Fin 113, w.w_23_1 i := by rw [l1, e27, e28]; exact sum_cells _
  have m2 : w.w_42_2 = ∑ i : Fin 113, w.w_23_2 i := by rw [l2, e30, e31]; exact sum_cells _
  have m3 : w.w_42_3 = ∑ i : Fin 113, w.w_23_3 i := by rw [l3, e33, e34]; exact sum_cells _
  have m4 : w.w_42_4 = ∑ i : Fin 113, w.w_23_4 i := by rw [l4, e36, e37]; exact sum_cells _
  have m5 : w.w_42_5 = ∑ i : Fin 113, w.w_23_5 i := by rw [l5, e39, e40]; exact sum_cells _
  have hA : (∑ i : Fin 113, ![![w.w_23_0 i, w.w_23_1 i], ![w.w_23_2 i, w.w_23_3 i]]) =
      ![![w.w_42_0, w.w_42_1], ![w.w_42_2, w.w_42_3]] := by
    funext k j
    fin_cases k <;> fin_cases j <;> simp [Finset.sum_apply, m0, m1, m2, m3]
  have hc : (∑ i : Fin 113, ![w.w_23_4 i, w.w_23_5 i]) = ![w.w_42_4, w.w_42_5] := by
    funext j
    fin_cases j <;> simp [Finset.sum_apply, m4, m5]
  rw [hA, hc] at hsum
  exact hsum

end LwrProof
