import LwrPreprocess

/-!
# Online evaluation, block 0

Cell `i` of block `0` multiplies the state encoding at `Q7` by member `coords 0 i` of `a` (a public
multiplication at the native gadget of `Q7`), then by the selector of the cell (a private
multiplication that drops `P7`); the block sums its fourteen cells.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The error bound of a block-0 cell for a state of error `B`. -/
abbrev cellBound0 (B : Nat) : Nat :=
  ((12 + 12) * N * ((14 + 14) * N * 26 * 524288 + N * 1 * B) * 524288 + (N + 2) * (P7 / 2)) / P7 +
    N * 1 * 26

set_option maxHeartbeats 16000000 in
/-- Cell `i` of block `0`: an encoding of `P6 (α_c μ ν_i)` at `Q6`. -/
theorem cell0 {model : HashModel} {key : ByteArray} {ca0 ca1 : Fin 17 → ExactMatrix Q7 N 1 14}
    {apay : Fin 17 → ExactMatrix Q7 N 1 1} {As : Fin 2 → Fin 2 → ExactMatrix Q7 N 1 12}
    {cs : Fin 2 → ExactMatrix Q7 N 1 12} {bot0 bot1 : Fin 14 → ExactMatrix Q6 N 1 10} {i : Fin 14}
    {outputs}
    (hcell : Stage_online_eval.parallel_generatedRoot_23 Backend.backend model
      GeneratedClaim.stage_4_params i
      (ca0, key, ca1, As 0 0, As 1 0, cs 0, apay, As 0 1, As 1 1, cs 1, bot0 i, bot1 i, ()) outputs)
    {sm sp μ ν : ErrorPoly N} {α : Fin 17 → ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    (hαμ : polyNorm (α ⟨coords 0 i, coords_lt 0 i⟩ * μ) ≤ 1) (hα : polyNorm (α ⟨coords 0 i, coords_lt 0 i⟩) ≤ 1)
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 1048576 14 G)
    {g' : ExactMatrix Q6 N 1 12} {g : ExactMatrix Q7 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 12 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q6 N 1 10}
    (ha : ∀ k : Fin 17, Rel sm (memberKey model key k)
      (fun j ↦ reducePoly Q7 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] 26)
    (hpay : ∀ k : Fin 17, apay k = fun _ _ ↦ reducePoly Q7 N (α k))
    {Bs : Nat} (hs : Rel sm As (fun j ↦ reducePoly Q7 N (((P7 : Nat) : Int) * μ * unitPair sp j) • g)
      cs Bs)
    {eB : Fin 2 → ErrorMatrix N 1 10}
    (hrhs : ∀ j, reducePoly Q6 N sp • rhsTop model key Q6 10 rhsPrefix0 i j + ![bot0 i, bot1 i] j =
      reducePoly Q6 N (((P6 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q6 N 1 10 (eB j))
    (heB : ∀ j, CoeffBound (eB j) 26) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) * (α ⟨coords 0 i, coords_lt 0 i⟩ * μ * ν) *
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
  -- The public multiplication at `Q7`.
  have hpub := mul_public (q := Q7) (m := 14) (m' := 12) (bound := 524288) hN hQ7
    Backend.layout6 layout7 rfl hG hα ha' hs h4_0_1 h6_0 h9_0_1 h11_0 h12_0
    (Xs := ![w_15_0, w_30_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h30_0])
    (D' := ![w_16_0, w_31_0]) (fun j ↦ by fin_cases j; exacts [h16_0, h31_0])
    (D := ![w_17_0, w_32_0]) (fun j ↦ by fin_cases j; exacts [h17_0, h32_0])
    (hpay _) (Y := ![w_19_0, w_34_0]) (A' := ![![w_25_0, w_38_0], ![w_26_0, w_39_0]])
    (fun j ↦ by fin_cases j; exacts [h19_0, h34_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h38_0])
    (fun j ↦ by fin_cases j; exacts [h26_0, h39_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q7 N (((P7 : Nat) : Int) *
      (α ⟨coords 0 i, coords_lt 0 i⟩ * μ) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `P7`.
  have htop : rhsTop model key Q6 10 rhsPrefix0 i = ![w_42_0, w_50_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h42_0).symm
    · exact (hashSample_eq h50_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q7) (q' := Q6) (p := P7) (m := 12) (m' := 10) (bound := 524288)
    hN q7_eq hQ7 hQ6 (by norm_num) Backend.layout5 layout6 rfl hg' hg hsm hpub'
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

/-- The six blocks `(A₀₀, A₀₁, A₁₀, A₁₁, c₀, c₁)` of a summed encoding. -/
abbrev SumState (q m : Nat) :=
  ExactMatrix q N 1 m × ExactMatrix q N 1 m × ExactMatrix q N 1 m × ExactMatrix q N 1 m ×
    ExactMatrix q N 1 m × ExactMatrix q N 1 m × Unit

/-- Member `k` of a family of fourteen cells, zero past the family. -/
noncomputable def cellAt {q m : Nat} (f : Fin 14 → ExactMatrix q N 1 m) (k : Nat) :
    ExactMatrix q N 1 m :=
  if h : k < 14 then f ⟨k, h⟩ else 0

/-- Cells `2, ..., k + 1` added to an initial state. -/
noncomputable def addCells {q m : Nat} (init : SumState q m)
    (f0 f1 f2 f3 f4 f5 : Fin 14 → ExactMatrix q N 1 m) (k : Nat) : SumState q m :=
  (init.1 + ∑ l ∈ Finset.range k, cellAt f0 (2 + l),
    init.2.1 + ∑ l ∈ Finset.range k, cellAt f1 (2 + l),
    init.2.2.1 + ∑ l ∈ Finset.range k, cellAt f2 (2 + l),
    init.2.2.2.1 + ∑ l ∈ Finset.range k, cellAt f3 (2 + l),
    init.2.2.2.2.1 + ∑ l ∈ Finset.range k, cellAt f4 (2 + l),
    init.2.2.2.2.2.1 + ∑ l ∈ Finset.range k, cellAt f5 (2 + l), ())

/-- A cell read at the shifted index `2 + i`. -/
theorem familyGet_cell {q m : Nat} {f : Fin 14 → ExactMatrix q N 1 m} {i : Nat} {out : ExactMatrix q N 1 m}
    (h : familyGetDynamic f (2 + Int.ofNat i) out) : out = cellAt f (2 + i) := by
  obtain ⟨hk, hout⟩ := familyGet_eq (k := 2 + i) (by
    rw [show (Int.ofNat (2 + i)) = 2 + Int.ofNat i by simp only [Int.ofNat_eq_natCast]; push_cast; ring]
    exact h)
  rw [hout, cellAt, dif_pos hk]

set_option maxHeartbeats 4000000 in
/-- One iteration of the block-0 sum adds cell `2 + i` to every block. -/
theorem sum0_step {model : HashModel} {f0 f1 f2 f3 f4 f5 : Fin 14 → ExactMatrix Q6 N 1 10} {i : Nat}
    {current next : SumState Q6 10}
    (hbody : Stage_online_eval.sequential_generatedRoot_42 Backend.backend model
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
theorem sum0 {model : HashModel} {f0 f1 f2 f3 f4 f5 : Fin 14 → ExactMatrix Q6 N 1 10} {count : Nat}
    {init final : SumState Q6 10}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : SumState Q6 10) ↦
      Stage_online_eval.sequential_generatedRoot_42 Backend.backend model GeneratedClaim.stage_4_params i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, f0, f1, f2, f3, f4, f5, ()) next) count init final) :
    final = addCells init f0 f1 f2 f3 f4 f5 count := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : SumState Q6 10) ↦ s = addCells init f0 f1 f2 f3 f4 f5 k) ?_ ?_ hruns
  · simp [addCells]
  · intro i current next hcur hbody
    rw [sum0_step hbody, hcur]
    simp only [addCells, Finset.sum_range_succ, add_assoc]

/-- Cells `0` and `1` plus cells `2, ..., 13` are all fourteen cells. -/
theorem sum_cells {q m : Nat} (f : Fin 14 → ExactMatrix q N 1 m) :
    f ⟨0, by decide⟩ + f ⟨1, by decide⟩ + ∑ l ∈ Finset.range 12, cellAt f (2 + l) =
      ∑ i : Fin 14, f i := by
  have hall : ∑ i : Fin 14, f i = ∑ k ∈ Finset.range 14, cellAt f k := by
    rw [← Fin.sum_univ_eq_sum_range]
    apply Finset.sum_congr rfl
    intro i _
    rw [cellAt, dif_pos i.isLt]
  have h := Finset.sum_range_add (fun k ↦ cellAt f k) 2 12
  rw [show (2 + 12 : Nat) = 14 from rfl] at h
  rw [hall, h]
  congr 1
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add]
  rfl

/-- A family read at a static index. -/
theorem familyGetStatic_eq {α : Type} {count : Nat} {f : Fin count → α} {k : Nat} {out : α}
    (h : familyGetStatic f (Int.ofNat k) out) : ∃ hk : k < count, out = f ⟨k, hk⟩ :=
  familyGet_eq h

set_option maxHeartbeats 64000000 in
/-- Block `0`: the sum of its fourteen cells, an encoding of `P6 Σ_i α_(c_i) μ ν_i` at `Q6`. -/
theorem block0 {model : HashModel} {key : ByteArray}
    {s0 s1 s2 s3 : Fin 16 → ExactMatrix Q1 N 1 2} {k0 k1 : Fin 16 → ExactMatrix Q2 N 1 2}
    {ca0 ca1 : Fin 17 → ExactMatrix Q7 N 1 14} {cs0 cs1 : ExactMatrix Q7 N 1 12}
    {apay : Fin 17 → ExactMatrix Q7 N 1 1}
    {r00 r01 : Fin 14 → ExactMatrix Q6 N 1 10} {r10 r11 : Fin 14 → ExactMatrix Q5 N 1 8}
    {r20 r21 : Fin 14 → ExactMatrix Q4 N 1 6} {r30 r31 : Fin 14 → ExactMatrix Q3 N 1 4}
    {r40 r41 : Fin 14 → ExactMatrix Q2 N 1 2} {outputs} {w : Stage_online_eval.generatedRoot.Witness}
    (hb : Stage_online_eval.generatedRoot.body Backend.backend model GeneratedClaim.stage_4_params
      (s0, s1, s2, s3, k0, key, k1, ca0, ca1, cs0, apay, cs1, r00, r01, r10, r11, r20, r21, r30, r31,
        r40, r41, ()) outputs w)
    {sm sp μ : ErrorPoly N} {α : Fin 17 → ErrorPoly N} {ν : Fin 14 → ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    (hαμ : ∀ i : Fin 14, polyNorm (α ⟨coords 0 i, coords_lt 0 i⟩ * μ) ≤ 1)
    (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 1048576 14 G)
    {g' : ExactMatrix Q6 N 1 12} {g : ExactMatrix Q7 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 12 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q6 N 1 10}
    (ha : ∀ k : Fin 17, Rel sm (memberKey model key k)
      (fun j ↦ reducePoly Q7 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] 26)
    (hpay : ∀ k : Fin 17, apay k = fun _ _ ↦ reducePoly Q7 N (α k))
    (hs : Rel sm (stateKey model key)
      (fun j ↦ reducePoly Q7 N (((P7 : Nat) : Int) * μ * unitPair sp j) • g) ![cs0, cs1] 26)
    (hrhs : ∀ i : Fin 14, ∃ e : Fin 2 → ErrorMatrix N 1 10, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q6 N sp • rhsTop model key Q6 10 rhsPrefix0 i j + ![r00 i, r01 i] j =
        reducePoly Q6 N (((P6 : Nat) : Int) * ν i * unitPair sp j) • h + reduceMatrix Q6 N 1 10 (e j)) :
    Rel sm ![![w.w_42_0, w.w_42_1], ![w.w_42_2, w.w_42_3]]
      (∑ i : Fin 14, fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) *
        (α ⟨coords 0 i, coords_lt 0 i⟩ * μ * ν i) * unitPair sp j) • h)
      ![w.w_42_4, w.w_42_5] (∑ _i : Fin 14, cellBound0 26) := by
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
  have hcells : ∀ i : Fin 14, Rel sm ![![w.w_23_0 i, w.w_23_1 i], ![w.w_23_2 i, w.w_23_3 i]]
      (fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) * (α ⟨coords 0 i, coords_lt 0 i⟩ * μ * ν i) *
        unitPair sp j) • h) ![w.w_23_4 i, w.w_23_5 i] (cellBound0 26) := by
    intro i
    obtain ⟨e, he, hr⟩ := hrhs i
    exact cell0 (h23_5 i) hsm (hαμ i) (hα _) hG hg' hg ha hpay hs hr he
  have hsum := Rel.sum Finset.univ (fun i _ ↦ hcells i)
  -- The generated sum is the sum of the fourteen cells.
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
  have hloop := sum0 (count := 12) (init := (w.w_24_0 + w.w_25_0, w.w_27_0 + w.w_28_0,
    w.w_30_0 + w.w_31_0, w.w_33_0 + w.w_34_0, w.w_36_0 + w.w_37_0, w.w_39_0 + w.w_40_0, ()))
    (final := (w.w_42_0, w.w_42_1, w.w_42_2, w.w_42_3, w.w_42_4, w.w_42_5, ())) h42_5
  simp only [addCells, Prod.mk.injEq] at hloop
  obtain ⟨l0, l1, l2, l3, l4, l5, -⟩ := hloop
  have m0 : w.w_42_0 = ∑ i : Fin 14, w.w_23_0 i := by rw [l0, e24, e25]; exact sum_cells _
  have m1 : w.w_42_1 = ∑ i : Fin 14, w.w_23_1 i := by rw [l1, e27, e28]; exact sum_cells _
  have m2 : w.w_42_2 = ∑ i : Fin 14, w.w_23_2 i := by rw [l2, e30, e31]; exact sum_cells _
  have m3 : w.w_42_3 = ∑ i : Fin 14, w.w_23_3 i := by rw [l3, e33, e34]; exact sum_cells _
  have m4 : w.w_42_4 = ∑ i : Fin 14, w.w_23_4 i := by rw [l4, e36, e37]; exact sum_cells _
  have m5 : w.w_42_5 = ∑ i : Fin 14, w.w_23_5 i := by rw [l5, e39, e40]; exact sum_cells _
  have hA : (∑ i : Fin 14, ![![w.w_23_0 i, w.w_23_1 i], ![w.w_23_2 i, w.w_23_3 i]]) =
      ![![w.w_42_0, w.w_42_1], ![w.w_42_2, w.w_42_3]] := by
    funext k j
    fin_cases k <;> fin_cases j <;> simp [Finset.sum_apply, m0, m1, m2, m3]
  have hc : (∑ i : Fin 14, ![w.w_23_4 i, w.w_23_5 i]) = ![w.w_42_4, w.w_42_5] := by
    funext j
    fin_cases j <;> simp [Finset.sum_apply, m4, m5]
  rw [hA, hc] at hsum
  exact hsum

end LwrProof
