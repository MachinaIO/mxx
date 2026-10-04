import LwrBlock0

/-!
# Online evaluation, block 1

Cell `i` of block `1` aligns member `coords 1 i` of `a` from `Q7` to `Q6`, multiplies the state
encoding at `Q6` by it, then by the selector of the cell, dropping `P6`; the block sums
its fourteen cells.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The error bound of a member of `a` aligned from `Q7` to `Q6`. -/
abbrev alignBound1 : Nat := (14 * N * 26 * ((1048576 : Int).toNat / 2) + (N + 2) * (A6 / 2)) / A6

/-- The error bound of a block-1 cell for a state of error `B`. -/
abbrev cellBound1 (B : Nat) : Nat :=
  ((10 + 10) * N * ((12 + 12) * N * alignBound1 * 524288 + N * 1 * B) * 524288 +
    (N + 2) * (P6 / 2)) / P6 + N * 1 * 26

set_option maxHeartbeats 16000000 in
/-- Cell `i` of block `1`: an encoding of `P5 (α_c μ ν_i)` at `Q5`. -/
theorem cell1 {model : HashModel} {key : ByteArray} {ca0 ca1 : Fin 17 → ExactMatrix Q7 N 1 14}
    {apay : Fin 17 → ExactMatrix Q7 N 1 1} {As : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10}
    {cs : Fin 2 → ExactMatrix Q6 N 1 10} {bot0 bot1 : Fin 14 → ExactMatrix Q5 N 1 8} {i : Fin 14}
    {outputs}
    (hcell : Stage_online_eval.parallel_generatedRoot_45 Backend.backend model
      GeneratedClaim.stage_4_params i
      (ca0, key, ca1, As 0 0, As 1 0, cs 0, apay, As 0 1, As 1 1, cs 1, bot0 i, bot1 i, ()) outputs)
    {sm sp μ ν : ErrorPoly N} {α : Fin 17 → ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    (hαμ : polyNorm (α ⟨coords 1 i, coords_lt 1 i⟩ * μ) ≤ 1)
    (hα : polyNorm (α ⟨coords 1 i, coords_lt 1 i⟩) ≤ 1)
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 1048576 14 G)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 10 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q5 N 1 8}
    (ha : ∀ k : Fin 17, Rel sm (memberKey model key k)
      (fun j ↦ reducePoly Q7 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] 26)
    (hpay : ∀ k : Fin 17, apay k = fun _ _ ↦ reducePoly Q7 N (α k))
    {Bs : Nat} (hs : Rel sm As (fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) * μ * unitPair sp j) • g)
      cs Bs)
    {eB : Fin 2 → ErrorMatrix N 1 8}
    (hrhs : ∀ j, reducePoly Q5 N sp • rhsTop model key Q5 8 rhsPrefix1 i j + ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((P5 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q5 N 1 8 (eB j))
    (heB : ∀ j, CoeffBound (eB j) 26) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q5 N (((P5 : Nat) : Int) * (α ⟨coords 1 i, coords_lt 1 i⟩ * μ * ν) *
        unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] (cellBound1 Bs) := by
  unfold Stage_online_eval.parallel_generatedRoot_45 at hcell
  try extract_lets at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_5_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_16_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_19_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_21_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_22_concat_1, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_24_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_26_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_27_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_30_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_32_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_33_concat_1, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_34_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_38_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_39_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_41_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_45_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_48_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_50_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_50_concat_1, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_53_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_54_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_55_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_63_concat_1, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_65_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_67_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_69_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_71_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_72_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_73_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_75_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_76_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_77_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_79_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_80_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_81_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_82_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_83_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_84_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_85_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_86_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_87_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_88_0, hcell⟩ := hcell
  try extract_lets at hcell
  have h2_0 := hcell.2.2.2.2.1
  have h4_0 := hcell.2.2.2.2.2.1
  have h5_0 := hcell.2.2.2.2.2.2.1
  have h7_0 := hcell.2.2.2.2.2.2.2.1
  have h9_0 := hcell.2.2.2.2.2.2.2.2.1
  have h11_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.1
  have h13_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h14_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h16_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h18_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h19_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h21_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h18_0_1 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h22_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h24_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h26_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h27_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h29_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h30_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h32_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h29_0_1 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h33_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h34_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h37_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h38_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h39_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h41_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h44_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h45_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h48_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h49_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h48_0_1 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h50_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h53_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h54_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h55_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h57_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h61_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h62_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h61_0_1 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h63_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h64_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h65_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h67_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h68_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h69_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h71_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h72_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h73_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h75_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h76_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h77_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h79_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h80_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h81_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h82_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h83_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h84_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h85_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h86_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h87_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h88_0 := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hout := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2

  -- The coordinate, the member of `a`, and its public key.
  have hc := selectInt_coords1 i
  rw [hc] at h11_0 h19_0 h27_0 h30_0
  obtain ⟨_, hw2⟩ := familyGet_eq (k := coords 1 i) (by rw [← hc]; exact h2_0)
  obtain ⟨_, hw24⟩ := familyGet_eq (k := coords 1 i) (by rw [← hc]; exact h24_0)
  obtain ⟨_, hw44⟩ := familyGet_eq (k := coords 1 i) (by rw [← hc]; exact h44_0)
  subst hw2 hw24 hw44
  have hkey : memberKey model key (coords 1 i) = ![![w_11_0, w_27_0], ![w_19_0, w_30_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h27_0).symm
    · exact (hashSample_eq h19_0).symm
    · exact (hashSample_eq h30_0).symm
  have ha' := ha ⟨coords 1 i, coords_lt 1 i⟩
  rw [hkey] at ha'
  -- The alignment of the member from `Q7` to `Q6`; the row and the key share one decomposition.
  have hg13 := gadgetMatrixRuns_unique h13_0 h4_0
  subst hg13
  have hg14 := centeredRebaseRuns_unique h14_0 h5_0
  subst hg14
  have hH := gadgetDecomposeRuns_unique h16_0 h7_0
  subst hH
  have hal := align_down (q := Q7) (q' := Q6) (p := A6) (m := 14) (m' := 12) hN q7_eq6 hQ7 hQ6
    (by norm_num) Backend.layout6 layout7 rfl hG h5_0 hsm ha'
    (by
      funext r c
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp (config := { zetaDelta := true }) [Matrix.mul_apply, matrixPolynomial_single, map_ofNat])
    h7_0 (Y := ![_, _]) (K := ![![_, _], ![_, _]])
    (fun j ↦ by fin_cases j <;> rfl) (fun k j ↦ by fin_cases k <;> fin_cases j <;> rfl)
    (c' := ![w_9_0, w_26_0]) (A' := ![![w_18_0, w_29_0], ![w_21_0, w_32_0]])
    (fun j ↦ by fin_cases j; exacts [h9_0, h26_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h18_0, h29_0, h21_0, h32_0])
  -- The public multiplication at `Q6`.
  have hpayb := modulusReduce_poly hQ7 hQ6 hN (s := α ⟨coords 1 i, coords_lt 1 i⟩)
    (by rw [← hpay]; exact h45_0)
  have hpub := mul_public (q := Q6) (m := 12) (m' := 10) (bound := 524288) hN hQ6
    Backend.layout5 layout6 rfl h4_0 hα hal hs h18_0_1 h22_0 h29_0_1 h33_0 h34_0
    (Xs := ![w_37_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h37_0, h53_0])
    (D' := ![w_38_0, w_54_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h54_0])
    (D := ![w_39_0, w_55_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h55_0])
    hpayb (Y := ![w_41_0, w_57_0]) (A' := ![![w_48_0, w_61_0], ![w_49_0, w_62_0]])
    (fun j ↦ by fin_cases j; exacts [h41_0, h57_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h61_0])
    (fun j ↦ by fin_cases j; exacts [h49_0, h62_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) *
      (α ⟨coords 1 i, coords_lt 1 i⟩ * μ) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `P6`.
  have htop : rhsTop model key Q5 8 rhsPrefix1 i = ![w_65_0, w_73_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h65_0).symm
    · exact (hashSample_eq h73_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q6) (q' := Q5) (p := P6) (m := 10) (m' := 8)
    (bound := 524288) hN q6_eq hQ6 hQ5 (by norm_num) Backend.layout4 layout5 rfl hg' hg hsm hpub'
    h48_0_1 h50_0 h61_0_1 h63_0 h64_0 (top := ![w_65_0, w_73_0]) (bot := ![bot0 i, bot1 i])
    (Xb := ![w_67_0, w_75_0]) (fun j ↦ by fin_cases j; exacts [h67_0, h75_0])
    (D' := ![w_68_0, w_76_0]) (fun j ↦ by fin_cases j; exacts [h68_0, h76_0])
    (D := ![w_69_0, w_77_0]) (fun j ↦ by fin_cases j; exacts [h69_0, h77_0])
    hrhs' heB hαμ (X := ![w_85_0, w_87_0]) (M := ![![w_71_0, w_79_0], ![w_81_0, w_83_0]])
    (fun j ↦ by fin_cases j; exacts [h85_0, h87_0]) (fun j ↦ by fin_cases j; exacts [h71_0, h79_0])
    (fun j ↦ by fin_cases j; exacts [h81_0, h83_0])
    (c' := ![w_86_0, w_88_0]) (A' := ![![w_72_0, w_80_0], ![w_82_0, w_84_0]])
    (fun j ↦ by fin_cases j; exacts [h86_0, h88_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h72_0, h80_0, h82_0, h84_0])
  subst hout
  refine hpriv.target_eq ?_
  funext j
  rw [smul_smul, ← map_mul]
  congr 2
  ring

set_option maxHeartbeats 4000000 in
/-- One iteration of the block-1 sum adds cell `2 + i` to every block. -/
theorem sum1_step {model : HashModel} {f0 f1 f2 f3 f4 f5 : Fin 14 → ExactMatrix Q5 N 1 8} {i : Nat}
    {current next : SumState Q5 8}
    (hbody : Stage_online_eval.sequential_generatedRoot_64 Backend.backend model
      GeneratedClaim.stage_4_params i
      (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
        current.2.2.2.2.2.1, f0, f1, f2, f3, f4, f5, ()) next) :
    next = (current.1 + cellAt f0 (2 + i), current.2.1 + cellAt f1 (2 + i),
      current.2.2.1 + cellAt f2 (2 + i), current.2.2.2.1 + cellAt f3 (2 + i),
      current.2.2.2.2.1 + cellAt f4 (2 + i), current.2.2.2.2.2.1 + cellAt f5 (2 + i), ()) := by
  unfold Stage_online_eval.sequential_generatedRoot_64 at hbody
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

/-- The generated block-1 sum adds cells `2, ..., count + 1` to its initial state. -/
theorem sum1 {model : HashModel} {f0 f1 f2 f3 f4 f5 : Fin 14 → ExactMatrix Q5 N 1 8} {count : Nat}
    {init final : SumState Q5 8}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : SumState Q5 8) ↦
      Stage_online_eval.sequential_generatedRoot_64 Backend.backend model GeneratedClaim.stage_4_params i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, f0, f1, f2, f3, f4, f5, ()) next) count init final) :
    final = addCells init f0 f1 f2 f3 f4 f5 count := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : SumState Q5 8) ↦ s = addCells init f0 f1 f2 f3 f4 f5 k) ?_ ?_ hruns
  · simp [addCells]
  · intro i current next hcur hbody
    rw [sum1_step hbody, hcur]
    simp only [addCells, Finset.sum_range_succ, add_assoc]


set_option maxHeartbeats 64000000 in
/-- Block `1`: the sum of its fourteen cells, an encoding of `P5 Σ_i α_(c_i) μ ν_i` at `Q5`. -/
theorem block1 {model : HashModel} {key : ByteArray}
    {s0 s1 s2 s3 : Fin 16 → ExactMatrix Q1 N 1 1} {k0 k1 : Fin 16 → ExactMatrix Q2 N 1 2}
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
    (hαμ : ∀ i : Fin 14, polyNorm (α ⟨coords 1 i, coords_lt 1 i⟩ * μ) ≤ 1)
    (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 1048576 14 G)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 10 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q5 N 1 8}
    (ha : ∀ k : Fin 17, Rel sm (memberKey model key k)
      (fun j ↦ reducePoly Q7 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] 26)
    (hpay : ∀ k : Fin 17, apay k = fun _ _ ↦ reducePoly Q7 N (α k))
    {Bs : Nat} (hs : Rel sm ![![w.w_42_0, w.w_42_1], ![w.w_42_2, w.w_42_3]]
      (fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) * μ * unitPair sp j) • g)
      ![w.w_42_4, w.w_42_5] Bs)
    (hrhs : ∀ i : Fin 14, ∃ e : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q5 N sp • rhsTop model key Q5 8 rhsPrefix1 i j + ![r10 i, r11 i] j =
        reducePoly Q5 N (((P5 : Nat) : Int) * ν i * unitPair sp j) • h +
          reduceMatrix Q5 N 1 8 (e j)) :
    Rel sm ![![w.w_64_0, w.w_64_1], ![w.w_64_2, w.w_64_3]]
      (∑ i : Fin 14, fun j ↦ reducePoly Q5 N (((P5 : Nat) : Int) *
        (α ⟨coords 1 i, coords_lt 1 i⟩ * μ * ν i) * unitPair sp j) • h)
      ![w.w_64_4, w.w_64_5] (∑ _i : Fin 14, cellBound1 Bs) := by
  unfold Stage_online_eval.generatedRoot.body at hb
  extract_lets at hb
  have h45_5 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h46_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h47_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h49_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h50_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h52_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h53_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h55_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h56_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h58_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h59_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h61_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h62_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h64_5 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  -- Every cell, then their sum.
  have hcells : ∀ i : Fin 14, Rel sm ![![w.w_45_0 i, w.w_45_1 i], ![w.w_45_2 i, w.w_45_3 i]]
      (fun j ↦ reducePoly Q5 N (((P5 : Nat) : Int) * (α ⟨coords 1 i, coords_lt 1 i⟩ * μ * ν i) *
        unitPair sp j) • h) ![w.w_45_4 i, w.w_45_5 i] (cellBound1 Bs) := by
    intro i
    obtain ⟨e, he, hr⟩ := hrhs i
    exact cell1 (h45_5 i) hsm (hαμ i) (hα _) hG hg' hg ha hpay hs hr he
  have hsum := Rel.sum Finset.univ (fun i _ ↦ hcells i)
  -- The generated sum is the sum of the fourteen cells.
  obtain ⟨_, e46⟩ := familyGetStatic_eq (k := 0) h46_0
  replace e46 : w.w_46_0 = w.w_45_0 ⟨0, by decide⟩ := e46
  obtain ⟨_, e47⟩ := familyGetStatic_eq (k := 1) h47_0
  replace e47 : w.w_47_0 = w.w_45_0 ⟨1, by decide⟩ := e47
  obtain ⟨_, e49⟩ := familyGetStatic_eq (k := 0) h49_0
  replace e49 : w.w_49_0 = w.w_45_1 ⟨0, by decide⟩ := e49
  obtain ⟨_, e50⟩ := familyGetStatic_eq (k := 1) h50_0
  replace e50 : w.w_50_0 = w.w_45_1 ⟨1, by decide⟩ := e50
  obtain ⟨_, e52⟩ := familyGetStatic_eq (k := 0) h52_0
  replace e52 : w.w_52_0 = w.w_45_2 ⟨0, by decide⟩ := e52
  obtain ⟨_, e53⟩ := familyGetStatic_eq (k := 1) h53_0
  replace e53 : w.w_53_0 = w.w_45_2 ⟨1, by decide⟩ := e53
  obtain ⟨_, e55⟩ := familyGetStatic_eq (k := 0) h55_0
  replace e55 : w.w_55_0 = w.w_45_3 ⟨0, by decide⟩ := e55
  obtain ⟨_, e56⟩ := familyGetStatic_eq (k := 1) h56_0
  replace e56 : w.w_56_0 = w.w_45_3 ⟨1, by decide⟩ := e56
  obtain ⟨_, e58⟩ := familyGetStatic_eq (k := 0) h58_0
  replace e58 : w.w_58_0 = w.w_45_4 ⟨0, by decide⟩ := e58
  obtain ⟨_, e59⟩ := familyGetStatic_eq (k := 1) h59_0
  replace e59 : w.w_59_0 = w.w_45_4 ⟨1, by decide⟩ := e59
  obtain ⟨_, e61⟩ := familyGetStatic_eq (k := 0) h61_0
  replace e61 : w.w_61_0 = w.w_45_5 ⟨0, by decide⟩ := e61
  obtain ⟨_, e62⟩ := familyGetStatic_eq (k := 1) h62_0
  replace e62 : w.w_62_0 = w.w_45_5 ⟨1, by decide⟩ := e62
  have hloop := sum1 (count := 12) (init := (w.w_46_0 + w.w_47_0, w.w_49_0 + w.w_50_0, w.w_52_0 + w.w_53_0, w.w_55_0 + w.w_56_0, w.w_58_0 + w.w_59_0, w.w_61_0 + w.w_62_0, ()))
    (final := (w.w_64_0, w.w_64_1, w.w_64_2, w.w_64_3, w.w_64_4, w.w_64_5, ())) h64_5
  simp only [addCells, Prod.mk.injEq] at hloop
  obtain ⟨l0, l1, l2, l3, l4, l5, -⟩ := hloop
  have m0 : w.w_64_0 = ∑ i : Fin 14, w.w_45_0 i := by rw [l0, e46, e47]; exact sum_cells _
  have m1 : w.w_64_1 = ∑ i : Fin 14, w.w_45_1 i := by rw [l1, e49, e50]; exact sum_cells _
  have m2 : w.w_64_2 = ∑ i : Fin 14, w.w_45_2 i := by rw [l2, e52, e53]; exact sum_cells _
  have m3 : w.w_64_3 = ∑ i : Fin 14, w.w_45_3 i := by rw [l3, e55, e56]; exact sum_cells _
  have m4 : w.w_64_4 = ∑ i : Fin 14, w.w_45_4 i := by rw [l4, e58, e59]; exact sum_cells _
  have m5 : w.w_64_5 = ∑ i : Fin 14, w.w_45_5 i := by rw [l5, e61, e62]; exact sum_cells _
  have hA : (∑ i : Fin 14, ![![w.w_45_0 i, w.w_45_1 i], ![w.w_45_2 i, w.w_45_3 i]]) =
      ![![w.w_64_0, w.w_64_1], ![w.w_64_2, w.w_64_3]] := by
    funext k j
    fin_cases k <;> fin_cases j <;> simp [Finset.sum_apply, m0, m1, m2, m3]
  have hc : (∑ i : Fin 14, ![w.w_45_4 i, w.w_45_5 i]) = ![w.w_64_4, w.w_64_5] := by
    funext j
    fin_cases j <;> simp [Finset.sum_apply, m4, m5]
  rw [hA, hc] at hsum
  exact hsum

end LwrProof
