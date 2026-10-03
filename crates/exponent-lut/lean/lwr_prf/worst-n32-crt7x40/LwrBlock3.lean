import LwrBlock2

/-!
# Online evaluation, block 3

Cell `i` of block `3` aligns member `coords 3 i` of `a` from `Q7` to `Q4`, multiplies the state
encoding at `Q4` by it, then by the selector of the cell, dropping `P4`; the block sums
its fourteen cells.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The error bound of a member of `a` aligned from `Q7` to `Q4`. -/
abbrev alignBound3 : Nat := (14 * N * 26 * ((1048576 : Int).toNat / 2) + (N + 2) * (A4 / 2)) / A4

/-- The error bound of a block-3 cell for a state of error `B`. -/
abbrev cellBound3 (B : Nat) : Nat :=
  ((6 + 6) * N * ((8 + 8) * N * alignBound3 * 524288 + N * 1 * B) * 524288 +
    (N + 2) * (P4 / 2)) / P4 + N * 1 * 26

set_option maxHeartbeats 16000000 in
/-- Cell `i` of block `3`: an encoding of `P3 (α_c μ ν_i)` at `Q3`. -/
theorem cell3 {model : HashModel} {key : ByteArray} {ca0 ca1 : Fin 17 → ExactMatrix Q7 N 1 14}
    {apay : Fin 17 → ExactMatrix Q7 N 1 1} {As : Fin 2 → Fin 2 → ExactMatrix Q4 N 1 6}
    {cs : Fin 2 → ExactMatrix Q4 N 1 6} {bot0 bot1 : Fin 14 → ExactMatrix Q3 N 1 4} {i : Fin 14}
    {outputs}
    (hcell : Stage_online.parallel_generatedRoot_89 Backend.backend model
      GeneratedClaim.stage_4_params i
      (ca0, key, ca1, As 0 0, As 1 0, cs 0, apay, As 0 1, As 1 1, cs 1, bot0 i, bot1 i, ()) outputs)
    {sm sp μ ν : ErrorPoly N} {α : Fin 17 → ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    (hαμ : polyNorm (α ⟨coords 3 i, coords_lt 3 i⟩ * μ) ≤ 1)
    (hα : polyNorm (α ⟨coords 3 i, coords_lt 3 i⟩) ≤ 1)
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 1048576 14 G)
    {g' : ExactMatrix Q3 N 1 6} {g : ExactMatrix Q4 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 6 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q3 N 1 4}
    (ha : ∀ k : Fin 17, Rel sm (memberKey model key k)
      (fun j ↦ reducePoly Q7 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] 26)
    (hpay : ∀ k : Fin 17, apay k = fun _ _ ↦ reducePoly Q7 N (α k))
    {Bs : Nat} (hs : Rel sm As (fun j ↦ reducePoly Q4 N (((P4 : Nat) : Int) * μ * unitPair sp j) • g)
      cs Bs)
    {eB : Fin 2 → ErrorMatrix N 1 4}
    (hrhs : ∀ j, reducePoly Q3 N sp • rhsTop model key Q3 4 rhsPrefix3 i j + ![bot0 i, bot1 i] j =
      reducePoly Q3 N (((P3 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q3 N 1 4 (eB j))
    (heB : ∀ j, CoeffBound (eB j) 26) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q3 N (((P3 : Nat) : Int) * (α ⟨coords 3 i, coords_lt 3 i⟩ * μ * ν) *
        unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] (cellBound3 Bs) := by
  unfold Stage_online.parallel_generatedRoot_89 at hcell
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
  have hc := selectInt_coords3 i
  rw [hc] at h11_0 h19_0 h27_0 h30_0
  obtain ⟨_, hw2⟩ := familyGet_eq (k := coords 3 i) (by rw [← hc]; exact h2_0)
  obtain ⟨_, hw24⟩ := familyGet_eq (k := coords 3 i) (by rw [← hc]; exact h24_0)
  obtain ⟨_, hw44⟩ := familyGet_eq (k := coords 3 i) (by rw [← hc]; exact h44_0)
  subst hw2 hw24 hw44
  have hkey : memberKey model key (coords 3 i) = ![![w_11_0, w_27_0], ![w_19_0, w_30_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h27_0).symm
    · exact (hashSample_eq h19_0).symm
    · exact (hashSample_eq h30_0).symm
  have ha' := ha ⟨coords 3 i, coords_lt 3 i⟩
  rw [hkey] at ha'
  -- The alignment of the member from `Q7` to `Q4`; the row and the key share one decomposition.
  have hg13 := gadgetMatrixRuns_unique h13_0 h4_0
  subst hg13
  have hg14 := centeredRebaseRuns_unique h14_0 h5_0
  subst hg14
  have hH := gadgetDecomposeRuns_unique h16_0 h7_0
  subst hH
  have hal := align_down (q := Q7) (q' := Q4) (p := A4) (m := 14) (m' := 8) hN q7_eq4 hQ7 hQ4
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
  -- The public multiplication at `Q4`.
  have hpayb := modulusReduce_poly hQ7 hQ4 hN (s := α ⟨coords 3 i, coords_lt 3 i⟩)
    (by rw [← hpay]; exact h45_0)
  have hpub := mul_public (q := Q4) (m := 8) (m' := 6) (bound := 524288) hN hQ4
    Backend.layout3 layout4 rfl h4_0 hα hal hs h18_0_1 h22_0 h29_0_1 h33_0 h34_0
    (Xs := ![w_37_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h37_0, h53_0])
    (D' := ![w_38_0, w_54_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h54_0])
    (D := ![w_39_0, w_55_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h55_0])
    hpayb (Y := ![w_41_0, w_57_0]) (A' := ![![w_48_0, w_61_0], ![w_49_0, w_62_0]])
    (fun j ↦ by fin_cases j; exacts [h41_0, h57_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h61_0])
    (fun j ↦ by fin_cases j; exacts [h49_0, h62_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q4 N (((P4 : Nat) : Int) *
      (α ⟨coords 3 i, coords_lt 3 i⟩ * μ) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `P4`.
  have htop : rhsTop model key Q3 4 rhsPrefix3 i = ![w_65_0, w_73_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h65_0).symm
    · exact (hashSample_eq h73_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q4) (q' := Q3) (p := P4) (m := 6) (m' := 4)
    (bound := 524288) hN q4_eq hQ4 hQ3 (by norm_num) Backend.layout2 layout3 rfl hg' hg hsm hpub'
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
/-- One iteration of the block-3 sum adds cell `2 + i` to every block. -/
theorem sum3_step {model : HashModel} {f0 f1 f2 f3 f4 f5 : Fin 14 → ExactMatrix Q3 N 1 4} {i : Nat}
    {current next : SumState Q3 4}
    (hbody : Stage_online.sequential_generatedRoot_108 Backend.backend model
      GeneratedClaim.stage_4_params i
      (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
        current.2.2.2.2.2.1, f0, f1, f2, f3, f4, f5, ()) next) :
    next = (current.1 + cellAt f0 (2 + i), current.2.1 + cellAt f1 (2 + i),
      current.2.2.1 + cellAt f2 (2 + i), current.2.2.2.1 + cellAt f3 (2 + i),
      current.2.2.2.2.1 + cellAt f4 (2 + i), current.2.2.2.2.2.1 + cellAt f5 (2 + i), ()) := by
  unfold Stage_online.sequential_generatedRoot_108 at hbody
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

/-- The generated block-3 sum adds cells `2, ..., count + 1` to its initial state. -/
theorem sum3 {model : HashModel} {f0 f1 f2 f3 f4 f5 : Fin 14 → ExactMatrix Q3 N 1 4} {count : Nat}
    {init final : SumState Q3 4}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : SumState Q3 4) ↦
      Stage_online.sequential_generatedRoot_108 Backend.backend model GeneratedClaim.stage_4_params i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, f0, f1, f2, f3, f4, f5, ()) next) count init final) :
    final = addCells init f0 f1 f2 f3 f4 f5 count := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : SumState Q3 4) ↦ s = addCells init f0 f1 f2 f3 f4 f5 k) ?_ ?_ hruns
  · simp [addCells]
  · intro i current next hcur hbody
    rw [sum3_step hbody, hcur]
    simp only [addCells, Finset.sum_range_succ, add_assoc]


set_option maxHeartbeats 64000000 in
/-- Block `3`: the sum of its fourteen cells, an encoding of `P3 Σ_i α_(c_i) μ ν_i` at `Q3`. -/
theorem block3 {model : HashModel} {key : ByteArray}
    {s0 s1 s2 s3 : Fin 16 → ExactMatrix Q1 N 1 1} {k0 k1 : Fin 16 → ExactMatrix Q2 N 1 2}
    {ca0 ca1 : Fin 17 → ExactMatrix Q7 N 1 14} {cs0 cs1 : ExactMatrix Q7 N 1 12}
    {apay : Fin 17 → ExactMatrix Q7 N 1 1}
    {r00 r01 : Fin 14 → ExactMatrix Q6 N 1 10} {r10 r11 : Fin 14 → ExactMatrix Q5 N 1 8}
    {r20 r21 : Fin 14 → ExactMatrix Q4 N 1 6} {r30 r31 : Fin 14 → ExactMatrix Q3 N 1 4}
    {r40 r41 : Fin 14 → ExactMatrix Q2 N 1 2} {outputs} {w : Stage_online.generatedRoot.Witness}
    (hb : Stage_online.generatedRoot.body Backend.backend model GeneratedClaim.stage_4_params
      (s0, s1, s2, s3, k0, key, k1, ca0, ca1, cs0, apay, cs1, r00, r01, r10, r11, r20, r21, r30, r31,
        r40, r41, ()) outputs w)
    {sm sp μ : ErrorPoly N} {α : Fin 17 → ErrorPoly N} {ν : Fin 14 → ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    (hαμ : ∀ i : Fin 14, polyNorm (α ⟨coords 3 i, coords_lt 3 i⟩ * μ) ≤ 1)
    (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 1048576 14 G)
    {g' : ExactMatrix Q3 N 1 6} {g : ExactMatrix Q4 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 6 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q3 N 1 4}
    (ha : ∀ k : Fin 17, Rel sm (memberKey model key k)
      (fun j ↦ reducePoly Q7 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] 26)
    (hpay : ∀ k : Fin 17, apay k = fun _ _ ↦ reducePoly Q7 N (α k))
    {Bs : Nat} (hs : Rel sm ![![w.w_86_0, w.w_86_1], ![w.w_86_2, w.w_86_3]]
      (fun j ↦ reducePoly Q4 N (((P4 : Nat) : Int) * μ * unitPair sp j) • g)
      ![w.w_86_4, w.w_86_5] Bs)
    (hrhs : ∀ i : Fin 14, ∃ e : Fin 2 → ErrorMatrix N 1 4, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q3 N sp • rhsTop model key Q3 4 rhsPrefix3 i j + ![r30 i, r31 i] j =
        reducePoly Q3 N (((P3 : Nat) : Int) * ν i * unitPair sp j) • h +
          reduceMatrix Q3 N 1 4 (e j)) :
    Rel sm ![![w.w_108_0, w.w_108_1], ![w.w_108_2, w.w_108_3]]
      (∑ i : Fin 14, fun j ↦ reducePoly Q3 N (((P3 : Nat) : Int) *
        (α ⟨coords 3 i, coords_lt 3 i⟩ * μ * ν i) * unitPair sp j) • h)
      ![w.w_108_4, w.w_108_5] (∑ _i : Fin 14, cellBound3 Bs) := by
  unfold Stage_online.generatedRoot.body at hb
  extract_lets at hb
  have h89_5 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h90_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h91_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h93_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h94_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h96_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h97_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h99_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h100_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h102_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h103_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h105_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h106_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h108_5 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  -- Every cell, then their sum.
  have hcells : ∀ i : Fin 14, Rel sm ![![w.w_89_0 i, w.w_89_1 i], ![w.w_89_2 i, w.w_89_3 i]]
      (fun j ↦ reducePoly Q3 N (((P3 : Nat) : Int) * (α ⟨coords 3 i, coords_lt 3 i⟩ * μ * ν i) *
        unitPair sp j) • h) ![w.w_89_4 i, w.w_89_5 i] (cellBound3 Bs) := by
    intro i
    obtain ⟨e, he, hr⟩ := hrhs i
    exact cell3 (h89_5 i) hsm (hαμ i) (hα _) hG hg' hg ha hpay hs hr he
  have hsum := Rel.sum Finset.univ (fun i _ ↦ hcells i)
  -- The generated sum is the sum of the fourteen cells.
  obtain ⟨_, e90⟩ := familyGetStatic_eq (k := 0) h90_0
  replace e90 : w.w_90_0 = w.w_89_0 ⟨0, by decide⟩ := e90
  obtain ⟨_, e91⟩ := familyGetStatic_eq (k := 1) h91_0
  replace e91 : w.w_91_0 = w.w_89_0 ⟨1, by decide⟩ := e91
  obtain ⟨_, e93⟩ := familyGetStatic_eq (k := 0) h93_0
  replace e93 : w.w_93_0 = w.w_89_1 ⟨0, by decide⟩ := e93
  obtain ⟨_, e94⟩ := familyGetStatic_eq (k := 1) h94_0
  replace e94 : w.w_94_0 = w.w_89_1 ⟨1, by decide⟩ := e94
  obtain ⟨_, e96⟩ := familyGetStatic_eq (k := 0) h96_0
  replace e96 : w.w_96_0 = w.w_89_2 ⟨0, by decide⟩ := e96
  obtain ⟨_, e97⟩ := familyGetStatic_eq (k := 1) h97_0
  replace e97 : w.w_97_0 = w.w_89_2 ⟨1, by decide⟩ := e97
  obtain ⟨_, e99⟩ := familyGetStatic_eq (k := 0) h99_0
  replace e99 : w.w_99_0 = w.w_89_3 ⟨0, by decide⟩ := e99
  obtain ⟨_, e100⟩ := familyGetStatic_eq (k := 1) h100_0
  replace e100 : w.w_100_0 = w.w_89_3 ⟨1, by decide⟩ := e100
  obtain ⟨_, e102⟩ := familyGetStatic_eq (k := 0) h102_0
  replace e102 : w.w_102_0 = w.w_89_4 ⟨0, by decide⟩ := e102
  obtain ⟨_, e103⟩ := familyGetStatic_eq (k := 1) h103_0
  replace e103 : w.w_103_0 = w.w_89_4 ⟨1, by decide⟩ := e103
  obtain ⟨_, e105⟩ := familyGetStatic_eq (k := 0) h105_0
  replace e105 : w.w_105_0 = w.w_89_5 ⟨0, by decide⟩ := e105
  obtain ⟨_, e106⟩ := familyGetStatic_eq (k := 1) h106_0
  replace e106 : w.w_106_0 = w.w_89_5 ⟨1, by decide⟩ := e106
  have hloop := sum3 (count := 12) (init := (w.w_90_0 + w.w_91_0, w.w_93_0 + w.w_94_0, w.w_96_0 + w.w_97_0, w.w_99_0 + w.w_100_0, w.w_102_0 + w.w_103_0, w.w_105_0 + w.w_106_0, ()))
    (final := (w.w_108_0, w.w_108_1, w.w_108_2, w.w_108_3, w.w_108_4, w.w_108_5, ())) h108_5
  simp only [addCells, Prod.mk.injEq] at hloop
  obtain ⟨l0, l1, l2, l3, l4, l5, -⟩ := hloop
  have m0 : w.w_108_0 = ∑ i : Fin 14, w.w_89_0 i := by rw [l0, e90, e91]; exact sum_cells _
  have m1 : w.w_108_1 = ∑ i : Fin 14, w.w_89_1 i := by rw [l1, e93, e94]; exact sum_cells _
  have m2 : w.w_108_2 = ∑ i : Fin 14, w.w_89_2 i := by rw [l2, e96, e97]; exact sum_cells _
  have m3 : w.w_108_3 = ∑ i : Fin 14, w.w_89_3 i := by rw [l3, e99, e100]; exact sum_cells _
  have m4 : w.w_108_4 = ∑ i : Fin 14, w.w_89_4 i := by rw [l4, e102, e103]; exact sum_cells _
  have m5 : w.w_108_5 = ∑ i : Fin 14, w.w_89_5 i := by rw [l5, e105, e106]; exact sum_cells _
  have hA : (∑ i : Fin 14, ![![w.w_89_0 i, w.w_89_1 i], ![w.w_89_2 i, w.w_89_3 i]]) =
      ![![w.w_108_0, w.w_108_1], ![w.w_108_2, w.w_108_3]] := by
    funext k j
    fin_cases k <;> fin_cases j <;> simp [Finset.sum_apply, m0, m1, m2, m3]
  have hc : (∑ i : Fin 14, ![w.w_89_4 i, w.w_89_5 i]) = ![w.w_108_4, w.w_108_5] := by
    funext j
    fin_cases j <;> simp [Finset.sum_apply, m4, m5]
  rw [hA, hc] at hsum
  exact hsum

end LwrProof
