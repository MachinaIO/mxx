import TopLemmas

/-!
# The PRF cells (written by `generator/gen_cells.py`)

Cell `i = 5 o + k` of bucket `b` of a PRF child multiplies member `o`'s state by member `k` of the
label `a` (a public multiplication, after aligning `a` to the state's modulus in bucket `1`) and by
its selector bit (a private multiplication that drops one prime).
-/

namespace RefreshTopProof.Part_0_0

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

def maskRhs0 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 55, 101, 55, 98, 52, 99, 99, 56, 98, 48, 100, 51, 52, 53, 99, 99, 102, 57, 55, 57, 50, 48, 102, 102, 48, 48, 55, 49, 48, 55, 52, 49, 57, 54, 56, 54, 56, 101, 56, 51, 51, 56, 101, 97, 51, 49, 99, 97, 52, 100, 102, 52, 54, 48, 99, 49, 49, 99, 99, 100, 53, 97, 51, 54, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 53, 57, 44, 32, 49, 52, 53, 44, 32, 50, 57, 44, 32, 49, 54, 51, 44, 32, 56, 57, 44, 32, 49, 49, 57, 44, 32, 49, 55, 56, 44, 32, 50, 49, 44, 32, 49, 53, 52, 44, 32, 49, 54, 50, 44, 32, 49, 55, 51, 44, 32, 49, 51, 55, 44, 32, 49, 54, 50, 44, 32, 50, 52, 52, 44, 32, 54, 51, 44, 32, 50, 52, 55, 44, 32, 50, 50, 56, 44, 32, 49, 53, 52, 44, 32, 49, 57, 54, 44, 32, 55, 44, 32, 49, 51, 55, 44, 32, 50, 51, 54, 44, 32, 49, 51, 57, 44, 32, 54, 55, 44, 32, 50, 48, 51, 44, 32, 49, 55, 52, 44, 32, 53, 52, 44, 32, 51, 48, 44, 32, 57, 56, 44, 32, 50, 53, 53, 44, 32, 49, 50, 48, 44, 32, 49, 53, 54, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49]
def maskRhs1 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 55, 101, 55, 98, 52, 99, 99, 56, 98, 48, 100, 51, 52, 53, 99, 99, 102, 57, 55, 57, 50, 48, 102, 102, 48, 48, 55, 49, 48, 55, 52, 49, 57, 54, 56, 54, 56, 101, 56, 51, 51, 56, 101, 97, 51, 49, 99, 97, 52, 100, 102, 52, 54, 48, 99, 49, 49, 99, 99, 100, 53, 97, 51, 54, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 53, 57, 44, 32, 49, 52, 53, 44, 32, 50, 57, 44, 32, 49, 54, 51, 44, 32, 56, 57, 44, 32, 49, 49, 57, 44, 32, 49, 55, 56, 44, 32, 50, 49, 44, 32, 49, 53, 52, 44, 32, 49, 54, 50, 44, 32, 49, 55, 51, 44, 32, 49, 51, 55, 44, 32, 49, 54, 50, 44, 32, 50, 52, 52, 44, 32, 54, 51, 44, 32, 50, 52, 55, 44, 32, 50, 50, 56, 44, 32, 49, 53, 52, 44, 32, 49, 57, 54, 44, 32, 55, 44, 32, 49, 51, 55, 44, 32, 50, 51, 54, 44, 32, 49, 51, 57, 44, 32, 54, 55, 44, 32, 50, 48, 51, 44, 32, 49, 55, 52, 44, 32, 53, 52, 44, 32, 51, 48, 44, 32, 57, 56, 44, 32, 50, 53, 53, 44, 32, 49, 50, 48, 44, 32, 49, 53, 54, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `0` of the mask PRF: an encoding of `p₅ (α_(i mod 5) μ ν_i)` at `Q6`. -/
theorem mask_cell0 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q7 N 1 14}
    {apay : Fin 5 → ExactMatrix Q7 N 1 1} {As : Fin 2 → Fin 2 → ExactMatrix Q7 N 1 12}
    {cs : Fin 2 → ExactMatrix Q7 N 1 12} {bot0 bot1 : Fin 40 → ExactMatrix Q6 N 1 10} {i : Fin 40}
    {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_115 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As 0 0, As 1 0, cs 0, apay, As 0 1, As 1 1, cs 1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp μ ν : ErrorPoly N} {α : Fin 5 → ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 32768 14 G)
    {g' : ExactMatrix Q6 N 1 12} {g : ExactMatrix Q7 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 32768 12 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q6 N 1 10} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q7 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q7 N (α k))
    {Bs : Nat} (hs : Rel sm As (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * μ * unitPair sp j) • g)
      cs Bs)
    {eB : Fin 2 → ErrorMatrix N 1 10} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q6 N sp • rhsTop model key Q6 10 maskRhs0 i j + ![bot0 i, bot1 i] j =
      reducePoly Q6 N (((p5 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q6 N 1 10 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ * μ * ν) *
        unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((12 + 12) * N * ((14 + 14) * N * Ba * 16384 + N * 1 * Bs) * 16384 + (N + 2) * (p6 / 2)) / p6) +
        N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_115 at hcell
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_6_0, hcell⟩ := hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  obtain ⟨w_7_concat_1, hcell⟩ := hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  obtain ⟨w_14_concat_1, hcell⟩ := hcell
  obtain ⟨w_15_0, hcell⟩ := hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_19_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hcell
  obtain ⟨w_28_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  obtain ⟨w_30_0, hcell⟩ := hcell
  obtain ⟨w_30_concat_1, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  obtain ⟨w_34_0, hcell⟩ := hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  extract_lets -merge w_36_0 at hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  extract_lets -merge w_40_0 at hcell
  obtain ⟨w_41_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_43_concat_1, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_46_0, hcell⟩ := hcell
  obtain ⟨w_48_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_50_0, hcell⟩ := hcell
  extract_lets -merge w_51_0 at hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  obtain ⟨w_53_0, hcell⟩ := hcell
  obtain ⟨w_54_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_58_0, hcell⟩ := hcell
  extract_lets -merge w_59_0 at hcell
  obtain ⟨w_60_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  obtain ⟨w_65_0, hcell⟩ := hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_67_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  obtain ⟨w_69_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_115.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_115.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_115.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_115.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_115.constraints_4] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member of `a`, and its public key.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h4_0 h6_0 h9_0 h11_0 h13_0 h25_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw9⟩ := familyGet_eq h9_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  subst hw2 hw4 hw6 hw9 hw11 hw13 hw25
  have ha' := ha ⟨i.val % 5, hk⟩
  -- The public multiplication at `Q7`.
  have hpub := mul_public (q := Q7) (m := 14) (m' := 12) (bound := 16384) hN (by norm_num)
    Backend.layout5 layoutQ7 rfl hG (hα _) ha' hs h4_0_1 h7_0 h11_0_1 h14_0 h15_0
    (Xs := ![w_18_0, w_33_0]) (fun j ↦ by fin_cases j; exacts [h18_0, h33_0])
    (D' := ![w_19_0, w_34_0]) (fun j ↦ by fin_cases j; exacts [h19_0, h34_0])
    (D := ![w_20_0, w_35_0]) (fun j ↦ by fin_cases j; exacts [h20_0, h35_0])
    (hpay _) (Y := ![w_22_0, w_37_0]) (A' := ![![w_28_0, w_41_0], ![w_29_0, w_42_0]])
    (fun j ↦ by fin_cases j; exacts [h22_0, h37_0]) (fun j ↦ by fin_cases j; exacts [h28_0, h41_0])
    (fun j ↦ by fin_cases j; exacts [h29_0, h42_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₆`.
  have htop : rhsTop model key Q6 10 maskRhs0 i = ![w_46_0, w_54_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h46_0).symm
    · exact (hashSample_eq h54_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q7) (q' := Q6) (p := p6) (m := 12) (m' := 10)
    (bound := 16384) hN q7_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout4 layoutQ6 rfl
    hg' hg hsm hpub' h28_0_1 h30_0 h41_0_1 h43_0 h44_0 (top := ![w_46_0, w_54_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_48_0, w_56_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h56_0])
    (D' := ![w_49_0, w_57_0]) (fun j ↦ by fin_cases j; exacts [h49_0, h57_0])
    (D := ![w_50_0, w_58_0]) (fun j ↦ by fin_cases j; exacts [h50_0, h58_0])
    hrhs' heB (hαμ _) (X := ![w_66_0, w_68_0]) (M := ![![w_52_0, w_60_0], ![w_62_0, w_64_0]])
    (fun j ↦ by fin_cases j; exacts [h66_0, h68_0]) (fun j ↦ by fin_cases j; exacts [h52_0, h60_0])
    (fun j ↦ by fin_cases j; exacts [h62_0, h64_0])
    (c' := ![w_67_0, w_69_0]) (A' := ![![w_53_0, w_61_0], ![w_63_0, w_65_0]])
    (fun j ↦ by fin_cases j; exacts [h67_0, h69_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h53_0, h61_0, h63_0, h65_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring


set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `1` of the mask PRF: an encoding of `p₄ (α_(i mod 5) μ_(i/5) ν_i)` at `Q5`. -/
theorem mask_cell1 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q7 N 1 14}
    {apay : Fin 5 → ExactMatrix Q7 N 1 1} {As00 As10 As01 As11 cs0 cs1 : Fin 8 → ExactMatrix Q6 N 1 10}
    {bot0 bot1 : Fin 40 → ExactMatrix Q5 N 1 8} {i : Fin 40} {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_119 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp ν : ErrorPoly N} {μ : Fin 8 → ErrorPoly N} {α : Fin 5 → ErrorPoly N}
    (hsm : polyNorm sm ≤ 1) (hαμ : ∀ k o, polyNorm (α k * μ o) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 32768 14 G)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q7 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q7 N (α k))
    {Bs : Nat} (hs : ∀ o : Fin 8, Rel sm ![![As00 o, As01 o], ![As10 o, As11 o]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * μ o * unitPair sp j) • g) ![cs0 o, cs1 o] Bs)
    {eB : Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q5 N sp • rhsTop model key Q5 8 maskRhs1 i j + ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q5 N 1 8 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ *
        μ ⟨i.val / 5, by omega⟩ * ν) * unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((10 + 10) * N * ((12 + 12) * N * ((14 * N * Ba * (32768 / 2) + (N + 2) * (p6 / 2)) / p6) *
        16384 + N * 1 * Bs) * 16384 + (N + 2) * (p5 / 2)) / p5) + N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_119 at hcell
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_5_0, hcell⟩ := hcell
  extract_lets -merge w_6_0 at hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  extract_lets -merge w_8_0 at hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  extract_lets -merge w_15_0 at hcell
  obtain ⟨w_16_0, hcell⟩ := hcell
  extract_lets -merge w_17_0 at hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_23_0, hcell⟩ := hcell
  obtain ⟨w_23_concat_1, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 at hcell
  obtain ⟨w_27_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  extract_lets -merge w_30_0 at hcell
  obtain ⟨w_31_0, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  extract_lets -merge w_34_0 at hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  obtain ⟨w_36_0, hcell⟩ := hcell
  obtain ⟨w_36_concat_1, hcell⟩ := hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  obtain ⟨w_40_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_45_0, hcell⟩ := hcell
  extract_lets -merge w_46_0 at hcell
  obtain ⟨w_47_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_51_0, hcell⟩ := hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  extract_lets -merge w_53_0 w_54_0 at hcell
  obtain ⟨w_55_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_57_concat_1, hcell⟩ := hcell
  obtain ⟨w_59_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  extract_lets -merge w_65_0 at hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  extract_lets -merge w_69_0 w_70_0 at hcell
  obtain ⟨w_71_0, hcell⟩ := hcell
  obtain ⟨w_72_0, hcell⟩ := hcell
  obtain ⟨w_73_0, hcell⟩ := hcell
  obtain ⟨w_73_concat_1, hcell⟩ := hcell
  obtain ⟨w_74_0, hcell⟩ := hcell
  obtain ⟨w_76_0, hcell⟩ := hcell
  obtain ⟨w_78_0, hcell⟩ := hcell
  obtain ⟨w_79_0, hcell⟩ := hcell
  obtain ⟨w_80_0, hcell⟩ := hcell
  extract_lets -merge w_81_0 at hcell
  obtain ⟨w_82_0, hcell⟩ := hcell
  obtain ⟨w_83_0, hcell⟩ := hcell
  obtain ⟨w_84_0, hcell⟩ := hcell
  obtain ⟨w_86_0, hcell⟩ := hcell
  obtain ⟨w_87_0, hcell⟩ := hcell
  obtain ⟨w_88_0, hcell⟩ := hcell
  extract_lets -merge w_89_0 at hcell
  obtain ⟨w_90_0, hcell⟩ := hcell
  obtain ⟨w_91_0, hcell⟩ := hcell
  obtain ⟨w_92_0, hcell⟩ := hcell
  obtain ⟨w_93_0, hcell⟩ := hcell
  obtain ⟨w_94_0, hcell⟩ := hcell
  obtain ⟨w_95_0, hcell⟩ := hcell
  obtain ⟨w_96_0, hcell⟩ := hcell
  obtain ⟨w_97_0, hcell⟩ := hcell
  obtain ⟨w_98_0, hcell⟩ := hcell
  obtain ⟨w_99_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_119.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_119.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_119.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_119.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_119.constraints_4, Stage_online_eval_0_0.parallel_generatedRoot_119.constraints_5] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member `i / 5`, the member of `a`.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h11_0 h20_0 h25_0 h29_0 h33_0 h51_0
  have ho : w_39_0 = ((i.val / 5 : Nat) : Int) := fdiv_slot i.val
  rw [ho] at h40_0 h42_0 h49_0 h59_0 h61_0 h68_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  obtain ⟨_, hw51⟩ := familyGet_eq h51_0
  obtain ⟨hko, hw40⟩ := familyGet_eq h40_0
  obtain ⟨_, hw42⟩ := familyGet_eq h42_0
  obtain ⟨_, hw49⟩ := familyGet_eq h49_0
  obtain ⟨_, hw59⟩ := familyGet_eq h59_0
  obtain ⟨_, hw61⟩ := familyGet_eq h61_0
  obtain ⟨_, hw68⟩ := familyGet_eq h68_0
  subst hw2 hw11 hw20 hw25 hw29 hw33 hw51 hw40 hw42 hw49 hw59 hw61 hw68
  have ha' := ha ⟨i.val % 5, hk⟩
  have hs' := hs ⟨i.val / 5, hko⟩
  -- The alignment of the member from `Q7` to `Q6`; the row and the key share one decomposition.
  have hg13 := gadgetMatrixRuns_unique h13_0 h4_0
  subst hg13
  have hg14 := centeredRebaseRuns_unique h14_0 h5_0
  subst hg14
  have hH := gadgetDecomposeRuns_unique h16_0 h7_0
  subst hH
  have hal := align_down (q := Q7) (q' := Q6) (p := p6) (m := 14) (m' := 12) hN q7_eq (by norm_num)
    (by norm_num) (by norm_num) Backend.layout5 layoutQ7 rfl hG h5_0 hsm ha'
    (by
      funext r c
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp (config := { zetaDelta := true }) [Matrix.mul_apply, matrixPolynomial_single, map_ofNat])
    h16_0 (Y := ![_, _]) (K := ![![_, _], ![_, _]])
    (fun j ↦ by fin_cases j <;> rfl) (fun k j ↦ by fin_cases k <;> fin_cases j <;> rfl)
    (c' := ![w_9_0, w_27_0]) (A' := ![![w_18_0, w_31_0], ![w_22_0, w_35_0]])
    (fun j ↦ by fin_cases j; exacts [h9_0, h27_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h18_0, h31_0, h22_0, h35_0])
  -- The public multiplication at `Q6`.
  have hpayb := modulusReduce_poly (q := Q7) (p := Q6) (by norm_num) (by norm_num) hN
    (s := α ⟨i.val % 5, hk⟩)
    (by rw [← hpay]; exact h52_0)
  have hpub := mul_public (q := Q6) (m := 12) (m' := 10) (bound := 16384) hN (by norm_num)
    Backend.layout4 layoutQ6 rfl h4_0 (hα _) hal hs' h18_0_1 h23_0 h31_0_1 h36_0 h37_0
    (Xs := ![w_43_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h43_0, h62_0])
    (D' := ![w_44_0, w_63_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h63_0])
    (D := ![w_45_0, w_64_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    hpayb (Y := ![w_47_0, w_66_0]) (A' := ![![w_55_0, w_71_0], ![w_56_0, w_72_0]])
    (fun j ↦ by fin_cases j; exacts [h47_0, h66_0]) (fun j ↦ by fin_cases j; exacts [h55_0, h71_0])
    (fun j ↦ by fin_cases j; exacts [h56_0, h72_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ ⟨i.val / 5, hko⟩) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₅`.
  have htop : rhsTop model key Q5 8 maskRhs1 i = ![w_76_0, w_84_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h76_0).symm
    · exact (hashSample_eq h84_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q6) (q' := Q5) (p := p5) (m := 10) (m' := 8)
    (bound := 16384) hN q6_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout3 layoutQ5 rfl
    hg' hg hsm hpub' h55_0_1 h57_0 h71_0_1 h73_0 h74_0 (top := ![w_76_0, w_84_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_78_0, w_86_0]) (fun j ↦ by fin_cases j; exacts [h78_0, h86_0])
    (D' := ![w_79_0, w_87_0]) (fun j ↦ by fin_cases j; exacts [h79_0, h87_0])
    (D := ![w_80_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h80_0, h88_0])
    hrhs' heB (hαμ _ _) (X := ![w_96_0, w_98_0]) (M := ![![w_82_0, w_90_0], ![w_92_0, w_94_0]])
    (fun j ↦ by fin_cases j; exacts [h96_0, h98_0]) (fun j ↦ by fin_cases j; exacts [h82_0, h90_0])
    (fun j ↦ by fin_cases j; exacts [h92_0, h94_0])
    (c' := ![w_97_0, w_99_0]) (A' := ![![w_83_0, w_91_0], ![w_93_0, w_95_0]])
    (fun j ↦ by fin_cases j; exacts [h97_0, h99_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h83_0, h91_0, h93_0, h95_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring
def freshRhs0 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 55, 101, 55, 98, 52, 99, 99, 56, 98, 48, 100, 51, 52, 53, 99, 99, 102, 57, 55, 57, 50, 48, 102, 102, 48, 48, 55, 49, 48, 55, 52, 49, 57, 54, 56, 54, 56, 101, 56, 51, 51, 56, 101, 97, 51, 49, 99, 97, 52, 100, 102, 52, 54, 48, 99, 49, 49, 99, 99, 100, 53, 97, 51, 54, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 44, 32, 50, 54, 44, 32, 49, 51, 52, 44, 32, 49, 56, 52, 44, 32, 49, 56, 50, 44, 32, 49, 57, 52, 44, 32, 55, 53, 44, 32, 52, 49, 44, 32, 50, 52, 57, 44, 32, 51, 57, 44, 32, 49, 53, 52, 44, 32, 50, 50, 44, 32, 55, 55, 44, 32, 49, 51, 50, 44, 32, 54, 50, 44, 32, 51, 55, 44, 32, 50, 52, 53, 44, 32, 50, 52, 50, 44, 32, 49, 55, 50, 44, 32, 49, 54, 53, 44, 32, 49, 55, 53, 44, 32, 56, 49, 44, 32, 49, 56, 55, 44, 32, 49, 54, 51, 44, 32, 49, 48, 57, 44, 32, 57, 52, 44, 32, 49, 53, 55, 44, 32, 49, 48, 51, 44, 32, 50, 52, 54, 44, 32, 53, 44, 32, 49, 49, 57, 44, 32, 57, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49]
def freshRhs1 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 55, 101, 55, 98, 52, 99, 99, 56, 98, 48, 100, 51, 52, 53, 99, 99, 102, 57, 55, 57, 50, 48, 102, 102, 48, 48, 55, 49, 48, 55, 52, 49, 57, 54, 56, 54, 56, 101, 56, 51, 51, 56, 101, 97, 51, 49, 99, 97, 52, 100, 102, 52, 54, 48, 99, 49, 49, 99, 99, 100, 53, 97, 51, 54, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 44, 32, 50, 54, 44, 32, 49, 51, 52, 44, 32, 49, 56, 52, 44, 32, 49, 56, 50, 44, 32, 49, 57, 52, 44, 32, 55, 53, 44, 32, 52, 49, 44, 32, 50, 52, 57, 44, 32, 51, 57, 44, 32, 49, 53, 52, 44, 32, 50, 50, 44, 32, 55, 55, 44, 32, 49, 51, 50, 44, 32, 54, 50, 44, 32, 51, 55, 44, 32, 50, 52, 53, 44, 32, 50, 52, 50, 44, 32, 49, 55, 50, 44, 32, 49, 54, 53, 44, 32, 49, 55, 53, 44, 32, 56, 49, 44, 32, 49, 56, 55, 44, 32, 49, 54, 51, 44, 32, 49, 48, 57, 44, 32, 57, 52, 44, 32, 49, 53, 55, 44, 32, 49, 48, 51, 44, 32, 50, 52, 54, 44, 32, 53, 44, 32, 49, 49, 57, 44, 32, 57, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `0` of the fresh PRF: an encoding of `p₅ (α_(i mod 5) μ ν_i)` at `Q6`. -/
theorem fresh_cell0 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q7 N 1 14}
    {apay : Fin 5 → ExactMatrix Q7 N 1 1} {As : Fin 2 → Fin 2 → ExactMatrix Q7 N 1 12}
    {cs : Fin 2 → ExactMatrix Q7 N 1 12} {bot0 bot1 : Fin 240 → ExactMatrix Q6 N 1 10} {i : Fin 240}
    {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_386 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As 0 0, As 1 0, cs 0, apay, As 0 1, As 1 1, cs 1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp μ ν : ErrorPoly N} {α : Fin 5 → ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 32768 14 G)
    {g' : ExactMatrix Q6 N 1 12} {g : ExactMatrix Q7 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 32768 12 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q6 N 1 10} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q7 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q7 N (α k))
    {Bs : Nat} (hs : Rel sm As (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * μ * unitPair sp j) • g)
      cs Bs)
    {eB : Fin 2 → ErrorMatrix N 1 10} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q6 N sp • rhsTop model key Q6 10 freshRhs0 i j + ![bot0 i, bot1 i] j =
      reducePoly Q6 N (((p5 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q6 N 1 10 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ * μ * ν) *
        unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((12 + 12) * N * ((14 + 14) * N * Ba * 16384 + N * 1 * Bs) * 16384 + (N + 2) * (p6 / 2)) / p6) +
        N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_386 at hcell
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_6_0, hcell⟩ := hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  obtain ⟨w_7_concat_1, hcell⟩ := hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  obtain ⟨w_14_concat_1, hcell⟩ := hcell
  obtain ⟨w_15_0, hcell⟩ := hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_19_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hcell
  obtain ⟨w_28_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  obtain ⟨w_30_0, hcell⟩ := hcell
  obtain ⟨w_30_concat_1, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  obtain ⟨w_34_0, hcell⟩ := hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  extract_lets -merge w_36_0 at hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  extract_lets -merge w_40_0 at hcell
  obtain ⟨w_41_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_43_concat_1, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_46_0, hcell⟩ := hcell
  obtain ⟨w_48_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_50_0, hcell⟩ := hcell
  extract_lets -merge w_51_0 at hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  obtain ⟨w_53_0, hcell⟩ := hcell
  obtain ⟨w_54_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_58_0, hcell⟩ := hcell
  extract_lets -merge w_59_0 at hcell
  obtain ⟨w_60_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  obtain ⟨w_65_0, hcell⟩ := hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_67_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  obtain ⟨w_69_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_386.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_386.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_386.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_386.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_386.constraints_4] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member of `a`, and its public key.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h4_0 h6_0 h9_0 h11_0 h13_0 h25_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw9⟩ := familyGet_eq h9_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  subst hw2 hw4 hw6 hw9 hw11 hw13 hw25
  have ha' := ha ⟨i.val % 5, hk⟩
  -- The public multiplication at `Q7`.
  have hpub := mul_public (q := Q7) (m := 14) (m' := 12) (bound := 16384) hN (by norm_num)
    Backend.layout5 layoutQ7 rfl hG (hα _) ha' hs h4_0_1 h7_0 h11_0_1 h14_0 h15_0
    (Xs := ![w_18_0, w_33_0]) (fun j ↦ by fin_cases j; exacts [h18_0, h33_0])
    (D' := ![w_19_0, w_34_0]) (fun j ↦ by fin_cases j; exacts [h19_0, h34_0])
    (D := ![w_20_0, w_35_0]) (fun j ↦ by fin_cases j; exacts [h20_0, h35_0])
    (hpay _) (Y := ![w_22_0, w_37_0]) (A' := ![![w_28_0, w_41_0], ![w_29_0, w_42_0]])
    (fun j ↦ by fin_cases j; exacts [h22_0, h37_0]) (fun j ↦ by fin_cases j; exacts [h28_0, h41_0])
    (fun j ↦ by fin_cases j; exacts [h29_0, h42_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₆`.
  have htop : rhsTop model key Q6 10 freshRhs0 i = ![w_46_0, w_54_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h46_0).symm
    · exact (hashSample_eq h54_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q7) (q' := Q6) (p := p6) (m := 12) (m' := 10)
    (bound := 16384) hN q7_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout4 layoutQ6 rfl
    hg' hg hsm hpub' h28_0_1 h30_0 h41_0_1 h43_0 h44_0 (top := ![w_46_0, w_54_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_48_0, w_56_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h56_0])
    (D' := ![w_49_0, w_57_0]) (fun j ↦ by fin_cases j; exacts [h49_0, h57_0])
    (D := ![w_50_0, w_58_0]) (fun j ↦ by fin_cases j; exacts [h50_0, h58_0])
    hrhs' heB (hαμ _) (X := ![w_66_0, w_68_0]) (M := ![![w_52_0, w_60_0], ![w_62_0, w_64_0]])
    (fun j ↦ by fin_cases j; exacts [h66_0, h68_0]) (fun j ↦ by fin_cases j; exacts [h52_0, h60_0])
    (fun j ↦ by fin_cases j; exacts [h62_0, h64_0])
    (c' := ![w_67_0, w_69_0]) (A' := ![![w_53_0, w_61_0], ![w_63_0, w_65_0]])
    (fun j ↦ by fin_cases j; exacts [h67_0, h69_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h53_0, h61_0, h63_0, h65_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring


set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `1` of the fresh PRF: an encoding of `p₄ (α_(i mod 5) μ_(i/5) ν_i)` at `Q5`. -/
theorem fresh_cell1 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q7 N 1 14}
    {apay : Fin 5 → ExactMatrix Q7 N 1 1} {As00 As10 As01 As11 cs0 cs1 : Fin 48 → ExactMatrix Q6 N 1 10}
    {bot0 bot1 : Fin 240 → ExactMatrix Q5 N 1 8} {i : Fin 240} {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_390 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp ν : ErrorPoly N} {μ : Fin 48 → ErrorPoly N} {α : Fin 5 → ErrorPoly N}
    (hsm : polyNorm sm ≤ 1) (hαμ : ∀ k o, polyNorm (α k * μ o) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 32768 14 G)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q7 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q7 N (α k))
    {Bs : Nat} (hs : ∀ o : Fin 48, Rel sm ![![As00 o, As01 o], ![As10 o, As11 o]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * μ o * unitPair sp j) • g) ![cs0 o, cs1 o] Bs)
    {eB : Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q5 N sp • rhsTop model key Q5 8 freshRhs1 i j + ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q5 N 1 8 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ *
        μ ⟨i.val / 5, by omega⟩ * ν) * unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((10 + 10) * N * ((12 + 12) * N * ((14 * N * Ba * (32768 / 2) + (N + 2) * (p6 / 2)) / p6) *
        16384 + N * 1 * Bs) * 16384 + (N + 2) * (p5 / 2)) / p5) + N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_390 at hcell
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_5_0, hcell⟩ := hcell
  extract_lets -merge w_6_0 at hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  extract_lets -merge w_8_0 at hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  extract_lets -merge w_15_0 at hcell
  obtain ⟨w_16_0, hcell⟩ := hcell
  extract_lets -merge w_17_0 at hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_23_0, hcell⟩ := hcell
  obtain ⟨w_23_concat_1, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 at hcell
  obtain ⟨w_27_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  extract_lets -merge w_30_0 at hcell
  obtain ⟨w_31_0, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  extract_lets -merge w_34_0 at hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  obtain ⟨w_36_0, hcell⟩ := hcell
  obtain ⟨w_36_concat_1, hcell⟩ := hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  obtain ⟨w_40_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_45_0, hcell⟩ := hcell
  extract_lets -merge w_46_0 at hcell
  obtain ⟨w_47_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_51_0, hcell⟩ := hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  extract_lets -merge w_53_0 w_54_0 at hcell
  obtain ⟨w_55_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_57_concat_1, hcell⟩ := hcell
  obtain ⟨w_59_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  extract_lets -merge w_65_0 at hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  extract_lets -merge w_69_0 w_70_0 at hcell
  obtain ⟨w_71_0, hcell⟩ := hcell
  obtain ⟨w_72_0, hcell⟩ := hcell
  obtain ⟨w_73_0, hcell⟩ := hcell
  obtain ⟨w_73_concat_1, hcell⟩ := hcell
  obtain ⟨w_74_0, hcell⟩ := hcell
  obtain ⟨w_76_0, hcell⟩ := hcell
  obtain ⟨w_78_0, hcell⟩ := hcell
  obtain ⟨w_79_0, hcell⟩ := hcell
  obtain ⟨w_80_0, hcell⟩ := hcell
  extract_lets -merge w_81_0 at hcell
  obtain ⟨w_82_0, hcell⟩ := hcell
  obtain ⟨w_83_0, hcell⟩ := hcell
  obtain ⟨w_84_0, hcell⟩ := hcell
  obtain ⟨w_86_0, hcell⟩ := hcell
  obtain ⟨w_87_0, hcell⟩ := hcell
  obtain ⟨w_88_0, hcell⟩ := hcell
  extract_lets -merge w_89_0 at hcell
  obtain ⟨w_90_0, hcell⟩ := hcell
  obtain ⟨w_91_0, hcell⟩ := hcell
  obtain ⟨w_92_0, hcell⟩ := hcell
  obtain ⟨w_93_0, hcell⟩ := hcell
  obtain ⟨w_94_0, hcell⟩ := hcell
  obtain ⟨w_95_0, hcell⟩ := hcell
  obtain ⟨w_96_0, hcell⟩ := hcell
  obtain ⟨w_97_0, hcell⟩ := hcell
  obtain ⟨w_98_0, hcell⟩ := hcell
  obtain ⟨w_99_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_390.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_390.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_390.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_390.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_390.constraints_4, Stage_online_eval_0_0.parallel_generatedRoot_390.constraints_5] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member `i / 5`, the member of `a`.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h11_0 h20_0 h25_0 h29_0 h33_0 h51_0
  have ho : w_39_0 = ((i.val / 5 : Nat) : Int) := fdiv_slot i.val
  rw [ho] at h40_0 h42_0 h49_0 h59_0 h61_0 h68_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  obtain ⟨_, hw51⟩ := familyGet_eq h51_0
  obtain ⟨hko, hw40⟩ := familyGet_eq h40_0
  obtain ⟨_, hw42⟩ := familyGet_eq h42_0
  obtain ⟨_, hw49⟩ := familyGet_eq h49_0
  obtain ⟨_, hw59⟩ := familyGet_eq h59_0
  obtain ⟨_, hw61⟩ := familyGet_eq h61_0
  obtain ⟨_, hw68⟩ := familyGet_eq h68_0
  subst hw2 hw11 hw20 hw25 hw29 hw33 hw51 hw40 hw42 hw49 hw59 hw61 hw68
  have ha' := ha ⟨i.val % 5, hk⟩
  have hs' := hs ⟨i.val / 5, hko⟩
  -- The alignment of the member from `Q7` to `Q6`; the row and the key share one decomposition.
  have hg13 := gadgetMatrixRuns_unique h13_0 h4_0
  subst hg13
  have hg14 := centeredRebaseRuns_unique h14_0 h5_0
  subst hg14
  have hH := gadgetDecomposeRuns_unique h16_0 h7_0
  subst hH
  have hal := align_down (q := Q7) (q' := Q6) (p := p6) (m := 14) (m' := 12) hN q7_eq (by norm_num)
    (by norm_num) (by norm_num) Backend.layout5 layoutQ7 rfl hG h5_0 hsm ha'
    (by
      funext r c
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp (config := { zetaDelta := true }) [Matrix.mul_apply, matrixPolynomial_single, map_ofNat])
    h16_0 (Y := ![_, _]) (K := ![![_, _], ![_, _]])
    (fun j ↦ by fin_cases j <;> rfl) (fun k j ↦ by fin_cases k <;> fin_cases j <;> rfl)
    (c' := ![w_9_0, w_27_0]) (A' := ![![w_18_0, w_31_0], ![w_22_0, w_35_0]])
    (fun j ↦ by fin_cases j; exacts [h9_0, h27_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h18_0, h31_0, h22_0, h35_0])
  -- The public multiplication at `Q6`.
  have hpayb := modulusReduce_poly (q := Q7) (p := Q6) (by norm_num) (by norm_num) hN
    (s := α ⟨i.val % 5, hk⟩)
    (by rw [← hpay]; exact h52_0)
  have hpub := mul_public (q := Q6) (m := 12) (m' := 10) (bound := 16384) hN (by norm_num)
    Backend.layout4 layoutQ6 rfl h4_0 (hα _) hal hs' h18_0_1 h23_0 h31_0_1 h36_0 h37_0
    (Xs := ![w_43_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h43_0, h62_0])
    (D' := ![w_44_0, w_63_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h63_0])
    (D := ![w_45_0, w_64_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    hpayb (Y := ![w_47_0, w_66_0]) (A' := ![![w_55_0, w_71_0], ![w_56_0, w_72_0]])
    (fun j ↦ by fin_cases j; exacts [h47_0, h66_0]) (fun j ↦ by fin_cases j; exacts [h55_0, h71_0])
    (fun j ↦ by fin_cases j; exacts [h56_0, h72_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ ⟨i.val / 5, hko⟩) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₅`.
  have htop : rhsTop model key Q5 8 freshRhs1 i = ![w_76_0, w_84_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h76_0).symm
    · exact (hashSample_eq h84_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q6) (q' := Q5) (p := p5) (m := 10) (m' := 8)
    (bound := 16384) hN q6_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout3 layoutQ5 rfl
    hg' hg hsm hpub' h55_0_1 h57_0 h71_0_1 h73_0 h74_0 (top := ![w_76_0, w_84_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_78_0, w_86_0]) (fun j ↦ by fin_cases j; exacts [h78_0, h86_0])
    (D' := ![w_79_0, w_87_0]) (fun j ↦ by fin_cases j; exacts [h79_0, h87_0])
    (D := ![w_80_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h80_0, h88_0])
    hrhs' heB (hαμ _ _) (X := ![w_96_0, w_98_0]) (M := ![![w_82_0, w_90_0], ![w_92_0, w_94_0]])
    (fun j ↦ by fin_cases j; exacts [h96_0, h98_0]) (fun j ↦ by fin_cases j; exacts [h82_0, h90_0])
    (fun j ↦ by fin_cases j; exacts [h92_0, h94_0])
    (c' := ![w_97_0, w_99_0]) (A' := ![![w_83_0, w_91_0], ![w_93_0, w_95_0]])
    (fun j ↦ by fin_cases j; exacts [h97_0, h99_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h83_0, h91_0, h93_0, h95_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring
def smudge0Rhs0 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49]
def smudge0Rhs1 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `0` of the smudge0 PRF: an encoding of `p₅ (α_(i mod 5) μ ν_i)` at `Q5`. -/
theorem smudge0_cell0 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {As : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10}
    {cs : Fin 2 → ExactMatrix Q6 N 1 10} {bot0 bot1 : Fin 800 → ExactMatrix Q5 N 1 8} {i : Fin 800}
    {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_554 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As 0 0, As 1 0, cs 0, apay, As 0 1, As 1 1, cs 1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp μ ν : ErrorPoly N} {α : Fin 5 → ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (α k))
    {Bs : Nat} (hs : Rel sm As (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * μ * unitPair sp j) • g)
      cs Bs)
    {eB : Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q5 N sp • rhsTop model key Q5 8 smudge0Rhs0 i j + ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q5 N 1 8 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ * μ * ν) *
        unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((10 + 10) * N * ((12 + 12) * N * Ba * 16384 + N * 1 * Bs) * 16384 + (N + 2) * (p5 / 2)) / p5) +
        N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_554 at hcell
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_6_0, hcell⟩ := hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  obtain ⟨w_7_concat_1, hcell⟩ := hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  obtain ⟨w_14_concat_1, hcell⟩ := hcell
  obtain ⟨w_15_0, hcell⟩ := hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_19_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hcell
  obtain ⟨w_28_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  obtain ⟨w_30_0, hcell⟩ := hcell
  obtain ⟨w_30_concat_1, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  obtain ⟨w_34_0, hcell⟩ := hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  extract_lets -merge w_36_0 at hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  extract_lets -merge w_40_0 at hcell
  obtain ⟨w_41_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_43_concat_1, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_46_0, hcell⟩ := hcell
  obtain ⟨w_48_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_50_0, hcell⟩ := hcell
  extract_lets -merge w_51_0 at hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  obtain ⟨w_53_0, hcell⟩ := hcell
  obtain ⟨w_54_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_58_0, hcell⟩ := hcell
  extract_lets -merge w_59_0 at hcell
  obtain ⟨w_60_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  obtain ⟨w_65_0, hcell⟩ := hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_67_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  obtain ⟨w_69_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_554.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_554.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_554.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_554.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_554.constraints_4] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member of `a`, and its public key.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h4_0 h6_0 h9_0 h11_0 h13_0 h25_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw9⟩ := familyGet_eq h9_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  subst hw2 hw4 hw6 hw9 hw11 hw13 hw25
  have ha' := ha ⟨i.val % 5, hk⟩
  -- The public multiplication at `Q6`.
  have hpub := mul_public (q := Q6) (m := 12) (m' := 10) (bound := 16384) hN (by norm_num)
    Backend.layout4 layoutQ6 rfl hG (hα _) ha' hs h4_0_1 h7_0 h11_0_1 h14_0 h15_0
    (Xs := ![w_18_0, w_33_0]) (fun j ↦ by fin_cases j; exacts [h18_0, h33_0])
    (D' := ![w_19_0, w_34_0]) (fun j ↦ by fin_cases j; exacts [h19_0, h34_0])
    (D := ![w_20_0, w_35_0]) (fun j ↦ by fin_cases j; exacts [h20_0, h35_0])
    (hpay _) (Y := ![w_22_0, w_37_0]) (A' := ![![w_28_0, w_41_0], ![w_29_0, w_42_0]])
    (fun j ↦ by fin_cases j; exacts [h22_0, h37_0]) (fun j ↦ by fin_cases j; exacts [h28_0, h41_0])
    (fun j ↦ by fin_cases j; exacts [h29_0, h42_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₆`.
  have htop : rhsTop model key Q5 8 smudge0Rhs0 i = ![w_46_0, w_54_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h46_0).symm
    · exact (hashSample_eq h54_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q6) (q' := Q5) (p := p5) (m := 10) (m' := 8)
    (bound := 16384) hN q6_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout3 layoutQ5 rfl
    hg' hg hsm hpub' h28_0_1 h30_0 h41_0_1 h43_0 h44_0 (top := ![w_46_0, w_54_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_48_0, w_56_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h56_0])
    (D' := ![w_49_0, w_57_0]) (fun j ↦ by fin_cases j; exacts [h49_0, h57_0])
    (D := ![w_50_0, w_58_0]) (fun j ↦ by fin_cases j; exacts [h50_0, h58_0])
    hrhs' heB (hαμ _) (X := ![w_66_0, w_68_0]) (M := ![![w_52_0, w_60_0], ![w_62_0, w_64_0]])
    (fun j ↦ by fin_cases j; exacts [h66_0, h68_0]) (fun j ↦ by fin_cases j; exacts [h52_0, h60_0])
    (fun j ↦ by fin_cases j; exacts [h62_0, h64_0])
    (c' := ![w_67_0, w_69_0]) (A' := ![![w_53_0, w_61_0], ![w_63_0, w_65_0]])
    (fun j ↦ by fin_cases j; exacts [h67_0, h69_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h53_0, h61_0, h63_0, h65_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring


set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `1` of the smudge0 PRF: an encoding of `p₄ (α_(i mod 5) μ_(i/5) ν_i)` at `Q0`. -/
theorem smudge0_cell1 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {As00 As10 As01 As11 cs0 cs1 : Fin 160 → ExactMatrix Q5 N 1 8}
    {bot0 bot1 : Fin 800 → ExactMatrix Q0 N 1 6} {i : Fin 800} {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_558 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp ν : ErrorPoly N} {μ : Fin 160 → ErrorPoly N} {α : Fin 5 → ErrorPoly N}
    (hsm : polyNorm sm ≤ 1) (hαμ : ∀ k o, polyNorm (α k * μ o) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q0 N 1 6} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (α k))
    {Bs : Nat} (hs : ∀ o : Fin 160, Rel sm ![![As00 o, As01 o], ![As10 o, As11 o]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * μ o * unitPair sp j) • g) ![cs0 o, cs1 o] Bs)
    {eB : Fin 2 → ErrorMatrix N 1 6} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q0 N sp • rhsTop model key Q0 6 smudge0Rhs1 i j + ![bot0 i, bot1 i] j =
      reducePoly Q0 N (((q3 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q0 N 1 6 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ *
        μ ⟨i.val / 5, by omega⟩ * ν) * unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((8 + 8) * N * ((10 + 10) * N * ((12 * N * Ba * (32768 / 2) + (N + 2) * (p5 / 2)) / p5) *
        16384 + N * 1 * Bs) * 16384 + (N + 2) * (p4 / 2)) / p4) + N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_558 at hcell
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_5_0, hcell⟩ := hcell
  extract_lets -merge w_6_0 at hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  extract_lets -merge w_8_0 at hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  extract_lets -merge w_15_0 at hcell
  obtain ⟨w_16_0, hcell⟩ := hcell
  extract_lets -merge w_17_0 at hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_23_0, hcell⟩ := hcell
  obtain ⟨w_23_concat_1, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 at hcell
  obtain ⟨w_27_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  extract_lets -merge w_30_0 at hcell
  obtain ⟨w_31_0, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  extract_lets -merge w_34_0 at hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  obtain ⟨w_36_0, hcell⟩ := hcell
  obtain ⟨w_36_concat_1, hcell⟩ := hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  obtain ⟨w_40_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_45_0, hcell⟩ := hcell
  extract_lets -merge w_46_0 at hcell
  obtain ⟨w_47_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_51_0, hcell⟩ := hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  extract_lets -merge w_53_0 w_54_0 at hcell
  obtain ⟨w_55_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_57_concat_1, hcell⟩ := hcell
  obtain ⟨w_59_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  extract_lets -merge w_65_0 at hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  extract_lets -merge w_69_0 w_70_0 at hcell
  obtain ⟨w_71_0, hcell⟩ := hcell
  obtain ⟨w_72_0, hcell⟩ := hcell
  obtain ⟨w_73_0, hcell⟩ := hcell
  obtain ⟨w_73_concat_1, hcell⟩ := hcell
  obtain ⟨w_74_0, hcell⟩ := hcell
  obtain ⟨w_76_0, hcell⟩ := hcell
  obtain ⟨w_78_0, hcell⟩ := hcell
  obtain ⟨w_79_0, hcell⟩ := hcell
  obtain ⟨w_80_0, hcell⟩ := hcell
  extract_lets -merge w_81_0 at hcell
  obtain ⟨w_82_0, hcell⟩ := hcell
  obtain ⟨w_83_0, hcell⟩ := hcell
  obtain ⟨w_84_0, hcell⟩ := hcell
  obtain ⟨w_86_0, hcell⟩ := hcell
  obtain ⟨w_87_0, hcell⟩ := hcell
  obtain ⟨w_88_0, hcell⟩ := hcell
  extract_lets -merge w_89_0 at hcell
  obtain ⟨w_90_0, hcell⟩ := hcell
  obtain ⟨w_91_0, hcell⟩ := hcell
  obtain ⟨w_92_0, hcell⟩ := hcell
  obtain ⟨w_93_0, hcell⟩ := hcell
  obtain ⟨w_94_0, hcell⟩ := hcell
  obtain ⟨w_95_0, hcell⟩ := hcell
  obtain ⟨w_96_0, hcell⟩ := hcell
  obtain ⟨w_97_0, hcell⟩ := hcell
  obtain ⟨w_98_0, hcell⟩ := hcell
  obtain ⟨w_99_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_558.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_558.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_558.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_558.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_558.constraints_4, Stage_online_eval_0_0.parallel_generatedRoot_558.constraints_5] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member `i / 5`, the member of `a`.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h11_0 h20_0 h25_0 h29_0 h33_0 h51_0
  have ho : w_39_0 = ((i.val / 5 : Nat) : Int) := fdiv_slot i.val
  rw [ho] at h40_0 h42_0 h49_0 h59_0 h61_0 h68_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  obtain ⟨_, hw51⟩ := familyGet_eq h51_0
  obtain ⟨hko, hw40⟩ := familyGet_eq h40_0
  obtain ⟨_, hw42⟩ := familyGet_eq h42_0
  obtain ⟨_, hw49⟩ := familyGet_eq h49_0
  obtain ⟨_, hw59⟩ := familyGet_eq h59_0
  obtain ⟨_, hw61⟩ := familyGet_eq h61_0
  obtain ⟨_, hw68⟩ := familyGet_eq h68_0
  subst hw2 hw11 hw20 hw25 hw29 hw33 hw51 hw40 hw42 hw49 hw59 hw61 hw68
  have ha' := ha ⟨i.val % 5, hk⟩
  have hs' := hs ⟨i.val / 5, hko⟩
  -- The alignment of the member from `Q6` to `Q5`; the row and the key share one decomposition.
  have hg13 := gadgetMatrixRuns_unique h13_0 h4_0
  subst hg13
  have hg14 := centeredRebaseRuns_unique h14_0 h5_0
  subst hg14
  have hH := gadgetDecomposeRuns_unique h16_0 h7_0
  subst hH
  have hal := align_down (q := Q6) (q' := Q5) (p := p5) (m := 12) (m' := 10) hN q6_eq (by norm_num)
    (by norm_num) (by norm_num) Backend.layout4 layoutQ6 rfl hG h5_0 hsm ha'
    (by
      funext r c
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp (config := { zetaDelta := true }) [Matrix.mul_apply, matrixPolynomial_single, map_ofNat])
    h16_0 (Y := ![_, _]) (K := ![![_, _], ![_, _]])
    (fun j ↦ by fin_cases j <;> rfl) (fun k j ↦ by fin_cases k <;> fin_cases j <;> rfl)
    (c' := ![w_9_0, w_27_0]) (A' := ![![w_18_0, w_31_0], ![w_22_0, w_35_0]])
    (fun j ↦ by fin_cases j; exacts [h9_0, h27_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h18_0, h31_0, h22_0, h35_0])
  -- The public multiplication at `Q5`.
  have hpayb := modulusReduce_poly (q := Q6) (p := Q5) (by norm_num) (by norm_num) hN
    (s := α ⟨i.val % 5, hk⟩)
    (by rw [← hpay]; exact h52_0)
  have hpub := mul_public (q := Q5) (m := 10) (m' := 8) (bound := 16384) hN (by norm_num)
    Backend.layout3 layoutQ5 rfl h4_0 (hα _) hal hs' h18_0_1 h23_0 h31_0_1 h36_0 h37_0
    (Xs := ![w_43_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h43_0, h62_0])
    (D' := ![w_44_0, w_63_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h63_0])
    (D := ![w_45_0, w_64_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    hpayb (Y := ![w_47_0, w_66_0]) (A' := ![![w_55_0, w_71_0], ![w_56_0, w_72_0]])
    (fun j ↦ by fin_cases j; exacts [h47_0, h66_0]) (fun j ↦ by fin_cases j; exacts [h55_0, h71_0])
    (fun j ↦ by fin_cases j; exacts [h56_0, h72_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ ⟨i.val / 5, hko⟩) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₅`.
  have htop : rhsTop model key Q0 6 smudge0Rhs1 i = ![w_76_0, w_84_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h76_0).symm
    · exact (hashSample_eq h84_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q5) (q' := Q0) (p := p4) (m := 8) (m' := 6)
    (bound := 16384) hN q5_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout2 layoutQ0 rfl
    hg' hg hsm hpub' h55_0_1 h57_0 h71_0_1 h73_0 h74_0 (top := ![w_76_0, w_84_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_78_0, w_86_0]) (fun j ↦ by fin_cases j; exacts [h78_0, h86_0])
    (D' := ![w_79_0, w_87_0]) (fun j ↦ by fin_cases j; exacts [h79_0, h87_0])
    (D := ![w_80_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h80_0, h88_0])
    hrhs' heB (hαμ _ _) (X := ![w_96_0, w_98_0]) (M := ![![w_82_0, w_90_0], ![w_92_0, w_94_0]])
    (fun j ↦ by fin_cases j; exacts [h96_0, h98_0]) (fun j ↦ by fin_cases j; exacts [h82_0, h90_0])
    (fun j ↦ by fin_cases j; exacts [h92_0, h94_0])
    (c' := ![w_97_0, w_99_0]) (A' := ![![w_83_0, w_91_0], ![w_93_0, w_95_0]])
    (fun j ↦ by fin_cases j; exacts [h97_0, h99_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h83_0, h91_0, h93_0, h95_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring
def smudge1Rhs0 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49]
def smudge1Rhs1 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `0` of the smudge1 PRF: an encoding of `p₅ (α_(i mod 5) μ ν_i)` at `Q5`. -/
theorem smudge1_cell0 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {As : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10}
    {cs : Fin 2 → ExactMatrix Q6 N 1 10} {bot0 bot1 : Fin 800 → ExactMatrix Q5 N 1 8} {i : Fin 800}
    {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_609 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As 0 0, As 1 0, cs 0, apay, As 0 1, As 1 1, cs 1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp μ ν : ErrorPoly N} {α : Fin 5 → ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (α k))
    {Bs : Nat} (hs : Rel sm As (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * μ * unitPair sp j) • g)
      cs Bs)
    {eB : Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q5 N sp • rhsTop model key Q5 8 smudge1Rhs0 i j + ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q5 N 1 8 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ * μ * ν) *
        unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((10 + 10) * N * ((12 + 12) * N * Ba * 16384 + N * 1 * Bs) * 16384 + (N + 2) * (p5 / 2)) / p5) +
        N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_609 at hcell
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_6_0, hcell⟩ := hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  obtain ⟨w_7_concat_1, hcell⟩ := hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  obtain ⟨w_14_concat_1, hcell⟩ := hcell
  obtain ⟨w_15_0, hcell⟩ := hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_19_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hcell
  obtain ⟨w_28_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  obtain ⟨w_30_0, hcell⟩ := hcell
  obtain ⟨w_30_concat_1, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  obtain ⟨w_34_0, hcell⟩ := hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  extract_lets -merge w_36_0 at hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  extract_lets -merge w_40_0 at hcell
  obtain ⟨w_41_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_43_concat_1, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_46_0, hcell⟩ := hcell
  obtain ⟨w_48_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_50_0, hcell⟩ := hcell
  extract_lets -merge w_51_0 at hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  obtain ⟨w_53_0, hcell⟩ := hcell
  obtain ⟨w_54_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_58_0, hcell⟩ := hcell
  extract_lets -merge w_59_0 at hcell
  obtain ⟨w_60_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  obtain ⟨w_65_0, hcell⟩ := hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_67_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  obtain ⟨w_69_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_609.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_609.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_609.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_609.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_609.constraints_4] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member of `a`, and its public key.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h4_0 h6_0 h9_0 h11_0 h13_0 h25_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw9⟩ := familyGet_eq h9_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  subst hw2 hw4 hw6 hw9 hw11 hw13 hw25
  have ha' := ha ⟨i.val % 5, hk⟩
  -- The public multiplication at `Q6`.
  have hpub := mul_public (q := Q6) (m := 12) (m' := 10) (bound := 16384) hN (by norm_num)
    Backend.layout4 layoutQ6 rfl hG (hα _) ha' hs h4_0_1 h7_0 h11_0_1 h14_0 h15_0
    (Xs := ![w_18_0, w_33_0]) (fun j ↦ by fin_cases j; exacts [h18_0, h33_0])
    (D' := ![w_19_0, w_34_0]) (fun j ↦ by fin_cases j; exacts [h19_0, h34_0])
    (D := ![w_20_0, w_35_0]) (fun j ↦ by fin_cases j; exacts [h20_0, h35_0])
    (hpay _) (Y := ![w_22_0, w_37_0]) (A' := ![![w_28_0, w_41_0], ![w_29_0, w_42_0]])
    (fun j ↦ by fin_cases j; exacts [h22_0, h37_0]) (fun j ↦ by fin_cases j; exacts [h28_0, h41_0])
    (fun j ↦ by fin_cases j; exacts [h29_0, h42_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₆`.
  have htop : rhsTop model key Q5 8 smudge1Rhs0 i = ![w_46_0, w_54_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h46_0).symm
    · exact (hashSample_eq h54_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q6) (q' := Q5) (p := p5) (m := 10) (m' := 8)
    (bound := 16384) hN q6_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout3 layoutQ5 rfl
    hg' hg hsm hpub' h28_0_1 h30_0 h41_0_1 h43_0 h44_0 (top := ![w_46_0, w_54_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_48_0, w_56_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h56_0])
    (D' := ![w_49_0, w_57_0]) (fun j ↦ by fin_cases j; exacts [h49_0, h57_0])
    (D := ![w_50_0, w_58_0]) (fun j ↦ by fin_cases j; exacts [h50_0, h58_0])
    hrhs' heB (hαμ _) (X := ![w_66_0, w_68_0]) (M := ![![w_52_0, w_60_0], ![w_62_0, w_64_0]])
    (fun j ↦ by fin_cases j; exacts [h66_0, h68_0]) (fun j ↦ by fin_cases j; exacts [h52_0, h60_0])
    (fun j ↦ by fin_cases j; exacts [h62_0, h64_0])
    (c' := ![w_67_0, w_69_0]) (A' := ![![w_53_0, w_61_0], ![w_63_0, w_65_0]])
    (fun j ↦ by fin_cases j; exacts [h67_0, h69_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h53_0, h61_0, h63_0, h65_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring


set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `1` of the smudge1 PRF: an encoding of `p₄ (α_(i mod 5) μ_(i/5) ν_i)` at `Q0`. -/
theorem smudge1_cell1 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {As00 As10 As01 As11 cs0 cs1 : Fin 160 → ExactMatrix Q5 N 1 8}
    {bot0 bot1 : Fin 800 → ExactMatrix Q0 N 1 6} {i : Fin 800} {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_613 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp ν : ErrorPoly N} {μ : Fin 160 → ErrorPoly N} {α : Fin 5 → ErrorPoly N}
    (hsm : polyNorm sm ≤ 1) (hαμ : ∀ k o, polyNorm (α k * μ o) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q0 N 1 6} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (α k))
    {Bs : Nat} (hs : ∀ o : Fin 160, Rel sm ![![As00 o, As01 o], ![As10 o, As11 o]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * μ o * unitPair sp j) • g) ![cs0 o, cs1 o] Bs)
    {eB : Fin 2 → ErrorMatrix N 1 6} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q0 N sp • rhsTop model key Q0 6 smudge1Rhs1 i j + ![bot0 i, bot1 i] j =
      reducePoly Q0 N (((q3 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q0 N 1 6 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ *
        μ ⟨i.val / 5, by omega⟩ * ν) * unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((8 + 8) * N * ((10 + 10) * N * ((12 * N * Ba * (32768 / 2) + (N + 2) * (p5 / 2)) / p5) *
        16384 + N * 1 * Bs) * 16384 + (N + 2) * (p4 / 2)) / p4) + N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_613 at hcell
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_5_0, hcell⟩ := hcell
  extract_lets -merge w_6_0 at hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  extract_lets -merge w_8_0 at hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  extract_lets -merge w_15_0 at hcell
  obtain ⟨w_16_0, hcell⟩ := hcell
  extract_lets -merge w_17_0 at hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_23_0, hcell⟩ := hcell
  obtain ⟨w_23_concat_1, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 at hcell
  obtain ⟨w_27_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  extract_lets -merge w_30_0 at hcell
  obtain ⟨w_31_0, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  extract_lets -merge w_34_0 at hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  obtain ⟨w_36_0, hcell⟩ := hcell
  obtain ⟨w_36_concat_1, hcell⟩ := hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  obtain ⟨w_40_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_45_0, hcell⟩ := hcell
  extract_lets -merge w_46_0 at hcell
  obtain ⟨w_47_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_51_0, hcell⟩ := hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  extract_lets -merge w_53_0 w_54_0 at hcell
  obtain ⟨w_55_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_57_concat_1, hcell⟩ := hcell
  obtain ⟨w_59_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  extract_lets -merge w_65_0 at hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  extract_lets -merge w_69_0 w_70_0 at hcell
  obtain ⟨w_71_0, hcell⟩ := hcell
  obtain ⟨w_72_0, hcell⟩ := hcell
  obtain ⟨w_73_0, hcell⟩ := hcell
  obtain ⟨w_73_concat_1, hcell⟩ := hcell
  obtain ⟨w_74_0, hcell⟩ := hcell
  obtain ⟨w_76_0, hcell⟩ := hcell
  obtain ⟨w_78_0, hcell⟩ := hcell
  obtain ⟨w_79_0, hcell⟩ := hcell
  obtain ⟨w_80_0, hcell⟩ := hcell
  extract_lets -merge w_81_0 at hcell
  obtain ⟨w_82_0, hcell⟩ := hcell
  obtain ⟨w_83_0, hcell⟩ := hcell
  obtain ⟨w_84_0, hcell⟩ := hcell
  obtain ⟨w_86_0, hcell⟩ := hcell
  obtain ⟨w_87_0, hcell⟩ := hcell
  obtain ⟨w_88_0, hcell⟩ := hcell
  extract_lets -merge w_89_0 at hcell
  obtain ⟨w_90_0, hcell⟩ := hcell
  obtain ⟨w_91_0, hcell⟩ := hcell
  obtain ⟨w_92_0, hcell⟩ := hcell
  obtain ⟨w_93_0, hcell⟩ := hcell
  obtain ⟨w_94_0, hcell⟩ := hcell
  obtain ⟨w_95_0, hcell⟩ := hcell
  obtain ⟨w_96_0, hcell⟩ := hcell
  obtain ⟨w_97_0, hcell⟩ := hcell
  obtain ⟨w_98_0, hcell⟩ := hcell
  obtain ⟨w_99_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_613.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_613.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_613.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_613.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_613.constraints_4, Stage_online_eval_0_0.parallel_generatedRoot_613.constraints_5] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member `i / 5`, the member of `a`.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h11_0 h20_0 h25_0 h29_0 h33_0 h51_0
  have ho : w_39_0 = ((i.val / 5 : Nat) : Int) := fdiv_slot i.val
  rw [ho] at h40_0 h42_0 h49_0 h59_0 h61_0 h68_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  obtain ⟨_, hw51⟩ := familyGet_eq h51_0
  obtain ⟨hko, hw40⟩ := familyGet_eq h40_0
  obtain ⟨_, hw42⟩ := familyGet_eq h42_0
  obtain ⟨_, hw49⟩ := familyGet_eq h49_0
  obtain ⟨_, hw59⟩ := familyGet_eq h59_0
  obtain ⟨_, hw61⟩ := familyGet_eq h61_0
  obtain ⟨_, hw68⟩ := familyGet_eq h68_0
  subst hw2 hw11 hw20 hw25 hw29 hw33 hw51 hw40 hw42 hw49 hw59 hw61 hw68
  have ha' := ha ⟨i.val % 5, hk⟩
  have hs' := hs ⟨i.val / 5, hko⟩
  -- The alignment of the member from `Q6` to `Q5`; the row and the key share one decomposition.
  have hg13 := gadgetMatrixRuns_unique h13_0 h4_0
  subst hg13
  have hg14 := centeredRebaseRuns_unique h14_0 h5_0
  subst hg14
  have hH := gadgetDecomposeRuns_unique h16_0 h7_0
  subst hH
  have hal := align_down (q := Q6) (q' := Q5) (p := p5) (m := 12) (m' := 10) hN q6_eq (by norm_num)
    (by norm_num) (by norm_num) Backend.layout4 layoutQ6 rfl hG h5_0 hsm ha'
    (by
      funext r c
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp (config := { zetaDelta := true }) [Matrix.mul_apply, matrixPolynomial_single, map_ofNat])
    h16_0 (Y := ![_, _]) (K := ![![_, _], ![_, _]])
    (fun j ↦ by fin_cases j <;> rfl) (fun k j ↦ by fin_cases k <;> fin_cases j <;> rfl)
    (c' := ![w_9_0, w_27_0]) (A' := ![![w_18_0, w_31_0], ![w_22_0, w_35_0]])
    (fun j ↦ by fin_cases j; exacts [h9_0, h27_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h18_0, h31_0, h22_0, h35_0])
  -- The public multiplication at `Q5`.
  have hpayb := modulusReduce_poly (q := Q6) (p := Q5) (by norm_num) (by norm_num) hN
    (s := α ⟨i.val % 5, hk⟩)
    (by rw [← hpay]; exact h52_0)
  have hpub := mul_public (q := Q5) (m := 10) (m' := 8) (bound := 16384) hN (by norm_num)
    Backend.layout3 layoutQ5 rfl h4_0 (hα _) hal hs' h18_0_1 h23_0 h31_0_1 h36_0 h37_0
    (Xs := ![w_43_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h43_0, h62_0])
    (D' := ![w_44_0, w_63_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h63_0])
    (D := ![w_45_0, w_64_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    hpayb (Y := ![w_47_0, w_66_0]) (A' := ![![w_55_0, w_71_0], ![w_56_0, w_72_0]])
    (fun j ↦ by fin_cases j; exacts [h47_0, h66_0]) (fun j ↦ by fin_cases j; exacts [h55_0, h71_0])
    (fun j ↦ by fin_cases j; exacts [h56_0, h72_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ ⟨i.val / 5, hko⟩) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₅`.
  have htop : rhsTop model key Q0 6 smudge1Rhs1 i = ![w_76_0, w_84_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h76_0).symm
    · exact (hashSample_eq h84_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q5) (q' := Q0) (p := p4) (m := 8) (m' := 6)
    (bound := 16384) hN q5_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout2 layoutQ0 rfl
    hg' hg hsm hpub' h55_0_1 h57_0 h71_0_1 h73_0 h74_0 (top := ![w_76_0, w_84_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_78_0, w_86_0]) (fun j ↦ by fin_cases j; exacts [h78_0, h86_0])
    (D' := ![w_79_0, w_87_0]) (fun j ↦ by fin_cases j; exacts [h79_0, h87_0])
    (D := ![w_80_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h80_0, h88_0])
    hrhs' heB (hαμ _ _) (X := ![w_96_0, w_98_0]) (M := ![![w_82_0, w_90_0], ![w_92_0, w_94_0]])
    (fun j ↦ by fin_cases j; exacts [h96_0, h98_0]) (fun j ↦ by fin_cases j; exacts [h82_0, h90_0])
    (fun j ↦ by fin_cases j; exacts [h92_0, h94_0])
    (c' := ![w_97_0, w_99_0]) (A' := ![![w_83_0, w_91_0], ![w_93_0, w_95_0]])
    (fun j ↦ by fin_cases j; exacts [h97_0, h99_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h83_0, h91_0, h93_0, h95_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring
def smudge2Rhs0 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49]
def smudge2Rhs1 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `0` of the smudge2 PRF: an encoding of `p₅ (α_(i mod 5) μ ν_i)` at `Q5`. -/
theorem smudge2_cell0 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {As : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10}
    {cs : Fin 2 → ExactMatrix Q6 N 1 10} {bot0 bot1 : Fin 800 → ExactMatrix Q5 N 1 8} {i : Fin 800}
    {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_663 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As 0 0, As 1 0, cs 0, apay, As 0 1, As 1 1, cs 1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp μ ν : ErrorPoly N} {α : Fin 5 → ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (α k))
    {Bs : Nat} (hs : Rel sm As (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * μ * unitPair sp j) • g)
      cs Bs)
    {eB : Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q5 N sp • rhsTop model key Q5 8 smudge2Rhs0 i j + ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q5 N 1 8 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ * μ * ν) *
        unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((10 + 10) * N * ((12 + 12) * N * Ba * 16384 + N * 1 * Bs) * 16384 + (N + 2) * (p5 / 2)) / p5) +
        N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_663 at hcell
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_6_0, hcell⟩ := hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  obtain ⟨w_7_concat_1, hcell⟩ := hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  obtain ⟨w_14_concat_1, hcell⟩ := hcell
  obtain ⟨w_15_0, hcell⟩ := hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_19_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hcell
  obtain ⟨w_28_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  obtain ⟨w_30_0, hcell⟩ := hcell
  obtain ⟨w_30_concat_1, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  obtain ⟨w_34_0, hcell⟩ := hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  extract_lets -merge w_36_0 at hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  extract_lets -merge w_40_0 at hcell
  obtain ⟨w_41_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_43_concat_1, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_46_0, hcell⟩ := hcell
  obtain ⟨w_48_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_50_0, hcell⟩ := hcell
  extract_lets -merge w_51_0 at hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  obtain ⟨w_53_0, hcell⟩ := hcell
  obtain ⟨w_54_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_58_0, hcell⟩ := hcell
  extract_lets -merge w_59_0 at hcell
  obtain ⟨w_60_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  obtain ⟨w_65_0, hcell⟩ := hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_67_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  obtain ⟨w_69_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_663.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_663.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_663.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_663.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_663.constraints_4] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member of `a`, and its public key.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h4_0 h6_0 h9_0 h11_0 h13_0 h25_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw9⟩ := familyGet_eq h9_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  subst hw2 hw4 hw6 hw9 hw11 hw13 hw25
  have ha' := ha ⟨i.val % 5, hk⟩
  -- The public multiplication at `Q6`.
  have hpub := mul_public (q := Q6) (m := 12) (m' := 10) (bound := 16384) hN (by norm_num)
    Backend.layout4 layoutQ6 rfl hG (hα _) ha' hs h4_0_1 h7_0 h11_0_1 h14_0 h15_0
    (Xs := ![w_18_0, w_33_0]) (fun j ↦ by fin_cases j; exacts [h18_0, h33_0])
    (D' := ![w_19_0, w_34_0]) (fun j ↦ by fin_cases j; exacts [h19_0, h34_0])
    (D := ![w_20_0, w_35_0]) (fun j ↦ by fin_cases j; exacts [h20_0, h35_0])
    (hpay _) (Y := ![w_22_0, w_37_0]) (A' := ![![w_28_0, w_41_0], ![w_29_0, w_42_0]])
    (fun j ↦ by fin_cases j; exacts [h22_0, h37_0]) (fun j ↦ by fin_cases j; exacts [h28_0, h41_0])
    (fun j ↦ by fin_cases j; exacts [h29_0, h42_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₆`.
  have htop : rhsTop model key Q5 8 smudge2Rhs0 i = ![w_46_0, w_54_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h46_0).symm
    · exact (hashSample_eq h54_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q6) (q' := Q5) (p := p5) (m := 10) (m' := 8)
    (bound := 16384) hN q6_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout3 layoutQ5 rfl
    hg' hg hsm hpub' h28_0_1 h30_0 h41_0_1 h43_0 h44_0 (top := ![w_46_0, w_54_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_48_0, w_56_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h56_0])
    (D' := ![w_49_0, w_57_0]) (fun j ↦ by fin_cases j; exacts [h49_0, h57_0])
    (D := ![w_50_0, w_58_0]) (fun j ↦ by fin_cases j; exacts [h50_0, h58_0])
    hrhs' heB (hαμ _) (X := ![w_66_0, w_68_0]) (M := ![![w_52_0, w_60_0], ![w_62_0, w_64_0]])
    (fun j ↦ by fin_cases j; exacts [h66_0, h68_0]) (fun j ↦ by fin_cases j; exacts [h52_0, h60_0])
    (fun j ↦ by fin_cases j; exacts [h62_0, h64_0])
    (c' := ![w_67_0, w_69_0]) (A' := ![![w_53_0, w_61_0], ![w_63_0, w_65_0]])
    (fun j ↦ by fin_cases j; exacts [h67_0, h69_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h53_0, h61_0, h63_0, h65_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring


set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `1` of the smudge2 PRF: an encoding of `p₄ (α_(i mod 5) μ_(i/5) ν_i)` at `Q0`. -/
theorem smudge2_cell1 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {As00 As10 As01 As11 cs0 cs1 : Fin 160 → ExactMatrix Q5 N 1 8}
    {bot0 bot1 : Fin 800 → ExactMatrix Q0 N 1 6} {i : Fin 800} {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_667 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp ν : ErrorPoly N} {μ : Fin 160 → ErrorPoly N} {α : Fin 5 → ErrorPoly N}
    (hsm : polyNorm sm ≤ 1) (hαμ : ∀ k o, polyNorm (α k * μ o) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q0 N 1 6} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (α k))
    {Bs : Nat} (hs : ∀ o : Fin 160, Rel sm ![![As00 o, As01 o], ![As10 o, As11 o]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * μ o * unitPair sp j) • g) ![cs0 o, cs1 o] Bs)
    {eB : Fin 2 → ErrorMatrix N 1 6} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q0 N sp • rhsTop model key Q0 6 smudge2Rhs1 i j + ![bot0 i, bot1 i] j =
      reducePoly Q0 N (((q3 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q0 N 1 6 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ *
        μ ⟨i.val / 5, by omega⟩ * ν) * unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((8 + 8) * N * ((10 + 10) * N * ((12 * N * Ba * (32768 / 2) + (N + 2) * (p5 / 2)) / p5) *
        16384 + N * 1 * Bs) * 16384 + (N + 2) * (p4 / 2)) / p4) + N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_667 at hcell
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_5_0, hcell⟩ := hcell
  extract_lets -merge w_6_0 at hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  extract_lets -merge w_8_0 at hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  extract_lets -merge w_15_0 at hcell
  obtain ⟨w_16_0, hcell⟩ := hcell
  extract_lets -merge w_17_0 at hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_23_0, hcell⟩ := hcell
  obtain ⟨w_23_concat_1, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 at hcell
  obtain ⟨w_27_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  extract_lets -merge w_30_0 at hcell
  obtain ⟨w_31_0, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  extract_lets -merge w_34_0 at hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  obtain ⟨w_36_0, hcell⟩ := hcell
  obtain ⟨w_36_concat_1, hcell⟩ := hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  obtain ⟨w_40_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_45_0, hcell⟩ := hcell
  extract_lets -merge w_46_0 at hcell
  obtain ⟨w_47_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_51_0, hcell⟩ := hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  extract_lets -merge w_53_0 w_54_0 at hcell
  obtain ⟨w_55_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_57_concat_1, hcell⟩ := hcell
  obtain ⟨w_59_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  extract_lets -merge w_65_0 at hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  extract_lets -merge w_69_0 w_70_0 at hcell
  obtain ⟨w_71_0, hcell⟩ := hcell
  obtain ⟨w_72_0, hcell⟩ := hcell
  obtain ⟨w_73_0, hcell⟩ := hcell
  obtain ⟨w_73_concat_1, hcell⟩ := hcell
  obtain ⟨w_74_0, hcell⟩ := hcell
  obtain ⟨w_76_0, hcell⟩ := hcell
  obtain ⟨w_78_0, hcell⟩ := hcell
  obtain ⟨w_79_0, hcell⟩ := hcell
  obtain ⟨w_80_0, hcell⟩ := hcell
  extract_lets -merge w_81_0 at hcell
  obtain ⟨w_82_0, hcell⟩ := hcell
  obtain ⟨w_83_0, hcell⟩ := hcell
  obtain ⟨w_84_0, hcell⟩ := hcell
  obtain ⟨w_86_0, hcell⟩ := hcell
  obtain ⟨w_87_0, hcell⟩ := hcell
  obtain ⟨w_88_0, hcell⟩ := hcell
  extract_lets -merge w_89_0 at hcell
  obtain ⟨w_90_0, hcell⟩ := hcell
  obtain ⟨w_91_0, hcell⟩ := hcell
  obtain ⟨w_92_0, hcell⟩ := hcell
  obtain ⟨w_93_0, hcell⟩ := hcell
  obtain ⟨w_94_0, hcell⟩ := hcell
  obtain ⟨w_95_0, hcell⟩ := hcell
  obtain ⟨w_96_0, hcell⟩ := hcell
  obtain ⟨w_97_0, hcell⟩ := hcell
  obtain ⟨w_98_0, hcell⟩ := hcell
  obtain ⟨w_99_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_667.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_667.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_667.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_667.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_667.constraints_4, Stage_online_eval_0_0.parallel_generatedRoot_667.constraints_5] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member `i / 5`, the member of `a`.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h11_0 h20_0 h25_0 h29_0 h33_0 h51_0
  have ho : w_39_0 = ((i.val / 5 : Nat) : Int) := fdiv_slot i.val
  rw [ho] at h40_0 h42_0 h49_0 h59_0 h61_0 h68_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  obtain ⟨_, hw51⟩ := familyGet_eq h51_0
  obtain ⟨hko, hw40⟩ := familyGet_eq h40_0
  obtain ⟨_, hw42⟩ := familyGet_eq h42_0
  obtain ⟨_, hw49⟩ := familyGet_eq h49_0
  obtain ⟨_, hw59⟩ := familyGet_eq h59_0
  obtain ⟨_, hw61⟩ := familyGet_eq h61_0
  obtain ⟨_, hw68⟩ := familyGet_eq h68_0
  subst hw2 hw11 hw20 hw25 hw29 hw33 hw51 hw40 hw42 hw49 hw59 hw61 hw68
  have ha' := ha ⟨i.val % 5, hk⟩
  have hs' := hs ⟨i.val / 5, hko⟩
  -- The alignment of the member from `Q6` to `Q5`; the row and the key share one decomposition.
  have hg13 := gadgetMatrixRuns_unique h13_0 h4_0
  subst hg13
  have hg14 := centeredRebaseRuns_unique h14_0 h5_0
  subst hg14
  have hH := gadgetDecomposeRuns_unique h16_0 h7_0
  subst hH
  have hal := align_down (q := Q6) (q' := Q5) (p := p5) (m := 12) (m' := 10) hN q6_eq (by norm_num)
    (by norm_num) (by norm_num) Backend.layout4 layoutQ6 rfl hG h5_0 hsm ha'
    (by
      funext r c
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp (config := { zetaDelta := true }) [Matrix.mul_apply, matrixPolynomial_single, map_ofNat])
    h16_0 (Y := ![_, _]) (K := ![![_, _], ![_, _]])
    (fun j ↦ by fin_cases j <;> rfl) (fun k j ↦ by fin_cases k <;> fin_cases j <;> rfl)
    (c' := ![w_9_0, w_27_0]) (A' := ![![w_18_0, w_31_0], ![w_22_0, w_35_0]])
    (fun j ↦ by fin_cases j; exacts [h9_0, h27_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h18_0, h31_0, h22_0, h35_0])
  -- The public multiplication at `Q5`.
  have hpayb := modulusReduce_poly (q := Q6) (p := Q5) (by norm_num) (by norm_num) hN
    (s := α ⟨i.val % 5, hk⟩)
    (by rw [← hpay]; exact h52_0)
  have hpub := mul_public (q := Q5) (m := 10) (m' := 8) (bound := 16384) hN (by norm_num)
    Backend.layout3 layoutQ5 rfl h4_0 (hα _) hal hs' h18_0_1 h23_0 h31_0_1 h36_0 h37_0
    (Xs := ![w_43_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h43_0, h62_0])
    (D' := ![w_44_0, w_63_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h63_0])
    (D := ![w_45_0, w_64_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    hpayb (Y := ![w_47_0, w_66_0]) (A' := ![![w_55_0, w_71_0], ![w_56_0, w_72_0]])
    (fun j ↦ by fin_cases j; exacts [h47_0, h66_0]) (fun j ↦ by fin_cases j; exacts [h55_0, h71_0])
    (fun j ↦ by fin_cases j; exacts [h56_0, h72_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ ⟨i.val / 5, hko⟩) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₅`.
  have htop : rhsTop model key Q0 6 smudge2Rhs1 i = ![w_76_0, w_84_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h76_0).symm
    · exact (hashSample_eq h84_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q5) (q' := Q0) (p := p4) (m := 8) (m' := 6)
    (bound := 16384) hN q5_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout2 layoutQ0 rfl
    hg' hg hsm hpub' h55_0_1 h57_0 h71_0_1 h73_0 h74_0 (top := ![w_76_0, w_84_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_78_0, w_86_0]) (fun j ↦ by fin_cases j; exacts [h78_0, h86_0])
    (D' := ![w_79_0, w_87_0]) (fun j ↦ by fin_cases j; exacts [h79_0, h87_0])
    (D := ![w_80_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h80_0, h88_0])
    hrhs' heB (hαμ _ _) (X := ![w_96_0, w_98_0]) (M := ![![w_82_0, w_90_0], ![w_92_0, w_94_0]])
    (fun j ↦ by fin_cases j; exacts [h96_0, h98_0]) (fun j ↦ by fin_cases j; exacts [h82_0, h90_0])
    (fun j ↦ by fin_cases j; exacts [h92_0, h94_0])
    (c' := ![w_97_0, w_99_0]) (A' := ![![w_83_0, w_91_0], ![w_93_0, w_95_0]])
    (fun j ↦ by fin_cases j; exacts [h97_0, h99_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h83_0, h91_0, h93_0, h95_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring
def smudge3Rhs0 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49]
def smudge3Rhs1 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `0` of the smudge3 PRF: an encoding of `p₅ (α_(i mod 5) μ ν_i)` at `Q5`. -/
theorem smudge3_cell0 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {As : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10}
    {cs : Fin 2 → ExactMatrix Q6 N 1 10} {bot0 bot1 : Fin 800 → ExactMatrix Q5 N 1 8} {i : Fin 800}
    {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_717 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As 0 0, As 1 0, cs 0, apay, As 0 1, As 1 1, cs 1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp μ ν : ErrorPoly N} {α : Fin 5 → ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (α k))
    {Bs : Nat} (hs : Rel sm As (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * μ * unitPair sp j) • g)
      cs Bs)
    {eB : Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q5 N sp • rhsTop model key Q5 8 smudge3Rhs0 i j + ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q5 N 1 8 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ * μ * ν) *
        unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((10 + 10) * N * ((12 + 12) * N * Ba * 16384 + N * 1 * Bs) * 16384 + (N + 2) * (p5 / 2)) / p5) +
        N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_717 at hcell
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_6_0, hcell⟩ := hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  obtain ⟨w_7_concat_1, hcell⟩ := hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  obtain ⟨w_14_concat_1, hcell⟩ := hcell
  obtain ⟨w_15_0, hcell⟩ := hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_19_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hcell
  obtain ⟨w_28_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  obtain ⟨w_30_0, hcell⟩ := hcell
  obtain ⟨w_30_concat_1, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  obtain ⟨w_34_0, hcell⟩ := hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  extract_lets -merge w_36_0 at hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  extract_lets -merge w_40_0 at hcell
  obtain ⟨w_41_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_43_concat_1, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_46_0, hcell⟩ := hcell
  obtain ⟨w_48_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_50_0, hcell⟩ := hcell
  extract_lets -merge w_51_0 at hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  obtain ⟨w_53_0, hcell⟩ := hcell
  obtain ⟨w_54_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_58_0, hcell⟩ := hcell
  extract_lets -merge w_59_0 at hcell
  obtain ⟨w_60_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  obtain ⟨w_65_0, hcell⟩ := hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_67_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  obtain ⟨w_69_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_717.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_717.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_717.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_717.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_717.constraints_4] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member of `a`, and its public key.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h4_0 h6_0 h9_0 h11_0 h13_0 h25_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw9⟩ := familyGet_eq h9_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  subst hw2 hw4 hw6 hw9 hw11 hw13 hw25
  have ha' := ha ⟨i.val % 5, hk⟩
  -- The public multiplication at `Q6`.
  have hpub := mul_public (q := Q6) (m := 12) (m' := 10) (bound := 16384) hN (by norm_num)
    Backend.layout4 layoutQ6 rfl hG (hα _) ha' hs h4_0_1 h7_0 h11_0_1 h14_0 h15_0
    (Xs := ![w_18_0, w_33_0]) (fun j ↦ by fin_cases j; exacts [h18_0, h33_0])
    (D' := ![w_19_0, w_34_0]) (fun j ↦ by fin_cases j; exacts [h19_0, h34_0])
    (D := ![w_20_0, w_35_0]) (fun j ↦ by fin_cases j; exacts [h20_0, h35_0])
    (hpay _) (Y := ![w_22_0, w_37_0]) (A' := ![![w_28_0, w_41_0], ![w_29_0, w_42_0]])
    (fun j ↦ by fin_cases j; exacts [h22_0, h37_0]) (fun j ↦ by fin_cases j; exacts [h28_0, h41_0])
    (fun j ↦ by fin_cases j; exacts [h29_0, h42_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₆`.
  have htop : rhsTop model key Q5 8 smudge3Rhs0 i = ![w_46_0, w_54_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h46_0).symm
    · exact (hashSample_eq h54_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q6) (q' := Q5) (p := p5) (m := 10) (m' := 8)
    (bound := 16384) hN q6_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout3 layoutQ5 rfl
    hg' hg hsm hpub' h28_0_1 h30_0 h41_0_1 h43_0 h44_0 (top := ![w_46_0, w_54_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_48_0, w_56_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h56_0])
    (D' := ![w_49_0, w_57_0]) (fun j ↦ by fin_cases j; exacts [h49_0, h57_0])
    (D := ![w_50_0, w_58_0]) (fun j ↦ by fin_cases j; exacts [h50_0, h58_0])
    hrhs' heB (hαμ _) (X := ![w_66_0, w_68_0]) (M := ![![w_52_0, w_60_0], ![w_62_0, w_64_0]])
    (fun j ↦ by fin_cases j; exacts [h66_0, h68_0]) (fun j ↦ by fin_cases j; exacts [h52_0, h60_0])
    (fun j ↦ by fin_cases j; exacts [h62_0, h64_0])
    (c' := ![w_67_0, w_69_0]) (A' := ![![w_53_0, w_61_0], ![w_63_0, w_65_0]])
    (fun j ↦ by fin_cases j; exacts [h67_0, h69_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h53_0, h61_0, h63_0, h65_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring


set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `1` of the smudge3 PRF: an encoding of `p₄ (α_(i mod 5) μ_(i/5) ν_i)` at `Q0`. -/
theorem smudge3_cell1 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {As00 As10 As01 As11 cs0 cs1 : Fin 160 → ExactMatrix Q5 N 1 8}
    {bot0 bot1 : Fin 800 → ExactMatrix Q0 N 1 6} {i : Fin 800} {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_721 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp ν : ErrorPoly N} {μ : Fin 160 → ErrorPoly N} {α : Fin 5 → ErrorPoly N}
    (hsm : polyNorm sm ≤ 1) (hαμ : ∀ k o, polyNorm (α k * μ o) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q0 N 1 6} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (α k))
    {Bs : Nat} (hs : ∀ o : Fin 160, Rel sm ![![As00 o, As01 o], ![As10 o, As11 o]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * μ o * unitPair sp j) • g) ![cs0 o, cs1 o] Bs)
    {eB : Fin 2 → ErrorMatrix N 1 6} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q0 N sp • rhsTop model key Q0 6 smudge3Rhs1 i j + ![bot0 i, bot1 i] j =
      reducePoly Q0 N (((q3 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q0 N 1 6 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ *
        μ ⟨i.val / 5, by omega⟩ * ν) * unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((8 + 8) * N * ((10 + 10) * N * ((12 * N * Ba * (32768 / 2) + (N + 2) * (p5 / 2)) / p5) *
        16384 + N * 1 * Bs) * 16384 + (N + 2) * (p4 / 2)) / p4) + N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_721 at hcell
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_5_0, hcell⟩ := hcell
  extract_lets -merge w_6_0 at hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  extract_lets -merge w_8_0 at hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  extract_lets -merge w_15_0 at hcell
  obtain ⟨w_16_0, hcell⟩ := hcell
  extract_lets -merge w_17_0 at hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_23_0, hcell⟩ := hcell
  obtain ⟨w_23_concat_1, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 at hcell
  obtain ⟨w_27_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  extract_lets -merge w_30_0 at hcell
  obtain ⟨w_31_0, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  extract_lets -merge w_34_0 at hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  obtain ⟨w_36_0, hcell⟩ := hcell
  obtain ⟨w_36_concat_1, hcell⟩ := hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  obtain ⟨w_40_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_45_0, hcell⟩ := hcell
  extract_lets -merge w_46_0 at hcell
  obtain ⟨w_47_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_51_0, hcell⟩ := hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  extract_lets -merge w_53_0 w_54_0 at hcell
  obtain ⟨w_55_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_57_concat_1, hcell⟩ := hcell
  obtain ⟨w_59_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  extract_lets -merge w_65_0 at hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  extract_lets -merge w_69_0 w_70_0 at hcell
  obtain ⟨w_71_0, hcell⟩ := hcell
  obtain ⟨w_72_0, hcell⟩ := hcell
  obtain ⟨w_73_0, hcell⟩ := hcell
  obtain ⟨w_73_concat_1, hcell⟩ := hcell
  obtain ⟨w_74_0, hcell⟩ := hcell
  obtain ⟨w_76_0, hcell⟩ := hcell
  obtain ⟨w_78_0, hcell⟩ := hcell
  obtain ⟨w_79_0, hcell⟩ := hcell
  obtain ⟨w_80_0, hcell⟩ := hcell
  extract_lets -merge w_81_0 at hcell
  obtain ⟨w_82_0, hcell⟩ := hcell
  obtain ⟨w_83_0, hcell⟩ := hcell
  obtain ⟨w_84_0, hcell⟩ := hcell
  obtain ⟨w_86_0, hcell⟩ := hcell
  obtain ⟨w_87_0, hcell⟩ := hcell
  obtain ⟨w_88_0, hcell⟩ := hcell
  extract_lets -merge w_89_0 at hcell
  obtain ⟨w_90_0, hcell⟩ := hcell
  obtain ⟨w_91_0, hcell⟩ := hcell
  obtain ⟨w_92_0, hcell⟩ := hcell
  obtain ⟨w_93_0, hcell⟩ := hcell
  obtain ⟨w_94_0, hcell⟩ := hcell
  obtain ⟨w_95_0, hcell⟩ := hcell
  obtain ⟨w_96_0, hcell⟩ := hcell
  obtain ⟨w_97_0, hcell⟩ := hcell
  obtain ⟨w_98_0, hcell⟩ := hcell
  obtain ⟨w_99_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_721.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_721.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_721.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_721.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_721.constraints_4, Stage_online_eval_0_0.parallel_generatedRoot_721.constraints_5] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member `i / 5`, the member of `a`.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h11_0 h20_0 h25_0 h29_0 h33_0 h51_0
  have ho : w_39_0 = ((i.val / 5 : Nat) : Int) := fdiv_slot i.val
  rw [ho] at h40_0 h42_0 h49_0 h59_0 h61_0 h68_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  obtain ⟨_, hw51⟩ := familyGet_eq h51_0
  obtain ⟨hko, hw40⟩ := familyGet_eq h40_0
  obtain ⟨_, hw42⟩ := familyGet_eq h42_0
  obtain ⟨_, hw49⟩ := familyGet_eq h49_0
  obtain ⟨_, hw59⟩ := familyGet_eq h59_0
  obtain ⟨_, hw61⟩ := familyGet_eq h61_0
  obtain ⟨_, hw68⟩ := familyGet_eq h68_0
  subst hw2 hw11 hw20 hw25 hw29 hw33 hw51 hw40 hw42 hw49 hw59 hw61 hw68
  have ha' := ha ⟨i.val % 5, hk⟩
  have hs' := hs ⟨i.val / 5, hko⟩
  -- The alignment of the member from `Q6` to `Q5`; the row and the key share one decomposition.
  have hg13 := gadgetMatrixRuns_unique h13_0 h4_0
  subst hg13
  have hg14 := centeredRebaseRuns_unique h14_0 h5_0
  subst hg14
  have hH := gadgetDecomposeRuns_unique h16_0 h7_0
  subst hH
  have hal := align_down (q := Q6) (q' := Q5) (p := p5) (m := 12) (m' := 10) hN q6_eq (by norm_num)
    (by norm_num) (by norm_num) Backend.layout4 layoutQ6 rfl hG h5_0 hsm ha'
    (by
      funext r c
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp (config := { zetaDelta := true }) [Matrix.mul_apply, matrixPolynomial_single, map_ofNat])
    h16_0 (Y := ![_, _]) (K := ![![_, _], ![_, _]])
    (fun j ↦ by fin_cases j <;> rfl) (fun k j ↦ by fin_cases k <;> fin_cases j <;> rfl)
    (c' := ![w_9_0, w_27_0]) (A' := ![![w_18_0, w_31_0], ![w_22_0, w_35_0]])
    (fun j ↦ by fin_cases j; exacts [h9_0, h27_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h18_0, h31_0, h22_0, h35_0])
  -- The public multiplication at `Q5`.
  have hpayb := modulusReduce_poly (q := Q6) (p := Q5) (by norm_num) (by norm_num) hN
    (s := α ⟨i.val % 5, hk⟩)
    (by rw [← hpay]; exact h52_0)
  have hpub := mul_public (q := Q5) (m := 10) (m' := 8) (bound := 16384) hN (by norm_num)
    Backend.layout3 layoutQ5 rfl h4_0 (hα _) hal hs' h18_0_1 h23_0 h31_0_1 h36_0 h37_0
    (Xs := ![w_43_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h43_0, h62_0])
    (D' := ![w_44_0, w_63_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h63_0])
    (D := ![w_45_0, w_64_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    hpayb (Y := ![w_47_0, w_66_0]) (A' := ![![w_55_0, w_71_0], ![w_56_0, w_72_0]])
    (fun j ↦ by fin_cases j; exacts [h47_0, h66_0]) (fun j ↦ by fin_cases j; exacts [h55_0, h71_0])
    (fun j ↦ by fin_cases j; exacts [h56_0, h72_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ ⟨i.val / 5, hko⟩) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₅`.
  have htop : rhsTop model key Q0 6 smudge3Rhs1 i = ![w_76_0, w_84_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h76_0).symm
    · exact (hashSample_eq h84_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q5) (q' := Q0) (p := p4) (m := 8) (m' := 6)
    (bound := 16384) hN q5_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout2 layoutQ0 rfl
    hg' hg hsm hpub' h55_0_1 h57_0 h71_0_1 h73_0 h74_0 (top := ![w_76_0, w_84_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_78_0, w_86_0]) (fun j ↦ by fin_cases j; exacts [h78_0, h86_0])
    (D' := ![w_79_0, w_87_0]) (fun j ↦ by fin_cases j; exacts [h79_0, h87_0])
    (D := ![w_80_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h80_0, h88_0])
    hrhs' heB (hαμ _ _) (X := ![w_96_0, w_98_0]) (M := ![![w_82_0, w_90_0], ![w_92_0, w_94_0]])
    (fun j ↦ by fin_cases j; exacts [h96_0, h98_0]) (fun j ↦ by fin_cases j; exacts [h82_0, h90_0])
    (fun j ↦ by fin_cases j; exacts [h92_0, h94_0])
    (c' := ![w_97_0, w_99_0]) (A' := ![![w_83_0, w_91_0], ![w_93_0, w_95_0]])
    (fun j ↦ by fin_cases j; exacts [h97_0, h99_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h83_0, h91_0, h93_0, h95_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring
def smudge4Rhs0 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49]
def smudge4Rhs1 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `0` of the smudge4 PRF: an encoding of `p₅ (α_(i mod 5) μ ν_i)` at `Q5`. -/
theorem smudge4_cell0 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {As : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10}
    {cs : Fin 2 → ExactMatrix Q6 N 1 10} {bot0 bot1 : Fin 800 → ExactMatrix Q5 N 1 8} {i : Fin 800}
    {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_771 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As 0 0, As 1 0, cs 0, apay, As 0 1, As 1 1, cs 1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp μ ν : ErrorPoly N} {α : Fin 5 → ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (α k))
    {Bs : Nat} (hs : Rel sm As (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * μ * unitPair sp j) • g)
      cs Bs)
    {eB : Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q5 N sp • rhsTop model key Q5 8 smudge4Rhs0 i j + ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q5 N 1 8 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ * μ * ν) *
        unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((10 + 10) * N * ((12 + 12) * N * Ba * 16384 + N * 1 * Bs) * 16384 + (N + 2) * (p5 / 2)) / p5) +
        N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_771 at hcell
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_6_0, hcell⟩ := hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  obtain ⟨w_7_concat_1, hcell⟩ := hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  obtain ⟨w_14_concat_1, hcell⟩ := hcell
  obtain ⟨w_15_0, hcell⟩ := hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_19_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hcell
  obtain ⟨w_28_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  obtain ⟨w_30_0, hcell⟩ := hcell
  obtain ⟨w_30_concat_1, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  obtain ⟨w_34_0, hcell⟩ := hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  extract_lets -merge w_36_0 at hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  extract_lets -merge w_40_0 at hcell
  obtain ⟨w_41_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_43_concat_1, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_46_0, hcell⟩ := hcell
  obtain ⟨w_48_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_50_0, hcell⟩ := hcell
  extract_lets -merge w_51_0 at hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  obtain ⟨w_53_0, hcell⟩ := hcell
  obtain ⟨w_54_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_58_0, hcell⟩ := hcell
  extract_lets -merge w_59_0 at hcell
  obtain ⟨w_60_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  obtain ⟨w_65_0, hcell⟩ := hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_67_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  obtain ⟨w_69_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_771.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_771.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_771.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_771.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_771.constraints_4] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member of `a`, and its public key.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h4_0 h6_0 h9_0 h11_0 h13_0 h25_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw9⟩ := familyGet_eq h9_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  subst hw2 hw4 hw6 hw9 hw11 hw13 hw25
  have ha' := ha ⟨i.val % 5, hk⟩
  -- The public multiplication at `Q6`.
  have hpub := mul_public (q := Q6) (m := 12) (m' := 10) (bound := 16384) hN (by norm_num)
    Backend.layout4 layoutQ6 rfl hG (hα _) ha' hs h4_0_1 h7_0 h11_0_1 h14_0 h15_0
    (Xs := ![w_18_0, w_33_0]) (fun j ↦ by fin_cases j; exacts [h18_0, h33_0])
    (D' := ![w_19_0, w_34_0]) (fun j ↦ by fin_cases j; exacts [h19_0, h34_0])
    (D := ![w_20_0, w_35_0]) (fun j ↦ by fin_cases j; exacts [h20_0, h35_0])
    (hpay _) (Y := ![w_22_0, w_37_0]) (A' := ![![w_28_0, w_41_0], ![w_29_0, w_42_0]])
    (fun j ↦ by fin_cases j; exacts [h22_0, h37_0]) (fun j ↦ by fin_cases j; exacts [h28_0, h41_0])
    (fun j ↦ by fin_cases j; exacts [h29_0, h42_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₆`.
  have htop : rhsTop model key Q5 8 smudge4Rhs0 i = ![w_46_0, w_54_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h46_0).symm
    · exact (hashSample_eq h54_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q6) (q' := Q5) (p := p5) (m := 10) (m' := 8)
    (bound := 16384) hN q6_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout3 layoutQ5 rfl
    hg' hg hsm hpub' h28_0_1 h30_0 h41_0_1 h43_0 h44_0 (top := ![w_46_0, w_54_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_48_0, w_56_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h56_0])
    (D' := ![w_49_0, w_57_0]) (fun j ↦ by fin_cases j; exacts [h49_0, h57_0])
    (D := ![w_50_0, w_58_0]) (fun j ↦ by fin_cases j; exacts [h50_0, h58_0])
    hrhs' heB (hαμ _) (X := ![w_66_0, w_68_0]) (M := ![![w_52_0, w_60_0], ![w_62_0, w_64_0]])
    (fun j ↦ by fin_cases j; exacts [h66_0, h68_0]) (fun j ↦ by fin_cases j; exacts [h52_0, h60_0])
    (fun j ↦ by fin_cases j; exacts [h62_0, h64_0])
    (c' := ![w_67_0, w_69_0]) (A' := ![![w_53_0, w_61_0], ![w_63_0, w_65_0]])
    (fun j ↦ by fin_cases j; exacts [h67_0, h69_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h53_0, h61_0, h63_0, h65_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring


set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `1` of the smudge4 PRF: an encoding of `p₄ (α_(i mod 5) μ_(i/5) ν_i)` at `Q0`. -/
theorem smudge4_cell1 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {As00 As10 As01 As11 cs0 cs1 : Fin 160 → ExactMatrix Q5 N 1 8}
    {bot0 bot1 : Fin 800 → ExactMatrix Q0 N 1 6} {i : Fin 800} {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_775 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp ν : ErrorPoly N} {μ : Fin 160 → ErrorPoly N} {α : Fin 5 → ErrorPoly N}
    (hsm : polyNorm sm ≤ 1) (hαμ : ∀ k o, polyNorm (α k * μ o) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q0 N 1 6} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (α k))
    {Bs : Nat} (hs : ∀ o : Fin 160, Rel sm ![![As00 o, As01 o], ![As10 o, As11 o]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * μ o * unitPair sp j) • g) ![cs0 o, cs1 o] Bs)
    {eB : Fin 2 → ErrorMatrix N 1 6} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q0 N sp • rhsTop model key Q0 6 smudge4Rhs1 i j + ![bot0 i, bot1 i] j =
      reducePoly Q0 N (((q3 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q0 N 1 6 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ *
        μ ⟨i.val / 5, by omega⟩ * ν) * unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((8 + 8) * N * ((10 + 10) * N * ((12 * N * Ba * (32768 / 2) + (N + 2) * (p5 / 2)) / p5) *
        16384 + N * 1 * Bs) * 16384 + (N + 2) * (p4 / 2)) / p4) + N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_775 at hcell
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_5_0, hcell⟩ := hcell
  extract_lets -merge w_6_0 at hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  extract_lets -merge w_8_0 at hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  extract_lets -merge w_15_0 at hcell
  obtain ⟨w_16_0, hcell⟩ := hcell
  extract_lets -merge w_17_0 at hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_23_0, hcell⟩ := hcell
  obtain ⟨w_23_concat_1, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 at hcell
  obtain ⟨w_27_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  extract_lets -merge w_30_0 at hcell
  obtain ⟨w_31_0, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  extract_lets -merge w_34_0 at hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  obtain ⟨w_36_0, hcell⟩ := hcell
  obtain ⟨w_36_concat_1, hcell⟩ := hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  obtain ⟨w_40_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_45_0, hcell⟩ := hcell
  extract_lets -merge w_46_0 at hcell
  obtain ⟨w_47_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_51_0, hcell⟩ := hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  extract_lets -merge w_53_0 w_54_0 at hcell
  obtain ⟨w_55_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_57_concat_1, hcell⟩ := hcell
  obtain ⟨w_59_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  extract_lets -merge w_65_0 at hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  extract_lets -merge w_69_0 w_70_0 at hcell
  obtain ⟨w_71_0, hcell⟩ := hcell
  obtain ⟨w_72_0, hcell⟩ := hcell
  obtain ⟨w_73_0, hcell⟩ := hcell
  obtain ⟨w_73_concat_1, hcell⟩ := hcell
  obtain ⟨w_74_0, hcell⟩ := hcell
  obtain ⟨w_76_0, hcell⟩ := hcell
  obtain ⟨w_78_0, hcell⟩ := hcell
  obtain ⟨w_79_0, hcell⟩ := hcell
  obtain ⟨w_80_0, hcell⟩ := hcell
  extract_lets -merge w_81_0 at hcell
  obtain ⟨w_82_0, hcell⟩ := hcell
  obtain ⟨w_83_0, hcell⟩ := hcell
  obtain ⟨w_84_0, hcell⟩ := hcell
  obtain ⟨w_86_0, hcell⟩ := hcell
  obtain ⟨w_87_0, hcell⟩ := hcell
  obtain ⟨w_88_0, hcell⟩ := hcell
  extract_lets -merge w_89_0 at hcell
  obtain ⟨w_90_0, hcell⟩ := hcell
  obtain ⟨w_91_0, hcell⟩ := hcell
  obtain ⟨w_92_0, hcell⟩ := hcell
  obtain ⟨w_93_0, hcell⟩ := hcell
  obtain ⟨w_94_0, hcell⟩ := hcell
  obtain ⟨w_95_0, hcell⟩ := hcell
  obtain ⟨w_96_0, hcell⟩ := hcell
  obtain ⟨w_97_0, hcell⟩ := hcell
  obtain ⟨w_98_0, hcell⟩ := hcell
  obtain ⟨w_99_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_775.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_775.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_775.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_775.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_775.constraints_4, Stage_online_eval_0_0.parallel_generatedRoot_775.constraints_5] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member `i / 5`, the member of `a`.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h11_0 h20_0 h25_0 h29_0 h33_0 h51_0
  have ho : w_39_0 = ((i.val / 5 : Nat) : Int) := fdiv_slot i.val
  rw [ho] at h40_0 h42_0 h49_0 h59_0 h61_0 h68_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  obtain ⟨_, hw51⟩ := familyGet_eq h51_0
  obtain ⟨hko, hw40⟩ := familyGet_eq h40_0
  obtain ⟨_, hw42⟩ := familyGet_eq h42_0
  obtain ⟨_, hw49⟩ := familyGet_eq h49_0
  obtain ⟨_, hw59⟩ := familyGet_eq h59_0
  obtain ⟨_, hw61⟩ := familyGet_eq h61_0
  obtain ⟨_, hw68⟩ := familyGet_eq h68_0
  subst hw2 hw11 hw20 hw25 hw29 hw33 hw51 hw40 hw42 hw49 hw59 hw61 hw68
  have ha' := ha ⟨i.val % 5, hk⟩
  have hs' := hs ⟨i.val / 5, hko⟩
  -- The alignment of the member from `Q6` to `Q5`; the row and the key share one decomposition.
  have hg13 := gadgetMatrixRuns_unique h13_0 h4_0
  subst hg13
  have hg14 := centeredRebaseRuns_unique h14_0 h5_0
  subst hg14
  have hH := gadgetDecomposeRuns_unique h16_0 h7_0
  subst hH
  have hal := align_down (q := Q6) (q' := Q5) (p := p5) (m := 12) (m' := 10) hN q6_eq (by norm_num)
    (by norm_num) (by norm_num) Backend.layout4 layoutQ6 rfl hG h5_0 hsm ha'
    (by
      funext r c
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp (config := { zetaDelta := true }) [Matrix.mul_apply, matrixPolynomial_single, map_ofNat])
    h16_0 (Y := ![_, _]) (K := ![![_, _], ![_, _]])
    (fun j ↦ by fin_cases j <;> rfl) (fun k j ↦ by fin_cases k <;> fin_cases j <;> rfl)
    (c' := ![w_9_0, w_27_0]) (A' := ![![w_18_0, w_31_0], ![w_22_0, w_35_0]])
    (fun j ↦ by fin_cases j; exacts [h9_0, h27_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h18_0, h31_0, h22_0, h35_0])
  -- The public multiplication at `Q5`.
  have hpayb := modulusReduce_poly (q := Q6) (p := Q5) (by norm_num) (by norm_num) hN
    (s := α ⟨i.val % 5, hk⟩)
    (by rw [← hpay]; exact h52_0)
  have hpub := mul_public (q := Q5) (m := 10) (m' := 8) (bound := 16384) hN (by norm_num)
    Backend.layout3 layoutQ5 rfl h4_0 (hα _) hal hs' h18_0_1 h23_0 h31_0_1 h36_0 h37_0
    (Xs := ![w_43_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h43_0, h62_0])
    (D' := ![w_44_0, w_63_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h63_0])
    (D := ![w_45_0, w_64_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    hpayb (Y := ![w_47_0, w_66_0]) (A' := ![![w_55_0, w_71_0], ![w_56_0, w_72_0]])
    (fun j ↦ by fin_cases j; exacts [h47_0, h66_0]) (fun j ↦ by fin_cases j; exacts [h55_0, h71_0])
    (fun j ↦ by fin_cases j; exacts [h56_0, h72_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ ⟨i.val / 5, hko⟩) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₅`.
  have htop : rhsTop model key Q0 6 smudge4Rhs1 i = ![w_76_0, w_84_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h76_0).symm
    · exact (hashSample_eq h84_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q5) (q' := Q0) (p := p4) (m := 8) (m' := 6)
    (bound := 16384) hN q5_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout2 layoutQ0 rfl
    hg' hg hsm hpub' h55_0_1 h57_0 h71_0_1 h73_0 h74_0 (top := ![w_76_0, w_84_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_78_0, w_86_0]) (fun j ↦ by fin_cases j; exacts [h78_0, h86_0])
    (D' := ![w_79_0, w_87_0]) (fun j ↦ by fin_cases j; exacts [h79_0, h87_0])
    (D := ![w_80_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h80_0, h88_0])
    hrhs' heB (hαμ _ _) (X := ![w_96_0, w_98_0]) (M := ![![w_82_0, w_90_0], ![w_92_0, w_94_0]])
    (fun j ↦ by fin_cases j; exacts [h96_0, h98_0]) (fun j ↦ by fin_cases j; exacts [h82_0, h90_0])
    (fun j ↦ by fin_cases j; exacts [h92_0, h94_0])
    (c' := ![w_97_0, w_99_0]) (A' := ![![w_83_0, w_91_0], ![w_93_0, w_95_0]])
    (fun j ↦ by fin_cases j; exacts [h97_0, h99_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h83_0, h91_0, h93_0, h95_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring
def smudge5Rhs0 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49]
def smudge5Rhs1 : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `0` of the smudge5 PRF: an encoding of `p₅ (α_(i mod 5) μ ν_i)` at `Q5`. -/
theorem smudge5_cell0 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {As : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10}
    {cs : Fin 2 → ExactMatrix Q6 N 1 10} {bot0 bot1 : Fin 800 → ExactMatrix Q5 N 1 8} {i : Fin 800}
    {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_825 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As 0 0, As 1 0, cs 0, apay, As 0 1, As 1 1, cs 1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp μ ν : ErrorPoly N} {α : Fin 5 → ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (α k))
    {Bs : Nat} (hs : Rel sm As (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * μ * unitPair sp j) • g)
      cs Bs)
    {eB : Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q5 N sp • rhsTop model key Q5 8 smudge5Rhs0 i j + ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q5 N 1 8 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ * μ * ν) *
        unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((10 + 10) * N * ((12 + 12) * N * Ba * 16384 + N * 1 * Bs) * 16384 + (N + 2) * (p5 / 2)) / p5) +
        N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_825 at hcell
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_6_0, hcell⟩ := hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  obtain ⟨w_7_concat_1, hcell⟩ := hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  obtain ⟨w_14_concat_1, hcell⟩ := hcell
  obtain ⟨w_15_0, hcell⟩ := hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_19_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hcell
  obtain ⟨w_28_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  obtain ⟨w_30_0, hcell⟩ := hcell
  obtain ⟨w_30_concat_1, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  obtain ⟨w_34_0, hcell⟩ := hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  extract_lets -merge w_36_0 at hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  extract_lets -merge w_40_0 at hcell
  obtain ⟨w_41_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_43_concat_1, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_46_0, hcell⟩ := hcell
  obtain ⟨w_48_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_50_0, hcell⟩ := hcell
  extract_lets -merge w_51_0 at hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  obtain ⟨w_53_0, hcell⟩ := hcell
  obtain ⟨w_54_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_58_0, hcell⟩ := hcell
  extract_lets -merge w_59_0 at hcell
  obtain ⟨w_60_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  obtain ⟨w_65_0, hcell⟩ := hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_67_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  obtain ⟨w_69_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_825.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_825.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_825.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_825.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_825.constraints_4] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member of `a`, and its public key.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h4_0 h6_0 h9_0 h11_0 h13_0 h25_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw9⟩ := familyGet_eq h9_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  subst hw2 hw4 hw6 hw9 hw11 hw13 hw25
  have ha' := ha ⟨i.val % 5, hk⟩
  -- The public multiplication at `Q6`.
  have hpub := mul_public (q := Q6) (m := 12) (m' := 10) (bound := 16384) hN (by norm_num)
    Backend.layout4 layoutQ6 rfl hG (hα _) ha' hs h4_0_1 h7_0 h11_0_1 h14_0 h15_0
    (Xs := ![w_18_0, w_33_0]) (fun j ↦ by fin_cases j; exacts [h18_0, h33_0])
    (D' := ![w_19_0, w_34_0]) (fun j ↦ by fin_cases j; exacts [h19_0, h34_0])
    (D := ![w_20_0, w_35_0]) (fun j ↦ by fin_cases j; exacts [h20_0, h35_0])
    (hpay _) (Y := ![w_22_0, w_37_0]) (A' := ![![w_28_0, w_41_0], ![w_29_0, w_42_0]])
    (fun j ↦ by fin_cases j; exacts [h22_0, h37_0]) (fun j ↦ by fin_cases j; exacts [h28_0, h41_0])
    (fun j ↦ by fin_cases j; exacts [h29_0, h42_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₆`.
  have htop : rhsTop model key Q5 8 smudge5Rhs0 i = ![w_46_0, w_54_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h46_0).symm
    · exact (hashSample_eq h54_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q6) (q' := Q5) (p := p5) (m := 10) (m' := 8)
    (bound := 16384) hN q6_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout3 layoutQ5 rfl
    hg' hg hsm hpub' h28_0_1 h30_0 h41_0_1 h43_0 h44_0 (top := ![w_46_0, w_54_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_48_0, w_56_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h56_0])
    (D' := ![w_49_0, w_57_0]) (fun j ↦ by fin_cases j; exacts [h49_0, h57_0])
    (D := ![w_50_0, w_58_0]) (fun j ↦ by fin_cases j; exacts [h50_0, h58_0])
    hrhs' heB (hαμ _) (X := ![w_66_0, w_68_0]) (M := ![![w_52_0, w_60_0], ![w_62_0, w_64_0]])
    (fun j ↦ by fin_cases j; exacts [h66_0, h68_0]) (fun j ↦ by fin_cases j; exacts [h52_0, h60_0])
    (fun j ↦ by fin_cases j; exacts [h62_0, h64_0])
    (c' := ![w_67_0, w_69_0]) (A' := ![![w_53_0, w_61_0], ![w_63_0, w_65_0]])
    (fun j ↦ by fin_cases j; exacts [h67_0, h69_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h53_0, h61_0, h63_0, h65_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring


set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Cell `i` of bucket `1` of the smudge5 PRF: an encoding of `p₄ (α_(i mod 5) μ_(i/5) ν_i)` at `Q0`. -/
theorem smudge5_cell1 {model : HashModel} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {As00 As10 As01 As11 cs0 cs1 : Fin 160 → ExactMatrix Q5 N 1 8}
    {bot0 bot1 : Fin 800 → ExactMatrix Q0 N 1 6} {i : Fin 800} {params} {outputs}
    (hcell : Stage_online_eval_0_0.parallel_generatedRoot_829 Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key,
        bot0 i, bot1 i, ()) outputs)
    {sm sp ν : ErrorPoly N} {μ : Fin 160 → ErrorPoly N} {α : Fin 5 → ErrorPoly N}
    (hsm : polyNorm sm ≤ 1) (hαμ : ∀ k o, polyNorm (α k * μ o) ≤ 1) (hα : ∀ k, polyNorm (α k) ≤ 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g)
    {h : ExactMatrix Q0 N 1 6} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (α k))
    {Bs : Nat} (hs : ∀ o : Fin 160, Rel sm ![![As00 o, As01 o], ![As10 o, As11 o]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * μ o * unitPair sp j) • g) ![cs0 o, cs1 o] Bs)
    {eB : Fin 2 → ErrorMatrix N 1 6} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q0 N sp • rhsTop model key Q0 6 smudge5Rhs1 i j + ![bot0 i, bot1 i] j =
      reducePoly Q0 N (((q3 : Nat) : Int) * ν * unitPair sp j) • h + reduceMatrix Q0 N 1 6 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) * (α ⟨i.val % 5, Nat.mod_lt _ (by norm_num)⟩ *
        μ ⟨i.val / 5, by omega⟩ * ν) * unitPair sp j) • h)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((((8 + 8) * N * ((10 + 10) * N * ((12 * N * Ba * (32768 / 2) + (N + 2) * (p5 / 2)) / p5) *
        16384 + N * 1 * Bs) * 16384 + (N + 2) * (p4 / 2)) / p4) + N * 1 * BB) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_829 at hcell
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  obtain ⟨w_5_0, hcell⟩ := hcell
  extract_lets -merge w_6_0 at hcell
  obtain ⟨w_7_0, hcell⟩ := hcell
  extract_lets -merge w_8_0 at hcell
  obtain ⟨w_9_0, hcell⟩ := hcell
  obtain ⟨w_11_0, hcell⟩ := hcell
  obtain ⟨w_13_0, hcell⟩ := hcell
  obtain ⟨w_14_0, hcell⟩ := hcell
  extract_lets -merge w_15_0 at hcell
  obtain ⟨w_16_0, hcell⟩ := hcell
  extract_lets -merge w_17_0 at hcell
  obtain ⟨w_18_0, hcell⟩ := hcell
  obtain ⟨w_20_0, hcell⟩ := hcell
  extract_lets -merge w_21_0 at hcell
  obtain ⟨w_22_0, hcell⟩ := hcell
  obtain ⟨w_23_0, hcell⟩ := hcell
  obtain ⟨w_23_concat_1, hcell⟩ := hcell
  obtain ⟨w_25_0, hcell⟩ := hcell
  extract_lets -merge w_26_0 at hcell
  obtain ⟨w_27_0, hcell⟩ := hcell
  obtain ⟨w_29_0, hcell⟩ := hcell
  extract_lets -merge w_30_0 at hcell
  obtain ⟨w_31_0, hcell⟩ := hcell
  obtain ⟨w_33_0, hcell⟩ := hcell
  extract_lets -merge w_34_0 at hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  obtain ⟨w_36_0, hcell⟩ := hcell
  obtain ⟨w_36_concat_1, hcell⟩ := hcell
  obtain ⟨w_37_0, hcell⟩ := hcell
  obtain ⟨w_40_0, hcell⟩ := hcell
  obtain ⟨w_42_0, hcell⟩ := hcell
  obtain ⟨w_43_0, hcell⟩ := hcell
  obtain ⟨w_44_0, hcell⟩ := hcell
  obtain ⟨w_45_0, hcell⟩ := hcell
  extract_lets -merge w_46_0 at hcell
  obtain ⟨w_47_0, hcell⟩ := hcell
  obtain ⟨w_49_0, hcell⟩ := hcell
  obtain ⟨w_51_0, hcell⟩ := hcell
  obtain ⟨w_52_0, hcell⟩ := hcell
  extract_lets -merge w_53_0 w_54_0 at hcell
  obtain ⟨w_55_0, hcell⟩ := hcell
  obtain ⟨w_56_0, hcell⟩ := hcell
  obtain ⟨w_57_0, hcell⟩ := hcell
  obtain ⟨w_57_concat_1, hcell⟩ := hcell
  obtain ⟨w_59_0, hcell⟩ := hcell
  obtain ⟨w_61_0, hcell⟩ := hcell
  obtain ⟨w_62_0, hcell⟩ := hcell
  obtain ⟨w_63_0, hcell⟩ := hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  extract_lets -merge w_65_0 at hcell
  obtain ⟨w_66_0, hcell⟩ := hcell
  obtain ⟨w_68_0, hcell⟩ := hcell
  extract_lets -merge w_69_0 w_70_0 at hcell
  obtain ⟨w_71_0, hcell⟩ := hcell
  obtain ⟨w_72_0, hcell⟩ := hcell
  obtain ⟨w_73_0, hcell⟩ := hcell
  obtain ⟨w_73_concat_1, hcell⟩ := hcell
  obtain ⟨w_74_0, hcell⟩ := hcell
  obtain ⟨w_76_0, hcell⟩ := hcell
  obtain ⟨w_78_0, hcell⟩ := hcell
  obtain ⟨w_79_0, hcell⟩ := hcell
  obtain ⟨w_80_0, hcell⟩ := hcell
  extract_lets -merge w_81_0 at hcell
  obtain ⟨w_82_0, hcell⟩ := hcell
  obtain ⟨w_83_0, hcell⟩ := hcell
  obtain ⟨w_84_0, hcell⟩ := hcell
  obtain ⟨w_86_0, hcell⟩ := hcell
  obtain ⟨w_87_0, hcell⟩ := hcell
  obtain ⟨w_88_0, hcell⟩ := hcell
  extract_lets -merge w_89_0 at hcell
  obtain ⟨w_90_0, hcell⟩ := hcell
  obtain ⟨w_91_0, hcell⟩ := hcell
  obtain ⟨w_92_0, hcell⟩ := hcell
  obtain ⟨w_93_0, hcell⟩ := hcell
  obtain ⟨w_94_0, hcell⟩ := hcell
  obtain ⟨w_95_0, hcell⟩ := hcell
  obtain ⟨w_96_0, hcell⟩ := hcell
  obtain ⟨w_97_0, hcell⟩ := hcell
  obtain ⟨w_98_0, hcell⟩ := hcell
  obtain ⟨w_99_0, hcell⟩ := hcell
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_829.constraints_0, Stage_online_eval_0_0.parallel_generatedRoot_829.constraints_1, Stage_online_eval_0_0.parallel_generatedRoot_829.constraints_2, Stage_online_eval_0_0.parallel_generatedRoot_829.constraints_3, Stage_online_eval_0_0.parallel_generatedRoot_829.constraints_4, Stage_online_eval_0_0.parallel_generatedRoot_829.constraints_5] at hcell
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hcell
  -- The coordinate `i mod 5`, the member `i / 5`, the member of `a`.
  have hc : w_1_0 = ((i.val % 5 : Nat) : Int) := select_slot i.val
  rw [hc] at h2_0 h11_0 h20_0 h25_0 h29_0 h33_0 h51_0
  have ho : w_39_0 = ((i.val / 5 : Nat) : Int) := fdiv_slot i.val
  rw [ho] at h40_0 h42_0 h49_0 h59_0 h61_0 h68_0
  obtain ⟨hk, hw2⟩ := familyGet_eq h2_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw25⟩ := familyGet_eq h25_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  obtain ⟨_, hw51⟩ := familyGet_eq h51_0
  obtain ⟨hko, hw40⟩ := familyGet_eq h40_0
  obtain ⟨_, hw42⟩ := familyGet_eq h42_0
  obtain ⟨_, hw49⟩ := familyGet_eq h49_0
  obtain ⟨_, hw59⟩ := familyGet_eq h59_0
  obtain ⟨_, hw61⟩ := familyGet_eq h61_0
  obtain ⟨_, hw68⟩ := familyGet_eq h68_0
  subst hw2 hw11 hw20 hw25 hw29 hw33 hw51 hw40 hw42 hw49 hw59 hw61 hw68
  have ha' := ha ⟨i.val % 5, hk⟩
  have hs' := hs ⟨i.val / 5, hko⟩
  -- The alignment of the member from `Q6` to `Q5`; the row and the key share one decomposition.
  have hg13 := gadgetMatrixRuns_unique h13_0 h4_0
  subst hg13
  have hg14 := centeredRebaseRuns_unique h14_0 h5_0
  subst hg14
  have hH := gadgetDecomposeRuns_unique h16_0 h7_0
  subst hH
  have hal := align_down (q := Q6) (q' := Q5) (p := p5) (m := 12) (m' := 10) hN q6_eq (by norm_num)
    (by norm_num) (by norm_num) Backend.layout4 layoutQ6 rfl hG h5_0 hsm ha'
    (by
      funext r c
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp (config := { zetaDelta := true }) [Matrix.mul_apply, matrixPolynomial_single, map_ofNat])
    h16_0 (Y := ![_, _]) (K := ![![_, _], ![_, _]])
    (fun j ↦ by fin_cases j <;> rfl) (fun k j ↦ by fin_cases k <;> fin_cases j <;> rfl)
    (c' := ![w_9_0, w_27_0]) (A' := ![![w_18_0, w_31_0], ![w_22_0, w_35_0]])
    (fun j ↦ by fin_cases j; exacts [h9_0, h27_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h18_0, h31_0, h22_0, h35_0])
  -- The public multiplication at `Q5`.
  have hpayb := modulusReduce_poly (q := Q6) (p := Q5) (by norm_num) (by norm_num) hN
    (s := α ⟨i.val % 5, hk⟩)
    (by rw [← hpay]; exact h52_0)
  have hpub := mul_public (q := Q5) (m := 10) (m' := 8) (bound := 16384) hN (by norm_num)
    Backend.layout3 layoutQ5 rfl h4_0 (hα _) hal hs' h18_0_1 h23_0 h31_0_1 h36_0 h37_0
    (Xs := ![w_43_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h43_0, h62_0])
    (D' := ![w_44_0, w_63_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h63_0])
    (D := ![w_45_0, w_64_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    hpayb (Y := ![w_47_0, w_66_0]) (A' := ![![w_55_0, w_71_0], ![w_56_0, w_72_0]])
    (fun j ↦ by fin_cases j; exacts [h47_0, h66_0]) (fun j ↦ by fin_cases j; exacts [h55_0, h71_0])
    (fun j ↦ by fin_cases j; exacts [h56_0, h72_0])
    (c' := ![_, _]) (fun j ↦ by fin_cases j <;> rfl)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
      (α ⟨i.val % 5, hk⟩ * μ ⟨i.val / 5, hko⟩) * unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, dropping `p₅`.
  have htop : rhsTop model key Q0 6 smudge5Rhs1 i = ![w_76_0, w_84_0] := by
    funext j
    fin_cases j
    · exact (hashSample_eq h76_0).symm
    · exact (hashSample_eq h84_0).symm
  have hrhs' := hrhs
  rw [htop] at hrhs'
  have hpriv := mul_private_down (q := Q5) (q' := Q0) (p := p4) (m := 8) (m' := 6)
    (bound := 16384) hN q5_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout2 layoutQ0 rfl
    hg' hg hsm hpub' h55_0_1 h57_0 h71_0_1 h73_0 h74_0 (top := ![w_76_0, w_84_0])
    (bot := ![bot0 i, bot1 i])
    (Xb := ![w_78_0, w_86_0]) (fun j ↦ by fin_cases j; exacts [h78_0, h86_0])
    (D' := ![w_79_0, w_87_0]) (fun j ↦ by fin_cases j; exacts [h79_0, h87_0])
    (D := ![w_80_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h80_0, h88_0])
    hrhs' heB (hαμ _ _) (X := ![w_96_0, w_98_0]) (M := ![![w_82_0, w_90_0], ![w_92_0, w_94_0]])
    (fun j ↦ by fin_cases j; exacts [h96_0, h98_0]) (fun j ↦ by fin_cases j; exacts [h82_0, h90_0])
    (fun j ↦ by fin_cases j; exacts [h92_0, h94_0])
    (c' := ![w_97_0, w_99_0]) (A' := ![![w_83_0, w_91_0], ![w_93_0, w_95_0]])
    (fun j ↦ by fin_cases j; exacts [h97_0, h99_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h83_0, h91_0, h93_0, h95_0])
  subst hout
  convert hpriv using 2 with j
  rw [smul_smul, ← map_mul]
  congr 2
  ring

end RefreshTopProof.Part_0_0
