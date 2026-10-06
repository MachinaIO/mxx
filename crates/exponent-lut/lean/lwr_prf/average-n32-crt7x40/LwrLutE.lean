import LwrBlockE4
import LwrEncryptE
import LwrLut

/-!
# The readout LUT with explicit errors and keys (prototype)

Branch `i` of the readout LUT routes the state through `σ_i`, switches it by the digits `D_i` of
the switching ciphertext, moves the switched key down (`Asw_i`), and acts on the helper by the
digits `E_i` of `Asw_i`. Its error is the helper error after `E_i`, the routed state error after
`D_i`, and the residues of the routed key, the first weighted by the substituted mask.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {tape : SampleTape} {path loopPath : List Nat} {loopPaths : Nat → List Nat}

/-- The digits `D_i` of the switching ciphertext of branch `i`. -/
noncomputable abbrev lutDigitsD (fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q1 N 1 2) (i : Fin 16)
    (j : Fin 2) : ExactMatrix Q2 N (2 + 2) 2 :=
  stackDigits Q2 2 524288 1048576 2 (fam 0 j i) (fam 1 j i)

/-- The routed key of branch `i` after the switching action, before its ModDown. -/
noncomputable abbrev lutKeyM (A : Fin 2 → Fin 2 → ExactMatrix Q2 N 1 2)
    (fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q1 N 1 2) (i : Fin 16) (k j : Fin 2) :
    ExactMatrix Q2 N 1 2 :=
  pairMul (fun j' ↦ substituteMatrix (sigma i).toNat (sigma_odd i) (A k j')) (lutDigitsD fam i j)

/-- The digits `E_i` of the moved-down routed key. -/
noncomputable abbrev lutDigitsE (A : Fin 2 → Fin 2 → ExactMatrix Q2 N 1 2)
    (fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q1 N 1 2) (i : Fin 16) (j : Fin 2) :
    ExactMatrix Q2 N (2 + 2) 2 :=
  stackDigits Q2 2 524288 1048576 2
    (theOutput (blockModSwitchRuns [1099511626049, 1099511625409] 1) (lutKeyM A fam i 0 j) : ExactMatrix Q1 N 1 2)
    (theOutput (blockModSwitchRuns [1099511626049, 1099511625409] 1) (lutKeyM A fam i 1 j))

/-- The helper key of branch `i` after the action by `E_i`. -/
noncomputable abbrev lutKeyK (model : HashModel) (key : ByteArray)
    (A : Fin 2 → Fin 2 → ExactMatrix Q2 N 1 2)
    (fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q1 N 1 2) (i : Fin 16) (k j : Fin 2) :
    ExactMatrix Q2 N 1 2 :=
  pairMul (lutA model key Q2 2 5 (sigma i) k) (lutDigitsE A fam i j)

/-- The error added by branch `i`. -/
noncomputable def lutBranchErr (sm : ErrorPoly N) (ec : Fin 2 → ErrorMatrix N 1 2)
    (eh : Fin 16 → Fin 2 → ErrorMatrix N 1 2) (A : Fin 2 → Fin 2 → ExactMatrix Q2 N 1 2)
    (fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q1 N 1 2) (i : Fin 16) (j : Fin 2) :
    ErrorMatrix N 1 2 :=
  pairMul (eh i) (liftMatrix (lutDigitsE A fam i j)) +
    pairMul (fun j' ↦ substituteMatrix (sigma i).toNat (sigma_odd i) (ec j'))
      (liftMatrix (lutDigitsD fam i j)) +
    (substitution (sigma i).toNat (sigma_odd i) sm • downResidue P2 (lutKeyM A fam i 0 j) +
      downResidue P2 (lutKeyM A fam i 1 j))

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 100000 in
/-- `lutLoop_step` with the error and key of the branch named. -/
theorem lutLoop_stepE {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    {g' : ExactMatrix Q1 N 1 2} {g : ExactMatrix Q2 N 1 2}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 2 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q2 N 1 2} {A : Fin 2 → Fin 2 → ExactMatrix Q2 N 1 2}
    {ec : Fin 2 → ErrorMatrix N 1 2}
    (henc : RelE sm A (fun j ↦ reducePoly Q2 N (((P2 : Nat) : Int) * μ * unitPair sp j) • g) c ec)
    {h0 h1 : Fin 16 → ExactMatrix Q2 N 1 2} {eh : Fin 16 → Fin 2 → ErrorMatrix N 1 2}
    (hhelp : ∀ i : Fin 16, RelE sm (lutA model key Q2 2 5 (sigma i))
      (fun j ↦ reducePoly Q2 N (((P2 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] (eh i))
    {fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q1 N 1 2}
    {i : Nat} (hi : i < 15) {state next : LoopState Q2 Q1 2}
    (hswitch : state.switch = fun k j ↦ fam k j ⟨i, by omega⟩)
    {T : Fin 2 → ExactMatrix Q2 N 1 2} {es : Fin 2 → ErrorMatrix N 1 2}
    (hstate : RelE sm state.key T state.row es)
    (hbody : Stage_online_eval.sequential_generatedRoot_131 Backend.backend model tape loopPath
      GeneratedClaim.stage_4_params i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0, c 1, A 0 1, A 1 1, fam 0 0,
        fam 0 1, fam 1 0, fam 1 1, ()) next) :
    RelE sm next.key (T + branchTarget (q := Q2) (p := P2) μ sp fam ⟨i, by omega⟩) next.row
        (es + lutBranchErr sm ec eh A fam ⟨i, by omega⟩) ∧
      next.switch = (fun k j ↦ fam k j ⟨i + 1, by omega⟩) ∧
      next.key = state.key + lutKeyK model key A fam ⟨i, by omega⟩ := by
  unfold Stage_online_eval.sequential_generatedRoot_131 at hbody
  try extract_lets at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  try extract_lets at hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  try extract_lets at hbody
  have h3_0 := hbody.2.2.1
  have h6_0 := hbody.2.2.2.2.2.1
  have h7_0 := hbody.2.2.2.2.2.2.1
  have h6_0_1 := hbody.2.2.2.2.2.2.2.1
  have h8_0 := hbody.2.2.2.2.2.2.2.2.1
  have h10_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.1
  have h11_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h12_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h11_0_1 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h13_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h14_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h16_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h18_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h20_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h18_0_1 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h21_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h23_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h25_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h27_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h25_0_1 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h28_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h29_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h32_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h33_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h34_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h36_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h37_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h38_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h39_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h40_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h41_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h42_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h44_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h45_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h51_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h52_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h53_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h55_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h56_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h57_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h58_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h59_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h60_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h61_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h63_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h64_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h68_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h71_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h74_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h77_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h82_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h84_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h86_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h88_0 := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hout := hbody.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨_, hw3⟩ := familyGet_eq h3_0
  obtain ⟨_, hw10⟩ := familyGet_eq h10_0
  obtain ⟨_, hw82⟩ := familyGet_eq h82_0
  obtain ⟨_, hw84⟩ := familyGet_eq h84_0
  obtain ⟨_, hw86⟩ := familyGet_eq h86_0
  obtain ⟨_, hw88⟩ := familyGet_eq h88_0
  subst hw3 hw10
  have hhelp' := hhelp ⟨i, by omega⟩
  have hAd : lutA model key Q2 2 5 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  simp only [hAd] at hhelp'
  obtain ⟨d, ε, hd, he, hMe, hKe, hbranch⟩ := lut_branchE (n := N) (q := Q2) (q' := Q1) (p := P2) (m := 2) (m' := 2) (log := 5)
    (bound := 524288) (index := sigma i) hN hpow q2_eq hQ2 hQ1 Backend.layout0 layout1 rfl hg' hg
    henc h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (bsw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![_, _]) (sigma_odd ⟨i, by omega⟩) hhelp' h6_0_1 h8_0 h11_0_1 h13_0 h14_0
    (Y := ![w_44_0, w_63_0]) (X := ![w_45_0, w_64_0])
    (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h44_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  -- The digits and keys as functions of the input key and the switching ciphertexts.
  have h32' : concatRows (fam 0 0 ⟨i, by omega⟩) (fam 1 0 ⟨i, by omega⟩) w_32_0 := by
    have h : concatRows (state.switch 0 0) (state.switch 1 0) w_32_0 := h32_0
    rw [hswitch] at h; exact h
  have h51' : concatRows (fam 0 1 ⟨i, by omega⟩) (fam 1 1 ⟨i, by omega⟩) w_51_0 := by
    have h : concatRows (state.switch 0 1) (state.switch 1 1) w_51_0 := h51_0
    rw [hswitch] at h; exact h
  have hDeq : ∀ j, lutDigitsD fam ⟨i, by omega⟩ j = ![w_34_0, w_53_0] j := by
    intro j
    fin_cases j
    · exact stackDigits_eq h32' h33_0 h34_0
    · exact stackDigits_eq h51' h52_0 h53_0
  have hMeq : ∀ k j, lutKeyM A fam ⟨i, by omega⟩ k j = ![![w_36_0, w_55_0], ![w_38_0, w_57_0]] k j := by
    intro k j
    rw [hMe k j, lutKeyM, hDeq j]
  have hAsw : ∀ k j, (theOutput (blockModSwitchRuns [1099511626049, 1099511625409] 1)
      (lutKeyM A fam ⟨i, by omega⟩ k j) : ExactMatrix Q1 N 1 2) =
      ![![w_37_0, w_56_0], ![w_39_0, w_58_0]] k j := by
    intro k j
    have hr : blockModSwitchRuns [1099511626049, 1099511625409] 1 (lutKeyM A fam ⟨i, by omega⟩ k j)
        (![![w_37_0, w_56_0], ![w_39_0, w_58_0]] k j) := by
      rw [hMeq]
      fin_cases k <;> fin_cases j
      exacts [h37_0, h56_0, h39_0, h58_0]
    exact theOutput_eq (fun _ h ↦ blockModSwitchRuns_unique h hr) hr
  have hEeq : ∀ j, lutDigitsE A fam ⟨i, by omega⟩ j = ![w_42_0, w_61_0] j := by
    intro j
    simp only [lutDigitsE, hAsw]
    fin_cases j
    · exact stackDigits_eq h40_0 h41_0 h42_0
    · exact stackDigits_eq h59_0 h60_0 h61_0
  have hKeq : lutKeyK model key A fam ⟨i, by omega⟩ = ![![w_68_0, w_71_0], ![w_74_0, w_77_0]] := by
    funext k j
    rw [lutKeyK, hKe k j, hAd]
    congr 1
    exact hEeq j
  have hErr : (fun j ↦ pairMul (eh ⟨i, by omega⟩) (ε j) +
      pairMul (fun j' ↦ substituteMatrix (sigma i).toNat (sigma_odd ⟨i, by omega⟩) (ec j')) (d j) +
      (substitution (sigma i).toNat (sigma_odd ⟨i, by omega⟩) sm •
          downResidue P2 (![![w_36_0, w_55_0], ![w_38_0, w_57_0]] 0 j) +
        downResidue P2 (![![w_36_0, w_55_0], ![w_38_0, w_57_0]] 1 j))) =
      lutBranchErr sm ec eh A fam ⟨i, by omega⟩ := by
    funext j
    have hε : liftMatrix (lutDigitsE A fam ⟨i, by omega⟩ j) = ε j := by
      rw [hEeq j, (he j).1, liftMatrix_reduce hQ2 hN (he j).2 (by decide)]
    have hδ : liftMatrix (lutDigitsD fam ⟨i, by omega⟩ j) = d j := by
      rw [hDeq j, (hd j).1, liftMatrix_reduce hQ2 hN (hd j).2 (by decide)]
    rw [lutBranchErr, hε, hδ, hMeq 0 j, hMeq 1 j]
  rw [hErr] at hbranch
  subst hout
  rw [hswitch] at hbranch
  refine ⟨?_, ?_, ?_⟩
  · convert hstate.add hbranch using 1
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
    · funext j
      fin_cases j <;> rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hw82, hw84, hw86, hw88]
  · rw [hKeq]
    funext k j
    fin_cases k <;> fin_cases j <;> rfl

/-- The error of branch `i`, zero past the sixteen branches. -/
noncomputable def lutBranchErrAt (sm : ErrorPoly N) (ec : Fin 2 → ErrorMatrix N 1 2)
    (eh : Fin 16 → Fin 2 → ErrorMatrix N 1 2) (A : Fin 2 → Fin 2 → ExactMatrix Q2 N 1 2)
    (fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q1 N 1 2) (i : Nat) : Fin 2 → ErrorMatrix N 1 2 :=
  if h : i < 16 then lutBranchErr sm ec eh A fam ⟨i, h⟩ else 0

/-- The helper key of branch `i`, zero past the sixteen branches. -/
noncomputable def lutKeyKAt (model : HashModel) (key : ByteArray)
    (A : Fin 2 → Fin 2 → ExactMatrix Q2 N 1 2)
    (fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q1 N 1 2) (i : Nat) :
    Fin 2 → Fin 2 → ExactMatrix Q2 N 1 2 :=
  if h : i < 16 then lutKeyK model key A fam ⟨i, h⟩ else 0

/-- `lutLoop` with the accumulated error and key named. -/
theorem lutLoopE {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    {g' : ExactMatrix Q1 N 1 2} {g : ExactMatrix Q2 N 1 2}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 2 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q2 N 1 2} {A : Fin 2 → Fin 2 → ExactMatrix Q2 N 1 2}
    {ec : Fin 2 → ErrorMatrix N 1 2}
    (henc : RelE sm A (fun j ↦ reducePoly Q2 N (((P2 : Nat) : Int) * μ * unitPair sp j) • g) c ec)
    {h0 h1 : Fin 16 → ExactMatrix Q2 N 1 2} {eh : Fin 16 → Fin 2 → ErrorMatrix N 1 2}
    (hhelp : ∀ i : Fin 16, RelE sm (lutA model key Q2 2 5 (sigma i))
      (fun j ↦ reducePoly Q2 N (((P2 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] (eh i))
    {fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q1 N 1 2} {count : Nat} (hcount : count ≤ 15)
    {final : LoopState Q2 Q1 2}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q2 Q1 2) ↦
      Stage_online_eval.sequential_generatedRoot_131 Backend.backend model tape (loopPaths i)
        GeneratedClaim.stage_4_params i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1,
          current.2.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0,
          c 1, A 0 1, A 1 1, fam 0 0, fam 0 1, fam 1 0, fam 1 1, ()) next) count
      (0, 0, 0, 0, 0, 0, fam 0 0 0, fam 0 1 0, fam 1 0 0, fam 1 1 0, ()) final) :
    RelE sm final.key (∑ i ∈ Finset.range count, lutBranchTarget μ sp fam i) final.row
        (∑ i ∈ Finset.range count, lutBranchErrAt sm ec eh A fam i) ∧
      final.switch = (fun k j ↦ fam k j ⟨count, by omega⟩) ∧
      final.key = ∑ i ∈ Finset.range count, lutKeyKAt model key A fam i := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q2 Q1 2) ↦ ∀ hk : k ≤ 15,
      RelE sm s.key (∑ i ∈ Finset.range k, lutBranchTarget μ sp fam i) s.row
          (∑ i ∈ Finset.range k, lutBranchErrAt sm ec eh A fam i) ∧
        s.switch = (fun k' j ↦ fam k' j ⟨k, by omega⟩) ∧
        s.key = ∑ i ∈ Finset.range k, lutKeyKAt model key A fam i) ?_ ?_ hruns hcount
  · intro _
    have hrow : (![0, 0] : Fin 2 → ExactMatrix Q2 N 1 2) = 0 := by
      funext j
      fin_cases j <;> rfl
    have hkey : (![![0, 0], ![0, 0]] : Fin 2 → Fin 2 → ExactMatrix Q2 N 1 2) = 0 := by
      funext k j
      fin_cases k <;> fin_cases j <;> rfl
    refine ⟨?_, ?_, ?_⟩
    · simp only [Finset.range_zero, Finset.sum_empty, LoopState.key, LoopState.row, hrow]
      exact RelE.zero
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
    · simp only [Finset.range_zero, Finset.sum_empty, LoopState.key, hkey]
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw, hkey⟩ := hinv (by omega)
    obtain ⟨hrel', hsw', hkey'⟩ := lutLoop_stepE hg' hg henc hhelp (by omega) hsw hrel hbody
    refine ⟨?_, hsw', ?_⟩
    · rw [Finset.sum_range_succ, Finset.sum_range_succ, lutBranchTarget, dif_pos (by omega),
        lutBranchErrAt, dif_pos (by omega)]
      exact hrel'
    · rw [hkey', hkey, Finset.sum_range_succ, lutKeyKAt, dif_pos (by omega)]

/-- The digits of the finalization scale `k`. -/
noncomputable def finalDigits : ErrorMatrix N 2 1 :=
  liftMatrix (theOutput (compactCenteredRebaseRuns 524288)
    (theOutput (gadgetDecomposeRuns Backend.backend 1048576 2)
      (matrixPolynomial [outScale] : ExactMatrix Q1 N 1 1) : ExactMatrix Q1 N 2 1) :
    ExactMatrix Q1 N 2 1)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 100000 in
/-- `lutRound` with its errors named: the sixteen branch errors, moved down to `Q1` with the row
and key residues, minus the switching errors, and finalized by the digits of `k`. -/
theorem lutRoundE {model : HashModel} {key : ByteArray}
    {s0 s1 s2 s3 k0 k1 ca0 ca1 cs0 cs1 apay r00 r01 r10 r11 r20 r21 r30 r31 r40 r41}
    {outputs} {w : Stage_online_eval.generatedRoot.Witness}
    (hb : Stage_online_eval.generatedRoot.body Backend.backend model tape path GeneratedClaim.stage_4_params
      (s0, s1, s2, s3, k0, key, k1, ca0, ca1, cs0, apay, cs1, r00, r01, r10, r11, r20, r21, r30, r31,
        r40, r41, ()) outputs w)
    {sm sp : ErrorPoly N} {z : Nat}
    {g' : ExactMatrix Q1 N 1 2} {g : ExactMatrix Q2 N 1 2}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 2 g') (hg : centeredRebaseRuns g' g)
    {ec : Fin 2 → ErrorMatrix N 1 2}
    (hinput : RelE sm ![![w.w_130_0, w.w_130_1], ![w.w_130_2, w.w_130_3]]
      (fun j ↦ reducePoly Q2 N (((P2 : Nat) : Int) * Xe (n := N) (2 * z) * unitPair sp j) • g)
      ![w.w_130_4, w.w_130_5] ec)
    {eh : Fin 16 → Fin 2 → ErrorMatrix N 1 2}
    (hhelp : ∀ i : Fin 16, RelE sm (lutA model key Q2 2 5 (sigma i))
      (fun j ↦ reducePoly Q2 N (((P2 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![k0 i, k1 i] (eh i))
    {eS : Fin 16 → Fin 2 → ErrorMatrix N 1 2}
    (hswitch : ∀ i : Fin 16, ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) • ![s0 i, s1 i] j +
          ![s2 i, s3 i] j =
        (lutCoefficient Q1 inverse1 (sigma i).toNat * reducePoly Q1 N (unitPair sp j)) • g' +
          reduceMatrix Q1 N 1 2 (eS i j)) :
    let A := ![![w.w_130_0, w.w_130_1], ![w.w_130_2, w.w_130_3]]
    let fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q1 N 1 2 := fun k j ↦ ![![s0, s1], ![s2, s3]] k j
    let Ksum := ∑ i : Fin 16, lutKeyK model key A fam i
    let err := ∑ i : Fin 16, lutBranchErr sm (ec) eh A fam i
    (∃ d : ErrorMatrix N 2 1, finalDigits = d ∧ CoeffBound d 524288 ∧
      (∀ r c (i : Fin N), i.val ≠ 0 → (d r c).coeff i = 0) ∧
      g' * reduceMatrix Q1 N 2 1 d = (matrixPolynomial [outScale] : ExactMatrix Q1 N 1 1)) ∧
    ∃ Rsum : Fin 2 → ExactMatrix Q2 N 1 2,
      RelE sm Ksum (fun j ↦ ∑ i : Fin 16, scaleUpMatrix Q2 P2
        (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (n := N) (2 * z))) •
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) • fam 0 j i + fam 1 j i)))
        Rsum err ∧
      ∃ v : Fin 2 → ErrorMatrix N 1 2,
        (∀ j row column, err j row column - downResidue P2 (Rsum j) row column +
            sm * downResidue P2 (Ksum 0 j) row column + downResidue P2 (Ksum 1 j) row column =
          ((P2 : Int) : ErrorPoly N) * v j row column) ∧
        RelE sm ![![matrixMul w.w_165_0 w.w_168_0, matrixMul w.w_184_0 w.w_168_0],
            ![matrixMul w.w_188_0 w.w_168_0, matrixMul w.w_192_0 w.w_168_0]]
          (fun j ↦ reducePoly Q1 N ((outScale : ErrorPoly N) * lutValue z * unitPair sp j) •
            (1 : ExactMatrix Q1 N 1 1))
          ![matrixMul w.w_198_0 w.w_168_0, matrixMul w.w_204_0 w.w_168_0]
          (fun j ↦ (v j - ∑ i : Fin 16,
            substitution (sigma i).toNat (sigma_odd i) (Xe (n := N) (2 * z)) • eS i j) * finalDigits) := by
  intro A fam Ksum err
  unfold Stage_online_eval.generatedRoot.body at hb
  extract_lets at hb
  have h2_0 := hb.1
  have h4_0 := hb.2.1
  have h6_0 := hb.2.2.1
  have h8_0 := hb.2.2.2.1
  have h131_9 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h132_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h134_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h134_0_1 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h135_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h136_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h137_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h138_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h138_0_1 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h139_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h140_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h141_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h142_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h143_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h143_0_1 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h144_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h145_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h146_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h147_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h147_0_1 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h148_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h149_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h150_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h151_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h152_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h153_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h155_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h156_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h157_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h158_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h159_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h160_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h161_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h163_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h165_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h167_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h168_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h170_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h171_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h172_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h174_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h175_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h176_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h177_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h178_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h179_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h180_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h182_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h184_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h186_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h188_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h190_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h192_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h194_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h195_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h198_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h200_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h201_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h204_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  -- Fifteen branches in the generated loop.
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h131_9
  obtain ⟨hloop, hloopSwitch, hloopKey⟩ := lutLoopE (fam := fun k j ↦ ![![s0, s1], ![s2, s3]] k j)
    (final := (w.w_131_0, w.w_131_1, w.w_131_2, w.w_131_3, w.w_131_4, w.w_131_5,
      w.w_131_6, w.w_131_7, w.w_131_8, w.w_131_9, ())) hg' hg hinput hhelp (count := 15)
    (by decide) h131_9
  -- The sixteenth branch in the root.
  obtain ⟨_, hw73⟩ := familyGet_eq (k := 15) h132_0
  obtain ⟨_, hw78⟩ := familyGet_eq (k := 15) h137_0
  rw [hw73] at h134_0_1
  rw [hw78] at h138_0_1
  have hhelp15 := hhelp 15
  have hAd : lutA model key Q2 2 5 (sigma (15 : Fin 16)) =
      ![![w.w_134_0, w.w_138_0], ![w.w_135_0, w.w_139_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h134_0).symm
    · exact (hashSample_eq h138_0).symm
    · exact (hashSample_eq h135_0).symm
    · exact (hashSample_eq h139_0).symm
  simp only [hAd] at hhelp15
  obtain ⟨d, ε, hd, he, hMe, hKe, hbranch⟩ := lut_branchE (n := N) (q := Q2) (q' := Q1) (p := P2)
    (m := 2) (m' := 2) (log := 5) (bound := 524288) (index := sigma (15 : Fin 16)) hN hpow q2_eq
    hQ2 hQ1 Backend.layout0 layout1 rfl hg' hg hinput h142_0 h143_0 h144_0 h146_0 h147_0 h148_0 h143_0_1 h145_0
    h147_0_1 h149_0 h150_0
    (bsw := LoopState.switch (w.w_131_0, w.w_131_1, w.w_131_2, w.w_131_3, w.w_131_4,
      w.w_131_5, w.w_131_6, w.w_131_7, w.w_131_8, w.w_131_9, ()))
    (Xb := ![w.w_151_0, w.w_170_0]) (fun j ↦ by fin_cases j; exacts [h151_0, h170_0])
    (D' := ![w.w_152_0, w.w_171_0]) (fun j ↦ by fin_cases j; exacts [h152_0, h171_0])
    (D := ![w.w_153_0, w.w_172_0]) (fun j ↦ by fin_cases j; exacts [h153_0, h172_0])
    (M := ![![w.w_155_0, w.w_174_0], ![w.w_157_0, w.w_176_0]])
    (fun j ↦ by fin_cases j; exacts [h155_0, h174_0])
    (fun j ↦ by fin_cases j; exacts [h157_0, h176_0])
    (Asw := ![![w.w_156_0, w.w_175_0], ![w.w_158_0, w.w_177_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h156_0, h175_0, h158_0, h177_0])
    (Z := ![w.w_159_0, w.w_178_0]) (fun j ↦ by fin_cases j; exacts [h159_0, h178_0])
    (E' := ![w.w_160_0, w.w_179_0]) (fun j ↦ by fin_cases j; exacts [h160_0, h179_0])
    (E := ![w.w_161_0, w.w_180_0]) (fun j ↦ by fin_cases j; exacts [h161_0, h180_0])
    (h := ![_, _]) (sigma_odd 15) hhelp15 h134_0_1 h136_0 h138_0_1 h140_0 h141_0
    (Y := ![w.w_194_0, w.w_200_0]) (X := ![w.w_195_0, w.w_201_0])
    (K := ![![w.w_163_0, w.w_182_0], ![w.w_186_0, w.w_190_0]])
    (fun j ↦ by fin_cases j; exacts [h194_0, h200_0])
    (fun j ↦ by fin_cases j; exacts [h195_0, h201_0])
    (fun j ↦ by fin_cases j; exacts [h163_0, h182_0])
    (fun j ↦ by fin_cases j; exacts [h186_0, h190_0])
  -- The digits and keys of branch `15` as functions of the input key and the switching ciphertexts.
  have h151' : concatRows (fam 0 0 15) (fam 1 0 15) w.w_151_0 := by
    have h : concatRows (LoopState.switch (w.w_131_0, w.w_131_1, w.w_131_2, w.w_131_3, w.w_131_4,
      w.w_131_5, w.w_131_6, w.w_131_7, w.w_131_8, w.w_131_9, ()) 0 0)
      (LoopState.switch (w.w_131_0, w.w_131_1, w.w_131_2, w.w_131_3, w.w_131_4,
      w.w_131_5, w.w_131_6, w.w_131_7, w.w_131_8, w.w_131_9, ()) 1 0) w.w_151_0 := h151_0
    rw [hloopSwitch] at h; exact h
  have h170' : concatRows (fam 0 1 15) (fam 1 1 15) w.w_170_0 := by
    have h : concatRows (LoopState.switch (w.w_131_0, w.w_131_1, w.w_131_2, w.w_131_3, w.w_131_4,
      w.w_131_5, w.w_131_6, w.w_131_7, w.w_131_8, w.w_131_9, ()) 0 1)
      (LoopState.switch (w.w_131_0, w.w_131_1, w.w_131_2, w.w_131_3, w.w_131_4,
      w.w_131_5, w.w_131_6, w.w_131_7, w.w_131_8, w.w_131_9, ()) 1 1) w.w_170_0 := h170_0
    rw [hloopSwitch] at h; exact h
  have hDeq : ∀ j, lutDigitsD fam 15 j = ![w.w_153_0, w.w_172_0] j := by
    intro j
    fin_cases j
    · exact stackDigits_eq h151' h152_0 h153_0
    · exact stackDigits_eq h170' h171_0 h172_0
  have hMeq : ∀ k j, lutKeyM A fam 15 k j =
      ![![w.w_155_0, w.w_174_0], ![w.w_157_0, w.w_176_0]] k j := by
    intro k j
    rw [hMe k j, lutKeyM, hDeq j]
  have hAsw : ∀ k j, (theOutput (blockModSwitchRuns [1099511626049, 1099511625409] 1) (lutKeyM A fam 15 k j) :
      ExactMatrix Q1 N 1 2) = ![![w.w_156_0, w.w_175_0], ![w.w_158_0, w.w_177_0]] k j := by
    intro k j
    have hr : blockModSwitchRuns [1099511626049, 1099511625409] 1 (lutKeyM A fam 15 k j)
        (![![w.w_156_0, w.w_175_0], ![w.w_158_0, w.w_177_0]] k j) := by
      rw [hMeq]
      fin_cases k <;> fin_cases j
      exacts [h156_0, h175_0, h158_0, h177_0]
    exact theOutput_eq (fun _ h ↦ blockModSwitchRuns_unique h hr) hr
  have hEeq : ∀ j, lutDigitsE A fam 15 j = ![w.w_161_0, w.w_180_0] j := by
    intro j
    simp only [lutDigitsE, hAsw]
    fin_cases j
    · exact stackDigits_eq h159_0 h160_0 h161_0
    · exact stackDigits_eq h178_0 h179_0 h180_0
  have hKeq : lutKeyK model key A fam 15 = ![![w.w_163_0, w.w_182_0], ![w.w_186_0, w.w_190_0]] := by
    funext k j
    rw [lutKeyK, hKe k j, hAd]
    congr 1
    exact hEeq j
  have hErr : (fun j ↦ pairMul (eh 15) (ε j) +
      pairMul (fun j' ↦ substituteMatrix (sigma (15 : Fin 16)).toNat (sigma_odd 15) (ec j')) (d j) +
      (substitution (sigma (15 : Fin 16)).toNat (sigma_odd 15) sm •
          downResidue P2 (![![w.w_155_0, w.w_174_0], ![w.w_157_0, w.w_176_0]] 0 j) +
        downResidue P2 (![![w.w_155_0, w.w_174_0], ![w.w_157_0, w.w_176_0]] 1 j))) =
      lutBranchErr sm ec eh A fam 15 := by
    funext j
    have hε : liftMatrix (lutDigitsE A fam 15 j) = ε j := by
      rw [hEeq j, (he j).1, liftMatrix_reduce hQ2 hN (he j).2 (by decide)]
    have hδ : liftMatrix (lutDigitsD fam 15 j) = d j := by
      rw [hDeq j, (hd j).1, liftMatrix_reduce hQ2 hN (hd j).2 (by decide)]
    rw [lutBranchErr, hε, hδ, hMeq 0 j, hMeq 1 j]
  rw [hErr] at hbranch
  rw [hloopSwitch] at hbranch
  have hsum := hloop.add hbranch
  have hKsum : LoopState.key (w.w_131_0, w.w_131_1, w.w_131_2, w.w_131_3, w.w_131_4, w.w_131_5,
      w.w_131_6, w.w_131_7, w.w_131_8, w.w_131_9, ()) +
      ![![w.w_163_0, w.w_182_0], ![w.w_186_0, w.w_190_0]] = Ksum := by
    rw [hloopKey, ← hKeq]
    simp only [Ksum]
    rw [Fin.sum_univ_castSucc, Finset.sum_range]
    refine congrArg₂ (· + ·) (Finset.sum_congr rfl fun i _ ↦ ?_) rfl
    rw [lutKeyKAt, dif_pos (by omega)]
    rfl
  have herr : (∑ i ∈ Finset.range 15, lutBranchErrAt sm ec eh A fam i) +
      lutBranchErr sm ec eh A fam 15 = err := by
    simp only [err]
    rw [Fin.sum_univ_castSucc, Finset.sum_range]
    refine congrArg₂ (· + ·) (Finset.sum_congr rfl fun i _ ↦ ?_) rfl
    rw [lutBranchErrAt, dif_pos (by omega)]
    rfl
  rw [hKsum, herr] at hsum
  -- The sum of the sixteen branches, moved down to `Q1`.
  have hsum' := hsum.target_eq (t' := fun j ↦ ∑ i : Fin 16, scaleUpMatrix Q2 P2
      (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * z))) •
        (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![s0, s1], ![s2, s3]] 0 j i + ![![s0, s1], ![s2, s3]] 1 j i))) (by
    funext j
    rw [Fin.sum_univ_castSucc, Pi.add_apply, Finset.sum_apply, Finset.sum_range]
    refine congrArg₂ (· + ·) (Finset.sum_congr rfl fun i _ ↦ ?_) ?_
    · rw [lutBranchTarget, dif_pos (by omega)]
      rfl
    · rw [show (Fin.last 15 : Fin 16) = 15 from rfl]
      rfl)
  have hwalsh := lut_walsh_value (q := Q1) inverse1_mul z
  have hK' : ∀ k j, blockModSwitchRuns [1099511626049, 1099511625409] 1 (Ksum k j)
      (![![w.w_165_0, w.w_184_0], ![w.w_188_0, w.w_192_0]] k j) := by
    rw [← hKsum]
    intro k j
    fin_cases k <;> fin_cases j
    exacts [h165_0, h184_0, h188_0, h192_0]
  obtain ⟨v, hv, hvp⟩ := lut_finishE (W := 16) (q := Q2) (q' := Q1) (p := P2) (m' := 2) hN q2_eq
    hQ2 hQ1 (sp := sp) (μ := Xe (2 * z)) (ν := lutValue z)
    (Δ' := 1) (fun i ↦ substitution (sigma i).toNat (sigma_odd i))
    (bsw := fun i k j ↦ ![![s0, s1], ![s2, s3]] k j i) hsum' (g' := g')
    (C := fun i ↦ lutCoefficient Q1 inverse1 (sigma i).toNat) (eB := eS)
    (fun i j ↦ by
      fin_cases j
      exacts [by simpa only [one_mul] using hswitch i 0, by simpa only [one_mul] using hswitch i 1])
    hwalsh
    (K' := ![![w.w_165_0, w.w_184_0], ![w.w_188_0, w.w_192_0]])
    (R' := ![w.w_198_0, w.w_204_0])
    hK' (fun j ↦ by fin_cases j; exacts [h198_0, h204_0])
  -- The finalization `G⁻¹(k)`.
  have hfin' := hv.target_eq (t' := fun j ↦ reducePoly Q1 N (lutValue z * unitPair sp j) • g')
    (funext fun j ↦ by rw [one_mul])
  obtain ⟨dk, hdk, hdkb, hdkc, hout⟩ := hfin'.finalizeCoefficient hQ1 hN Backend.layout0 layout1 rfl hg'
    outScale rfl h167_0 h168_0
  have hfinal : finalDigits = dk := by
    have h1 : (theOutput (gadgetDecomposeRuns Backend.backend 1048576 2)
        (matrixPolynomial [outScale] : ExactMatrix Q1 N 1 1) : ExactMatrix Q1 N 2 1) = w.w_167_0 :=
      theOutput_eq (fun _ h ↦ gadgetDecomposeRuns_unique h h167_0) h167_0
    have h2 : (theOutput (compactCenteredRebaseRuns 524288) w.w_167_0 : ExactMatrix Q1 N 2 1) =
        w.w_168_0 := theOutput_eq (fun _ h ↦ compactCenteredRebaseRuns_unique h h168_0) h168_0
    have hdk' : w.w_168_0 = reduceMatrix Q1 N 2 1 dk := hdk
    rw [finalDigits, h1, h2, hdk', liftMatrix_reduce hQ1 hN hdkb (by decide)]
  have hrecomp : g' * reduceMatrix Q1 N 2 1 dk = (matrixPolynomial [outScale] : ExactMatrix Q1 N 1 1) := by
    have h3 := gadget_mul_decompose hQ1 hN Backend.layout0 layout1 rfl hg' h167_0
    have h4 : w.w_167_0 = w.w_168_0 :=
      centeredRebaseRuns_unique (centeredRebaseRuns_self hQ1 hN _) h168_0.2.2.1
    have hdk' : w.w_168_0 = reduceMatrix Q1 N 2 1 dk := hdk
    rw [← hdk', ← h4]
    exact h3
  refine ⟨⟨dk, hfinal, hdkb, hdkc, hrecomp⟩, _, hsum', v, hvp, ?_⟩
  rw [hfinal]
  convert hout using 1
  · funext a j
    fin_cases a <;> fin_cases j <;> rfl
  · funext j
    fin_cases j <;> rfl

end LwrProof
