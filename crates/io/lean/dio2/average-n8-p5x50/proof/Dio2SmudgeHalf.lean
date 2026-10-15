import Dio2SmudgeValues
import Dio2Online0Assembly0
import W_Stage_online_1

/-!
# Coefficientwise half-offset smudge bounds

The corrected decoder subtracts the modular half at every coefficient. An odd readout
digit in `[-7, 7]` therefore contributes exactly `(digit - 1) / 2`, in `[-4, 3]`.
The witnesses below have coefficient bound four, not a dimension-dependent bound and not
the scalar-offset bound in `Dio2SmudgeValues`.

`smudge_half_values` works with modular LUT-value identities, without bounding arbitrary
integer lifts. `online0_part_smudge_half` covers all 36 parts and all five primes;
`online0_smudge_hTH` has the prime-zero `Online0Inputs.hTH` shape.
`online0_refresh_smudge_half` transports the same witnesses to each part's actual offset
input. `online1_smudge_half` supplies the `hTH` and `hM` of `online1_output`.
The readout adapters accept exactly the residue identity returned by
`online0_smudge_digits_of_walsh`; their oddness and bounds are derived from the readout table.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen RefreshTopProof.GenFinal
open scoped BigOperators

/-- One odd digit of magnitude at most seven has the exact centered interval `[-4, 3]`. -/
theorem smudge_half_digit_range (digit : Int) (hodd : digit % 2 = 1)
    (hbound : |digit| ≤ 7) :
    -4 ≤ (digit - 1) / 2 ∧ (digit - 1) / 2 ≤ 3 := by
  have := abs_le.mp hbound
  have := smudge_one_digit_two_mul (P := 8) (by decide) digit hodd
  omega

/-- The permitted readout `-7` attains magnitude four, so the uniform bound is sharp. -/
theorem smudge_half_bound_sharp : (((-7 : Int) - 1) / 2).natAbs = 4 := by decide

/-- The all-coefficients runtime constant is the reduction of the corresponding integer sum. -/
theorem allCoefficients_reduce {q n rows columns : Nat} (half : Int) :
    (allCoefficients half : ExactMatrix q n rows columns) =
      fun _ _ ↦ reducePoly q n (intPoly (fun _ ↦ half)) := by
  funext row column
  exact polynomialOfCoefficients_eq_reduce _

/-- Subtracting a half at every coefficient gives packed witnesses bounded by `P / 2`. -/
theorem smudge_half_values {q P columns : Nat} (hP : P % 2 = 0) {half : Int}
    (hhalf : ((2 * half : Int) : ZMod q) = 1)
    (ν : Fin (8 * columns) → ErrorPoly 8) (digit : Fin (8 * columns) → Int)
    (hν : ∀ member, reducePoly q 8 (ν member) =
      reducePoly q 8 (digit member : ErrorPoly 8))
    (hodd : ∀ member, digit member % 2 = 1)
    (hbound : ∀ member, |digit member| ≤ (P : Int) - 1)
    {width : Nat} (select : Fin width → Fin columns) :
    ∃ noise : ErrorMatrix 8 1 width, CoeffBound noise (P / 2) ∧
      Matrix.of (fun (_ : Fin 1) column ↦
        reducePoly q 8 (placedReadout ν (select column)) * (half : ExactPoly q 8)) -
          allCoefficients half = reduceMatrix q 8 1 width noise := by
  refine ⟨fun _ column ↦ centeredSmudge (fun coefficient ↦
    digit (readoutMember (select column) coefficient)), ?_, ?_⟩
  · intro _ column coefficient
    exact centeredSmudge_bound (by decide) hP _ (fun _ ↦ hodd _) (fun _ ↦ hbound _)
      coefficient
  · rw [allCoefficients_reduce]
    funext row column
    change reducePoly q 8 (placedReadout ν (select column)) * half -
      reducePoly q 8 (intPoly (fun _ ↦ half)) = _
    rw [placedReadout_reduce ν digit hν]
    exact smudge_coefficientwise_offset hhalf _ (fun _ ↦ hodd _)

/-- Concatenation preserves an all-coefficients constant, independently of the column counts. -/
theorem concatColumns_allCoefficients {q n rows leftColumns rightColumns : Nat}
    {left : ExactMatrix q n rows leftColumns} {right : ExactMatrix q n rows rightColumns}
    {output : ExactMatrix q n rows (leftColumns + rightColumns)} {half : Int}
    (hconcat : concatColumns left right output)
    (hleft : left = allCoefficients half) (hright : right = allCoefficients half) :
    output = allCoefficients half := by
  funext row column
  rw [hconcat]
  split_ifs <;> simp only [hleft, hright, allCoefficients]

/-- The corrected step-zero offset repeats the coefficientwise half in all 25 columns. -/
theorem online0_smudge_half_offset {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_0.Params}
    (ctx : W_Stage_online_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_online_0.generatedRoot.Facts ctx outputs) :
    W_Stage_online_0.generatedRoot.w_1276_0 ctx = allCoefficients smudgeHalf0 := by
  have hbase : W_Stage_online_0.generatedRoot.w_1275_0 ctx =
      allCoefficients smudgeHalf0 := rfl
  have h1 := concatColumns_allCoefficients F.c48.h1275_0 hbase hbase
  have h2 := concatColumns_allCoefficients F.c48.h1275_0_1 h1 hbase
  have h3 := concatColumns_allCoefficients F.c48.h1275_0_2 h2 hbase
  have h4 := concatColumns_allCoefficients F.c48.h1275_0_3 h3 hbase
  have h5 := concatColumns_allCoefficients F.c48.h1275_0_4 h4 hbase
  have h6 := concatColumns_allCoefficients F.c48.h1275_0_5 h5 hbase
  have h7 := concatColumns_allCoefficients F.c48.h1275_0_6 h6 hbase
  have h8 := concatColumns_allCoefficients F.c48.h1275_0_7 h7 hbase
  have h9 := concatColumns_allCoefficients F.c48.h1275_0_8 h8 hbase
  have h10 := concatColumns_allCoefficients F.c48.h1275_0_9 h9 hbase
  have h11 := concatColumns_allCoefficients F.c48.h1275_0_10 h10 hbase
  have h12 := concatColumns_allCoefficients F.c48.h1275_0_11 h11 hbase
  have h13 := concatColumns_allCoefficients F.c48.h1275_0_12 h12 hbase
  have h14 := concatColumns_allCoefficients F.c48.h1275_0_13 h13 hbase
  have h15 := concatColumns_allCoefficients F.c48.h1275_0_14 h14 hbase
  have h16 := concatColumns_allCoefficients F.c48.h1275_0_15 h15 hbase
  have h17 := concatColumns_allCoefficients F.c48.h1275_0_16 h16 hbase
  have h18 := concatColumns_allCoefficients F.c48.h1275_0_17 h17 hbase
  have h19 := concatColumns_allCoefficients F.c48.h1275_0_18 h18 hbase
  have h20 := concatColumns_allCoefficients F.c48.h1275_0_19 h19 hbase
  have h21 := concatColumns_allCoefficients F.c48.h1275_0_20 h20 hbase
  have h22 := concatColumns_allCoefficients F.c48.h1275_0_21 h21 hbase
  have h23 := concatColumns_allCoefficients F.c48.h1275_0_22 h22 hbase
  exact concatColumns_allCoefficients F.c48.h1276_0 h23 hbase

/-- A unit-scale assembly cell is the packed readout, with the same member placement. -/
theorem cellVal_one_placedReadout {columns : Nat}
    (ν : Fin (8 * columns) → ErrorPoly 8) (column : Fin columns) :
    cellVal (le_refl (8 * columns)) 1 ν column = placedReadout ν column := by
  unfold cellVal placedReadout
  apply Finset.sum_congr rfl
  intro coefficient _
  have hindex : (readoutMember column coefficient).val % 8 = coefficient.val := by
    simp [readoutMember, Nat.add_mod, Nat.mod_eq_of_lt coefficient.isLt]
  simp only [one_mul, hindex]
  rfl

/-- Every step-zero part and prime has a smudge witness with coefficient bound four. -/
theorem online0_part_smudge_half {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_0.Params}
    (ctx : W_Stage_online_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_online_0.generatedRoot.Facts ctx outputs)
    (ν : Fin 36000 → ErrorPoly 8) (digit : Fin 36000 → Int)
    (hν : ∀ member, reducePoly W_Stage_online_0.modulus_0 8 (ν member) =
      reducePoly W_Stage_online_0.modulus_0 8 (digit member : ErrorPoly 8))
    (hodd : ∀ member, digit member % 2 = 1)
    (hbound : ∀ member, |digit member| ≤ 7)
    {Δ : ErrorPoly 8} (hΔ : Δ = 1) (part : Fin 36) (prime : Fin 5) :
    ∃ noise : ErrorMatrix 8 1 25, CoeffBound noise 4 ∧
      Matrix.of (fun (_ : Fin 1) column ↦ reducePoly W_Stage_online_0.modulus_0 8
        (cellVal (C := 4500) (by decide) Δ ν (smudgePartColumn part prime column)) *
          (smudgeHalf0 : ExactPoly W_Stage_online_0.modulus_0 8)) -
        W_Stage_online_0.generatedRoot.w_1276_0 ctx =
          reduceMatrix W_Stage_online_0.modulus_0 8 1 25 noise := by
  subst Δ
  simpa only [cellVal_one_placedReadout, online0_smudge_half_offset ctx F] using
    smudge_half_values (q := W_Stage_online_0.modulus_0) (P := 8) (columns := 4500)
      (by decide) smudgeHalf0_correct.2 ν digit hν hodd hbound (smudgePartColumn part prime)

/-- The prime-zero family has exactly the `Online0Inputs.hTH` shape, with `BM = 4`. -/
theorem online0_smudge_hTH {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_0.Params}
    (ctx : W_Stage_online_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_online_0.generatedRoot.Facts ctx outputs)
    (ν : Fin 36000 → ErrorPoly 8) (digit : Fin 36000 → Int)
    (hν : ∀ member, reducePoly W_Stage_online_0.modulus_0 8 (ν member) =
      reducePoly W_Stage_online_0.modulus_0 8 (digit member : ErrorPoly 8))
    (hodd : ∀ member, digit member % 2 = 1)
    (hbound : ∀ member, |digit member| ≤ 7)
    {Δ : ErrorPoly 8} (hΔ : Δ = 1)
    {half : ExactPoly W_Stage_online_0.modulus_0 8} (hhalf : half = smudgeHalf0) :
    ∀ part : Fin 36, ∃ noise : ErrorMatrix 8 1 25, CoeffBound noise 4 ∧
      Matrix.of (fun (_ : Fin 1) column ↦ reducePoly W_Stage_online_0.modulus_0 8
        (cellVal (C := 4500) (by decide) Δ ν
          ⟨125 * part.val + column.val, by have := part.isLt; have := column.isLt; omega⟩) *
            half) - W_Stage_online_0.generatedRoot.w_1276_0 ctx =
              reduceMatrix W_Stage_online_0.modulus_0 8 1 25 noise := by
  subst half
  intro part
  simpa only [smudgePartColumn, Fin.val_zero, mul_zero, add_zero] using
    online0_part_smudge_half ctx F ν digit hν hodd hbound hΔ part 0

/-- Actual refresh-part inputs use the same bound-four witnesses for all five primes. -/
theorem online0_refresh_smudge_half {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_0.Params}
    (ctx : W_Stage_online_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_online_0.generatedRoot.Facts ctx outputs)
    (ν : Fin 36000 → ErrorPoly 8) (digit : Fin 36000 → Int)
    (hν : ∀ member, reducePoly W_Stage_online_0.modulus_0 8 (ν member) =
      reducePoly W_Stage_online_0.modulus_0 8 (digit member : ErrorPoly 8))
    (hodd : ∀ member, digit member % 2 = 1)
    (hbound : ∀ member, |digit member| ≤ 7)
    {Δ : ErrorPoly 8} (hΔ : Δ = 1) (part : Fin 36) (prime : Fin 5)
    (partCtx : W_Stage_online_0.parallel_generatedRoot_1281.Ctx backend hashModel params part)
    (hctx : partCtx.inputs = Online0Inputs.PartInputs0 ctx part) :
    ∃ noise : ErrorMatrix 8 1 25, CoeffBound noise 4 ∧
      Matrix.of (fun (_ : Fin 1) column ↦ reducePoly W_Stage_online_0.modulus_0 8
        (cellVal (C := 4500) (by decide) Δ ν (smudgePartColumn part prime column)) *
          (smudgeHalf0 : ExactPoly W_Stage_online_0.modulus_0 8)) -
        W_Stage_online_0.parallel_generatedRoot_1281.w_59_0 partCtx =
          reduceMatrix W_Stage_online_0.modulus_0 8 1 25 noise := by
  have hoff : W_Stage_online_0.parallel_generatedRoot_1281.w_59_0 partCtx =
      W_Stage_online_0.generatedRoot.w_1276_0 ctx := by
    unfold W_Stage_online_0.parallel_generatedRoot_1281.w_59_0
    rw [hctx]
  rw [hoff]
  exact online0_part_smudge_half ctx F ν digit hν hodd hbound hΔ part prime

/-- Signed readout residues alone discharge every step-zero smudge part and prime. -/
theorem online0_smudge_half_of_readout {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_0.Params}
    (ctx : W_Stage_online_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_online_0.generatedRoot.Facts ctx outputs)
    (ν : Fin 36000 → ErrorPoly 8) (exponent : Fin 36000 → Nat)
    (hvalue : ∀ member, reducePoly W_Stage_online_0.modulus_0 8 (ν member) =
      reducePoly W_Stage_online_0.modulus_0 8
        (smudgeReadoutDigit 8 8 (exponent member) : ErrorPoly 8))
    {Δ : ErrorPoly 8} (hΔ : Δ = 1) (part : Fin 36) (prime : Fin 5) :
    ∃ noise : ErrorMatrix 8 1 25, CoeffBound noise 4 ∧
      Matrix.of (fun (_ : Fin 1) column ↦ reducePoly W_Stage_online_0.modulus_0 8
        (cellVal (C := 4500) (by decide) Δ ν (smudgePartColumn part prime column)) *
          (smudgeHalf0 : ExactPoly W_Stage_online_0.modulus_0 8)) -
        W_Stage_online_0.generatedRoot.w_1276_0 ctx =
          reduceMatrix W_Stage_online_0.modulus_0 8 1 25 noise := by
  exact online0_part_smudge_half ctx F ν (fun member ↦ smudgeReadoutDigit 8 8 (exponent member))
    hvalue (fun member ↦ (smudgeReadoutDigit_range (by decide) (by decide) (by decide)
      (exponent member)).1)
    (fun member ↦ (smudgeReadoutDigit_range (by decide) (by decide) (by decide)
      (exponent member)).2) hΔ part prime

/-- The readout table and monomial-input Walsh identities discharge all step-zero smudge bounds. -/
theorem online0_smudge_half_of_walsh {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_0.Params}
    (ctx : W_Stage_online_0.generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : W_Stage_online_0.generatedRoot.Facts ctx outputs) {inverse : Nat}
    (hinverse : (inverse : ExactPoly W_Stage_online_0.modulus_0 8) * 8 = 1)
    (index : Fin 8 → Nat) (hodd : ∀ branch, Odd (index branch))
    (perm : Fin 8 ≃ Fin 8)
    (hindex : ∀ branch, index branch % 16 = 1 + 2 * 1 * (perm branch).val)
    {C : Fin 8 → ExactPoly W_Stage_online_0.modulus_0 8}
    (hC : ∀ branch, C branch = lutFourierCoefficient 8 (X W_Stage_online_0.modulus_0 ^ 1)
      (inverse : ExactPoly W_Stage_online_0.modulus_0 8)
      (fun entry : Fin 8 ↦ reducePoly W_Stage_online_0.modulus_0 8
        (MxxIR.selectInt (entry.val : Int) [1, 1, 3, 3, 5, 5, 7, 7] : ErrorPoly 8))
      (index branch))
    (μ ν : Fin 36000 → ErrorPoly 8) (exponent : Fin 36000 → Nat)
    (hμ : ∀ member, reducePoly W_Stage_online_0.modulus_0 8 (μ member) =
      reducePoly W_Stage_online_0.modulus_0 8 (Xe (n := 8) (exponent member)))
    (hwalsh : ∀ member, ∑ branch, reducePoly W_Stage_online_0.modulus_0 8
      (substitution (index branch) (hodd branch) (μ member)) * C branch =
        reducePoly W_Stage_online_0.modulus_0 8 (ν member))
    {Δ : ErrorPoly 8} (hΔ : Δ = 1) (part : Fin 36) (prime : Fin 5) :
    ∃ noise : ErrorMatrix 8 1 25, CoeffBound noise 4 ∧
      Matrix.of (fun (_ : Fin 1) column ↦ reducePoly W_Stage_online_0.modulus_0 8
        (cellVal (C := 4500) (by decide) Δ ν (smudgePartColumn part prime column)) *
          (smudgeHalf0 : ExactPoly W_Stage_online_0.modulus_0 8)) -
        W_Stage_online_0.generatedRoot.w_1276_0 ctx =
          reduceMatrix W_Stage_online_0.modulus_0 8 1 25 noise := by
  have hvalues := online0_smudge_digits_of_walsh hinverse index hodd perm hindex hC
    μ ν exponent hμ hwalsh
  exact online0_smudge_half_of_readout ctx F ν exponent (fun member ↦ (hvalues member).1)
    hΔ part prime

/-- The single packed readout is the placed total used by the step-one smudge LUT. -/
theorem placedReadout_single (ν : Fin 8 → ErrorPoly 8) :
    placedReadout (columns := 1) ν 0 =
      ∑ coefficient : Fin 8, Xe (n := 8) coefficient.val * ν coefficient := by
  unfold placedReadout
  apply Finset.sum_congr rfl
  intro coefficient _
  have hmember : readoutMember (0 : Fin 1) coefficient = coefficient := by
    apply Fin.ext
    simp [readoutMember]
  simp only [hmember, Nat.mod_eq_of_lt coefficient.isLt]

/-- The fixed step-one graph subtracts the same modular half at every coefficient. -/
theorem online1_smudge_half_offset {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_1.Params}
    (ctx : W_Stage_online_1.generatedRoot.Ctx backend hashModel params) :
    W_Stage_online_1.generatedRoot.w_186_0 ctx = allCoefficients smudgeHalf0 := rfl

/-- The step-one smudge target discharges `online1_output.hTH` and `.hM` with `BM = 4`. -/
theorem online1_smudge_half {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_1.Params}
    (ctx : W_Stage_online_1.generatedRoot.Ctx backend hashModel params)
    (ν : Fin 8 → ErrorPoly 8) (digit : Fin 8 → Int)
    (hν : ∀ member, reducePoly W_Stage_online_1.modulus_0 8 (ν member) =
      reducePoly W_Stage_online_1.modulus_0 8 (digit member : ErrorPoly 8))
    (hodd : ∀ member, digit member % 2 = 1)
    (hbound : ∀ member, |digit member| ≤ 7)
    {Δ : ErrorPoly 8} (hΔ : Δ = 1)
    {Tsm : ExactMatrix W_Stage_online_1.modulus_0 8 1 1}
    (hTsm : Tsm = fun _ _ ↦ reducePoly W_Stage_online_1.modulus_0 8
      (∑ member : Fin 8, Xe (n := 8) member.val * (Δ * ν member)) *
        (smudgeHalf0 : ExactPoly W_Stage_online_1.modulus_0 8)) :
    ∃ noise : ErrorMatrix 8 1 1, CoeffBound noise 4 ∧
      Tsm - W_Stage_online_1.generatedRoot.w_186_0 ctx =
        reduceMatrix W_Stage_online_1.modulus_0 8 1 1 noise := by
  subst Δ
  rw [hTsm, online1_smudge_half_offset]
  simpa only [placedReadout_single, one_mul] using
    smudge_half_values (q := W_Stage_online_1.modulus_0) (P := 8) (columns := 1)
      (width := 1) (by decide) smudgeHalf0_correct.2 ν digit hν hodd hbound (fun _ ↦ 0)

/-- The step-one Walsh readout residues give a bound-four output-smudge witness. -/
theorem online1_smudge_half_of_readout {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_1.Params}
    (ctx : W_Stage_online_1.generatedRoot.Ctx backend hashModel params)
    (ν : Fin 8 → ErrorPoly 8) (exponent : Fin 8 → Nat)
    (hvalue : ∀ member, reducePoly W_Stage_online_1.modulus_0 8 (ν member) =
      reducePoly W_Stage_online_1.modulus_0 8
        (smudgeReadoutDigit 8 8 (exponent member) : ErrorPoly 8))
    {Δ : ErrorPoly 8} (hΔ : Δ = 1)
    {Tsm : ExactMatrix W_Stage_online_1.modulus_0 8 1 1}
    (hTsm : Tsm = fun _ _ ↦ reducePoly W_Stage_online_1.modulus_0 8
      (∑ member : Fin 8, Xe (n := 8) member.val * (Δ * ν member)) *
        (smudgeHalf0 : ExactPoly W_Stage_online_1.modulus_0 8)) :
    ∃ noise : ErrorMatrix 8 1 1, CoeffBound noise 4 ∧
      Tsm - W_Stage_online_1.generatedRoot.w_186_0 ctx =
        reduceMatrix W_Stage_online_1.modulus_0 8 1 1 noise := by
  exact online1_smudge_half ctx ν (fun member ↦ smudgeReadoutDigit 8 8 (exponent member))
    hvalue (fun member ↦ (smudgeReadoutDigit_range (by decide) (by decide) (by decide)
      (exponent member)).1)
    (fun member ↦ (smudgeReadoutDigit_range (by decide) (by decide) (by decide)
      (exponent member)).2) hΔ hTsm

/-- The step-one readout's Walsh identities supply both output-smudge hypotheses at bound four. -/
theorem online1_smudge_half_of_walsh {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_1.Params}
    (ctx : W_Stage_online_1.generatedRoot.Ctx backend hashModel params) {inverse : Nat}
    (hinverse : (inverse : ExactPoly W_Stage_online_1.modulus_0 8) * 8 = 1)
    (index : Fin 8 → Nat) (hodd : ∀ branch, Odd (index branch))
    (perm : Fin 8 ≃ Fin 8)
    (hindex : ∀ branch, index branch % 16 = 1 + 2 * 1 * (perm branch).val)
    {C : Fin 8 → ExactPoly W_Stage_online_1.modulus_0 8}
    (hC : ∀ branch, C branch = lutFourierCoefficient 8 (X W_Stage_online_1.modulus_0 ^ 1)
      (inverse : ExactPoly W_Stage_online_1.modulus_0 8)
      (fun entry : Fin 8 ↦ reducePoly W_Stage_online_1.modulus_0 8
        (MxxIR.selectInt (entry.val : Int) [1, 1, 3, 3, 5, 5, 7, 7] : ErrorPoly 8))
      (index branch))
    (μ ν : Fin 8 → ErrorPoly 8) (exponent : Fin 8 → Nat)
    (hμ : ∀ member, reducePoly W_Stage_online_1.modulus_0 8 (μ member) =
      reducePoly W_Stage_online_1.modulus_0 8 (Xe (n := 8) (exponent member)))
    (hwalsh : ∀ member, ∑ branch, reducePoly W_Stage_online_1.modulus_0 8
      (substitution (index branch) (hodd branch) (μ member)) * C branch =
        reducePoly W_Stage_online_1.modulus_0 8 (ν member))
    {Δ : ErrorPoly 8} (hΔ : Δ = 1)
    {Tsm : ExactMatrix W_Stage_online_1.modulus_0 8 1 1}
    (hTsm : Tsm = fun _ _ ↦ reducePoly W_Stage_online_1.modulus_0 8
      (∑ member : Fin 8, Xe (n := 8) member.val * (Δ * ν member)) *
        (smudgeHalf0 : ExactPoly W_Stage_online_1.modulus_0 8)) :
    ∃ noise : ErrorMatrix 8 1 1, CoeffBound noise 4 ∧
      Tsm - W_Stage_online_1.generatedRoot.w_186_0 ctx =
        reduceMatrix W_Stage_online_1.modulus_0 8 1 1 noise := by
  have hvalues := online0_smudge_digits_of_walsh hinverse index hodd perm hindex hC
    μ ν exponent hμ hwalsh
  exact online1_smudge_half_of_readout ctx ν exponent (fun member ↦ (hvalues member).1)
    hΔ hTsm

end Dio2Proof
