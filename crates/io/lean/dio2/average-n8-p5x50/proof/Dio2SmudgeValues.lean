import GenSmudge
import GenFinalFresh
import W_Stage_online_0
import W_Stage_preprocess_0_0

set_option maxHeartbeats 2000000

/-!
# Step-0 smudge and fresh-noise values

The value hypotheses below use the integer polynomial `ν` returned by a LUT's `hwalsh`
identity. Only its reduction is prescribed: a Walsh identity cannot bound an arbitrary
integer lift. `placedReadout` packs eight such values into one column, exactly as
`online0_smudge_row` and `online0_fresh_row` do with `Δ' = 1`.
The family adapters derive their value hypotheses from the online Walsh equations and
the exported readout/sampler tables; the Program lemmas verify those tables' outputs.

One odd smudge digit gives `(δ - 1) / 2`, bounded by `P / 2`, after subtracting a half
**at every coefficient**. This distinction matters for the current graph: `w_1274` is
the constant polynomial `(Q + 1) / 2`, and `w_1275` repeats that constant across columns.
It does not repeat it across coefficients. `smudge_scalar_offset` records the residual
nonconstant half terms; `smudge_scalar_offset_obstruction` proves that the intended small
bound cannot hold for this scalar offset. No graph or implementation correction is assumed.

The CBD results give sharp coefficient bounds, not a sum-of-eight norm estimate. They
establish bounded representatives of LUT values; they do not assert their distribution
or independence, nor that an arbitrary congruent integer lift has the same bound.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen RefreshTopProof.GenFinal
open scoped BigOperators

/-- The signed coefficient readout `±(2 f - 1)` of a smudge PRF member. -/
def smudgeReadoutDigit (P width z : Nat) : Int :=
  (-1) ^ (z / width) * readoutVal (P / 2) width (z % width)

/-- The coefficient readout table produces an odd digit in `[-(P - 1), P - 1]`. -/
theorem smudgeReadoutDigit_range {P width : Nat} (hP : P % 2 = 0) (hP2 : 2 ≤ P)
    (hwidth : 0 < width) (z : Nat) :
    smudgeReadoutDigit P width z % 2 = 1 ∧
      |smudgeReadoutDigit P width z| ≤ (P : Int) - 1 := by
  have hodd := readoutVal_odd (P / 2) width (z % width)
  have hlt := readoutVal_lt (p' := P / 2) (by omega) (Nat.mod_lt z hwidth)
  have hsign : (-1 : Int) ^ (z / width) = 1 ∨ (-1 : Int) ^ (z / width) = -1 :=
    neg_one_pow_eq_or Int _
  rcases hsign with hsign | hsign <;>
    simp only [smudgeReadoutDigit, hsign, one_mul, neg_one_mul] <;>
    constructor
  · omega
  · rw [abs_of_nonneg (by positivity)]
    omega
  · omega
  · rw [abs_neg, abs_of_nonneg (by positivity)]
    omega

/-- One digit specializes `GenSmudge` to the integer half and its explicit bound. -/
theorem smudge_one_digit_bound {P : Nat} (hP : P % 2 = 0) (digit : Int)
    (hodd : digit % 2 = 1) (hbound : |digit| ≤ (P : Int) - 1) :
    ((digit - 1) / 2).natAbs ≤ P / 2 := by
  simpa [smudgeInt] using
    smudge_natAbs_le (d := 0) hP (fun _ ↦ digit) (fun _ ↦ hodd) (fun _ ↦ hbound)

/-- The one-digit target is twice the integer smudge plus one, with weight `P^0 = 1`. -/
theorem smudge_one_digit_two_mul {P : Nat} (hP : P % 2 = 0) (digit : Int)
    (hodd : digit % 2 = 1) : digit = 2 * ((digit - 1) / 2) + 1 := by
  simpa [smudgeInt] using smudge_two_mul (d := 0) hP (fun _ ↦ digit) hodd

/-- Subtracting a modular half from an odd scalar readout gives its integer half. -/
theorem smudge_one_digit_value {q n : Nat} {half : Int}
    (hhalf : ((2 * half : Int) : ZMod q) = 1) (digit : Int)
    (hodd : digit % 2 = 1) :
    reducePoly q n (digit : ErrorPoly n) * (half : ExactPoly q n) - half =
      reducePoly q n (((digit - 1) / 2 : Int) : ErrorPoly n) := by
  calc
    _ = reducePoly q n ((half * digit - half : Int) : ErrorPoly n) := by
      simp only [reducePoly_intCast]
      push_cast
      ring
    _ = _ := half_digit_gen hhalf digit hodd

/-- The member index of coefficient `j` in one packed column. -/
def readoutMember {columns : Nat} (column : Fin columns) (coefficient : Fin 8) :
    Fin (8 * columns) :=
  ⟨8 * column.val + coefficient.val, by
    have := column.isLt
    have := coefficient.isLt
    omega⟩

/-- Eight placed LUT readouts, with the placement used by the step-0 column total. -/
noncomputable def placedReadout {columns : Nat} (ν : Fin (8 * columns) → ErrorPoly 8)
    (column : Fin columns) : ErrorPoly 8 :=
  ∑ coefficient : Fin 8,
    Xe (n := 8) ((readoutMember column coefficient).val % 8) *
      ν (readoutMember column coefficient)

/-- Modular LUT values suffice to identify a packed column's bounded representative. -/
theorem placedReadout_reduce {q columns : Nat} (ν : Fin (8 * columns) → ErrorPoly 8)
    (value : Fin (8 * columns) → Int)
    (hν : ∀ member, reducePoly q 8 (ν member) =
      reducePoly q 8 (value member : ErrorPoly 8)) (column : Fin columns) :
    reducePoly q 8 (placedReadout ν column) =
      reducePoly q 8 (intPoly (fun coefficient ↦ value (readoutMember column coefficient))) := by
  simp only [placedReadout, intPoly, map_sum, map_mul]
  apply Finset.sum_congr rfl
  intro coefficient _
  rw [hν]
  have hindex : (readoutMember column coefficient).val % 8 = coefficient.val := by
    simp [readoutMember, Nat.add_mod, Nat.mod_eq_of_lt coefficient.isLt]
  rw [hindex]
  exact mul_comm _ _

/-- Exact chosen LUT lifts give exactly the packed integer polynomial, not just a residue. -/
theorem placedReadout_eq {columns : Nat} (ν : Fin (8 * columns) → ErrorPoly 8)
    (value : Fin (8 * columns) → Int) (hν : ∀ member, ν member = (value member : ErrorPoly 8))
    (column : Fin columns) :
    placedReadout ν column =
      intPoly (fun coefficient ↦ value (readoutMember column coefficient)) := by
  simp only [placedReadout, intPoly]
  apply Finset.sum_congr rfl
  intro coefficient _
  rw [hν]
  have hindex : (readoutMember column coefficient).val % 8 = coefficient.val := by
    simp [readoutMember, Nat.add_mod, Nat.mod_eq_of_lt coefficient.isLt]
  rw [hindex]
  exact mul_comm _ _

/-- The coefficientwise centered one-digit smudge polynomial. -/
noncomputable def centeredSmudge {n : Nat} (digit : Fin n → Int) : ErrorPoly n :=
  intPoly (fun coefficient ↦ (digit coefficient - 1) / 2)

/-- The centered polynomial has coefficient bound `P / 2`, independently of ring dimension. -/
theorem centeredSmudge_bound {n P : Nat} (hn : 0 < n) (hP : P % 2 = 0)
    (digit : Fin n → Int) (hodd : ∀ coefficient, digit coefficient % 2 = 1)
    (hbound : ∀ coefficient, |digit coefficient| ≤ (P : Int) - 1) :
    ∀ coefficient, ((centeredSmudge digit).coeff coefficient).natAbs ≤ P / 2 := by
  intro coefficient
  rw [centeredSmudge, intPoly_coeff hn]
  exact smudge_one_digit_bound hP _ (hodd _) (hbound _)

/-- Coefficientwise half subtraction reduces to the small integer smudge polynomial. -/
theorem smudge_coefficientwise_offset {q n : Nat} {half : Int}
    (hhalf : ((2 * half : Int) : ZMod q) = 1) (digit : Fin n → Int)
    (hodd : ∀ coefficient, digit coefficient % 2 = 1) :
    reducePoly q n (intPoly digit) * (half : ExactPoly q n) -
        reducePoly q n (intPoly (fun _ ↦ half)) =
      reducePoly q n (centeredSmudge digit) := by
  simp only [intPoly, centeredSmudge, map_sum, map_mul]
  rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro coefficient _
  have hone := smudge_one_digit_value (n := n) hhalf (digit coefficient) (hodd coefficient)
  simp only [reducePoly_intCast] at hone
  calc
    _ = ((digit coefficient : ExactPoly q n) * half - half) *
        reducePoly q n (Xe coefficient.val) := by
      simp only [reducePoly_intCast]
      ring
    _ = _ := by rw [hone, reducePoly_intCast]

/-- Coefficients of a constant polynomial, including the zero coefficient. -/
theorem smudge_constant_coeff {n : Nat} (hn : 0 < n) (value : Int) (coefficient : Fin n) :
    ((value : ErrorPoly n)).coeff coefficient = if coefficient.val = 0 then value else 0 := by
  have hone := coeff_root_pow_lt (n := n) hn (w := 0) hn coefficient
  rw [pow_zero] at hone
  have hvalue := intCast_mul_coeff value (1 : ErrorPoly n) coefficient
  rw [mul_one, hone] at hvalue
  rw [hvalue]
  split_ifs <;> simp

/-- The residual left by subtracting the scalar half instead of coefficientwise halves. -/
noncomputable def scalarHalfResidual (n : Nat) (half : Int) : ErrorPoly n :=
  intPoly (fun coefficient ↦ if coefficient.val = 0 then 0 else half)

/-- The residual is the difference between coefficientwise and scalar offsets. -/
theorem scalarHalfResidual_eq {n : Nat} (hn : 0 < n) (half : Int) :
    scalarHalfResidual n half = intPoly (fun _ ↦ half) - (half : ErrorPoly n) := by
  apply negacyclic_ext hn
  intro coefficient
  rw [scalarHalfResidual, intPoly_coeff hn, Negacyclic.coeff_sub, intPoly_coeff hn,
    smudge_constant_coeff hn]
  split_ifs <;> simp

/-- The actual scalar-offset graph has the centered smudge plus a nonconstant half residual. -/
theorem smudge_scalar_offset {q n : Nat} (hn : 0 < n) {half : Int}
    (hhalf : ((2 * half : Int) : ZMod q) = 1) (digit : Fin n → Int)
    (hodd : ∀ coefficient, digit coefficient % 2 = 1) :
    reducePoly q n (intPoly digit) * (half : ExactPoly q n) - half =
      reducePoly q n (centeredSmudge digit + scalarHalfResidual n half) := by
  rw [map_add, scalarHalfResidual_eq hn, map_sub, reducePoly_intCast,
    ← smudge_coefficientwise_offset hhalf digit hodd]
  ring

/-- The scalar-offset witness is bounded, but includes the half-modulus residual. -/
theorem smudge_scalar_offset_bound {n P : Nat} (hn : 0 < n) (hP : P % 2 = 0)
    (half : Int) (digit : Fin n → Int) (hodd : ∀ coefficient, digit coefficient % 2 = 1)
    (hbound : ∀ coefficient, |digit coefficient| ≤ (P : Int) - 1) :
    ∀ coefficient,
      ((centeredSmudge digit + scalarHalfResidual n half).coeff coefficient).natAbs ≤
        P / 2 + half.natAbs := by
  intro coefficient
  rw [Negacyclic.coeff_add]
  apply (Int.natAbs_add_le _ _).trans
  apply Nat.add_le_add (centeredSmudge_bound hn hP digit hodd hbound coefficient)
  rw [scalarHalfResidual, intPoly_coeff hn]
  split_ifs <;> simp

/-- Quantified packed-column witnesses for `hTH` and `hM` with the graph's scalar offset.
The bound is intentionally not the invalid small bound `P / 2`. -/
theorem online0_smudge_values {q P columns : Nat} (hP : P % 2 = 0) {half : Int}
    (hhalf : ((2 * half : Int) : ZMod q) = 1)
    (ν : Fin (8 * columns) → ErrorPoly 8) (digit : Fin (8 * columns) → Int)
    (hν : ∀ member, reducePoly q 8 (ν member) =
      reducePoly q 8 (digit member : ErrorPoly 8))
    (hodd : ∀ member, digit member % 2 = 1)
    (hbound : ∀ member, |digit member| ≤ (P : Int) - 1)
    {width : Nat} (select : Fin width → Fin columns) :
    ∃ noise : ErrorMatrix 8 1 width, CoeffBound noise (P / 2 + half.natAbs) ∧
      Matrix.of (fun (_ : Fin 1) column ↦
        reducePoly q 8 (placedReadout ν (select column)) * (half : ExactPoly q 8)) -
          Matrix.of (fun (_ : Fin 1) (_ : Fin width) ↦ (half : ExactPoly q 8)) =
            reduceMatrix q 8 1 width noise := by
  refine ⟨fun _ column ↦ centeredSmudge (fun coefficient ↦
    digit (readoutMember (select column) coefficient)) + scalarHalfResidual 8 half, ?_, ?_⟩
  · intro _ column coefficient
    exact smudge_scalar_offset_bound (by decide) hP half _
      (fun _ ↦ hodd _) (fun _ ↦ hbound _) coefficient
  · funext row column
    change reducePoly q 8 (placedReadout ν (select column)) * half - half = _
    rw [placedReadout_reduce ν digit hν]
    exact smudge_scalar_offset (by decide) hhalf _ (fun _ ↦ hodd _)

/-- The CBD(1) bin table has zeros in its first half and ones in its second half. -/
def cbd1Bin (P bin : Nat) : Int := if bin < P / 4 then 0 else 1

/-- Every CBD(1) sampler value is in `[-1, 1]`, including the negacyclic sign. -/
theorem cbd1_value_bound (P width : Nat) (z : Int) :
    (samplerVal P width (cbd1Bin P) z).natAbs ≤ 1 := by
  apply samplerVal_natAbs_le
  intro bin
  simp only [cbd1Bin]
  split_ifs <;> norm_num

/-- Chosen integer readout values pack to sharp CBD coefficient bound `η = 1`. -/
theorem online0_fresh_values {columns : Nat} (ν : Fin (8 * columns) → ErrorPoly 8)
    (value : Fin (8 * columns) → Int) (hν : ∀ member, ν member = (value member : ErrorPoly 8))
    (hbound : ∀ member, (value member).natAbs ≤ 1)
    {width : Nat} (select : Fin width → Fin columns) :
    CoeffBound (fun _ column ↦ placedReadout ν (select column) : ErrorMatrix 8 1 width) 1 := by
  intro row column coefficient
  change ((placedReadout ν (select column)).coeff coefficient).natAbs ≤ 1
  rw [placedReadout_eq ν value hν, intPoly_coeff (by decide)]
  exact hbound _

/-- Modular LUT outputs admit bounded fresh representatives without bounding arbitrary lifts. -/
theorem online0_fresh_value_representatives {q columns : Nat}
    (ν : Fin (8 * columns) → ErrorPoly 8) (value : Fin (8 * columns) → Int)
    (hν : ∀ member, reducePoly q 8 (ν member) =
      reducePoly q 8 (value member : ErrorPoly 8))
    (hbound : ∀ member, (value member).natAbs ≤ 1)
    {width : Nat} (select : Fin width → Fin columns) :
    ∃ noise : ErrorMatrix 8 1 width, CoeffBound noise 1 ∧
      (fun _ column ↦ reducePoly q 8 (placedReadout ν (select column))) =
        reduceMatrix q 8 1 width noise := by
  refine ⟨fun _ column ↦ intPoly (fun coefficient ↦
    value (readoutMember (select column) coefficient)), ?_, ?_⟩
  · intro row column coefficient
    rw [intPoly_coeff (by decide)]
    exact hbound _
  · funext row column
    exact placedReadout_reduce ν value hν (select column)

/-- The concrete modular half used by `w_1274` and every refresh part of step 0. -/
def smudgeHalf0 : Int :=
  904625697163682872125158399399175198226681342373069820760217620276403972497

/-- The exported half is exactly `(Q + 1) / 2`, and twice it is one modulo `Q`. -/
theorem smudgeHalf0_correct :
    smudgeHalf0 = ((W_Stage_online_0.modulus_0 : Int) + 1) / 2 ∧
      ((2 * smudgeHalf0 : Int) : ZMod W_Stage_online_0.modulus_0) = 1 := by
  constructor
  · norm_num [smudgeHalf0, W_Stage_online_0.modulus_0]
  · apply (ZMod.intCast_eq_intCast_iff (2 * smudgeHalf0) 1 _).2
    norm_num [Int.ModEq, smudgeHalf0, W_Stage_online_0.modulus_0]

/-- The coefficient readout's Walsh identity determines its signed odd digit modulo `q`.
The table hypothesis is the Fourier coefficient table, not an integer-lift equality. -/
theorem smudge_value_of_walsh {q n inverse stride log P : Nat}
    (hn : n = stride * 2 ^ log) (hnpos : 0 < n)
    (hinverse : (inverse : ExactPoly q n) * ((2 ^ log : Nat) : ExactPoly q n) = 1)
    (index : Fin (2 ^ log) → Nat) (hodd : ∀ branch, Odd (index branch))
    (perm : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ branch,
      index branch % (2 * 2 ^ log) = 1 + 2 * 1 * (perm branch).val)
    {C : Fin (2 ^ log) → ExactPoly q n}
    (hC : ∀ branch,
      C branch = lutFourierCoefficient (2 ^ log) (X q ^ stride)
        (inverse : ExactPoly q n)
        (fun entry : Fin (2 ^ log) ↦ reducePoly q n
          (readoutVal (P / 2) (2 ^ log) entry.val : ErrorPoly n)) (index branch))
    {μ ν : ErrorPoly n} {z : Nat}
    (hμ : reducePoly q n μ = reducePoly q n (Xe (n := n) (stride * z)))
    (hwalsh : ∑ branch, reducePoly q n
      (substitution (index branch) (hodd branch) μ) * C branch = reducePoly q n ν) :
    reducePoly q n ν = reducePoly q n
      (smudgeReadoutDigit P (2 ^ log) z : ErrorPoly n) := by
  have hvalue := lut_value_of_walsh hn hnpos
    (fun entry ↦ (readoutVal (P / 2) (2 ^ log) entry.val : ErrorPoly n))
    hinverse index hodd perm hindex hC hμ hwalsh
  rw [hvalue]
  congr 1
  simp only [smudgeReadoutDigit, Int.cast_mul, Int.cast_natCast]

/-- The sampler's Walsh identity gives a CBD(1) value with absolute coefficient bound one. -/
theorem cbd1_value_of_walsh {q n inverse stride log P width : Nat}
    (hP : P = 2 ^ log) (hP2 : 2 ≤ P) (hwidth : 0 < width)
    (hn : n = stride * 2 ^ log) (hnpos : 0 < n)
    (hinverse : (inverse : ExactPoly q n) * ((2 ^ log : Nat) : ExactPoly q n) = 1)
    (index : Fin (2 ^ log) → Nat) (hodd : ∀ branch, Odd (index branch))
    (perm : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ branch,
      index branch % (2 * 2 ^ log) = 1 + 2 * 1 * (perm branch).val)
    {C : Fin (2 ^ log) → ExactPoly q n}
    (hC : ∀ branch,
      C branch = lutFourierCoefficient (2 ^ log) (X q ^ stride)
        (inverse : ExactPoly q n)
        (fun entry ↦ reducePoly q n (binTable (n := n) (2 ^ log) (cbd1Bin P) entry))
        (index branch))
    {μ ν : ErrorPoly n} {z : Nat}
    (hμ : reducePoly q n μ =
      reducePoly q n (Xe (n := n) (stride * readoutExpG P width z)))
    (hwalsh : ∑ branch, reducePoly q n
      (substitution (index branch) (hodd branch) μ) * C branch = reducePoly q n ν) :
    reducePoly q n ν = reducePoly q n
        (samplerVal P width (cbd1Bin P) (z : Int) : ErrorPoly n) ∧
      (samplerVal P width (cbd1Bin P) (z : Int)).natAbs ≤ 1 := by
  exact ⟨sampler_value_gen hP hP2 hwidth hn hnpos (cbd1Bin P)
    hinverse index hodd perm hindex hC hμ hwalsh, cbd1_value_bound P width z⟩

/-- With a scalar offset, a nonconstant odd digit cannot have a small integer half.
In particular the proposed `BM = P / 2` fails whenever `2 * (P / 2) + (P - 1) < q`. -/
theorem smudge_scalar_offset_obstruction {q n P bound : Nat} (hq : 1 < q)
    (hn : 0 < n) (hP : 1 ≤ P) {half : Int}
    (hhalf : ((2 * half : Int) : ZMod q) = 1)
    (digit : Fin n → Int) (hodd : ∀ coefficient, digit coefficient % 2 = 1)
    (hbound : ∀ coefficient, |digit coefficient| ≤ (P : Int) - 1)
    (coefficient : Fin n) (hnonzero : coefficient.val ≠ 0)
    {noise : ErrorPoly n}
    (hvalue : reducePoly q n (intPoly digit) * (half : ExactPoly q n) - half =
      reducePoly q n noise)
    (hnoise : ∀ index, (noise.coeff index).natAbs ≤ bound)
    (hsmall : 2 * bound + (P - 1) < q) : False := by
  have hhalfpoly : (2 : ExactPoly q n) * (half : ExactPoly q n) = 1 := by
    have hcast : ((2 * half : Int) : ExactPoly q n) = 1 := by
      rw [intCast_eq_algebraMap, hhalf, map_one]
    simpa only [Int.cast_mul, Int.cast_ofNat] using hcast
  have htwice : reducePoly q n ((2 : ErrorPoly n) * noise) =
      reducePoly q n (intPoly digit) - 1 := by
    rw [map_mul, map_ofNat]
    calc
      _ = (2 : ExactPoly q n) *
          (reducePoly q n (intPoly digit) * half - half) := by rw [hvalue]
      _ = reducePoly q n (intPoly digit) * (2 * half) - 2 * half := by ring
      _ = _ := by rw [hhalfpoly, mul_one]
  have hone : (1 : ExactPoly q n).coeff coefficient = 0 := by
    have hconst : (1 : ErrorPoly n).coeff coefficient = 0 := by
      simpa only [Int.cast_one, if_neg hnonzero] using
        smudge_constant_coeff hn 1 coefficient
    have hone := reducePoly_coeff_eq hq hn (1 : ErrorPoly n) coefficient
    simpa only [map_one, hconst, Int.cast_zero] using hone
  have hcoeff := congrArg (fun polynomial ↦ polynomial.coeff coefficient) htwice
  dsimp only at hcoeff
  have hcoefftwo : ((2 : ErrorPoly n) * noise).coeff coefficient =
      2 * noise.coeff coefficient := by
    simpa only [Int.cast_ofNat] using intCast_mul_coeff 2 noise coefficient
  rw [reducePoly_coeff_eq hq hn, hcoefftwo, Negacyclic.coeff_sub,
    reducePoly_coeff_eq hq hn, intPoly_coeff hn, hone, sub_zero] at hcoeff
  have hdivides : (q : Int) ∣ digit coefficient - 2 * noise.coeff coefficient :=
    (ZMod.intCast_eq_intCast_iff_dvd_sub _ _ _).1 hcoeff
  have hdigit : (digit coefficient).natAbs ≤ P - 1 := by
    have hdigit := hbound coefficient
    rw [Int.abs_eq_natAbs] at hdigit
    omega
  have hsize : (digit coefficient - 2 * noise.coeff coefficient).natAbs < (q : Int).natAbs := by
    calc
      _ ≤ (digit coefficient).natAbs + (2 * noise.coeff coefficient).natAbs :=
        Int.natAbs_sub_le _ _
      _ ≤ (P - 1) + 2 * bound := by
        rw [Int.natAbs_mul]
        exact Nat.add_le_add hdigit (Nat.mul_le_mul_left 2 (hnoise coefficient))
      _ < _ := by simpa [Nat.add_comm, Int.natAbs_natCast] using hsmall
  have hzero := Int.eq_zero_of_dvd_of_natAbs_lt_natAbs hdivides hsize
  have := hodd coefficient
  omega

/-- The smudge-column slice used by part `i`, prime `p`, column `l` of step 0. -/
def smudgePartColumn (part : Fin 36) (prime : Fin 5) (column : Fin 25) : Fin 4500 :=
  ⟨125 * part.val + 25 * prime.val + column.val, by
    have := part.isLt
    have := prime.isLt
    have := column.isLt
    omega⟩

/-- The fresh-column slice used by part `i`, column `l` of step 0. -/
def freshPartColumn (part : Fin 36) (column : Fin 25) : Fin 900 :=
  ⟨25 * part.val + column.val, by
    have := part.isLt
    have := column.isLt
    omega⟩

/-- The exact `125 i + 25 p + l` smudge slice discharges `hTH` and `hM`, at the
actual scalar-offset bound `4 + |(Q + 1) / 2|`, not the intended bound four. -/
theorem online0_part_smudge_values (ν : Fin 36000 → ErrorPoly 8)
    (digit : Fin 36000 → Int)
    (hν : ∀ member, reducePoly W_Stage_online_0.modulus_0 8 (ν member) =
      reducePoly W_Stage_online_0.modulus_0 8 (digit member : ErrorPoly 8))
    (hodd : ∀ member, digit member % 2 = 1)
    (hbound : ∀ member, |digit member| ≤ 7) (part : Fin 36) (prime : Fin 5) :
    ∃ noise : ErrorMatrix 8 1 25, CoeffBound noise (4 + smudgeHalf0.natAbs) ∧
      Matrix.of (fun (_ : Fin 1) column ↦
        reducePoly W_Stage_online_0.modulus_0 8
          (placedReadout ν (smudgePartColumn part prime column)) *
            (smudgeHalf0 : ExactPoly W_Stage_online_0.modulus_0 8)) -
        Matrix.of (fun (_ : Fin 1) (_ : Fin 25) ↦
          (smudgeHalf0 : ExactPoly W_Stage_online_0.modulus_0 8)) =
        reduceMatrix W_Stage_online_0.modulus_0 8 1 25 noise := by
  exact online0_smudge_values (q := W_Stage_online_0.modulus_0) (columns := 4500)
    (width := 25) (P := 8) (by decide) smudgeHalf0_correct.2 ν digit hν hodd
    hbound (smudgePartColumn part prime)

/-- The exact `25 i + l` fresh slice has CBD coefficient bound one for chosen LUT lifts. -/
theorem online0_part_fresh_values (ν : Fin 7200 → ErrorPoly 8) (value : Fin 7200 → Int)
    (hν : ∀ member, ν member = (value member : ErrorPoly 8))
    (hbound : ∀ member, (value member).natAbs ≤ 1) (part : Fin 36) :
    CoeffBound (fun _ column ↦ placedReadout ν (freshPartColumn part column) :
      ErrorMatrix 8 1 25) 1 :=
  online0_fresh_values (columns := 900) (width := 25) ν value hν hbound
    (freshPartColumn part)

/-- The exported width-eight readout table is the canonical `2 f - 1` table for `P = 8`.
This is the table in `W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_4_10`. -/
theorem smudge_readout_table0 (entry : Fin 8) :
    MxxIR.selectInt (entry.val : Int) [1, 1, 3, 3, 5, 5, 7, 7] =
      (readoutVal 4 8 entry.val : Int) := by
  revert entry
  decide

/-- The exported CBD(1) table is precisely the odd-entry bin table.
This is the table in `W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_8_10`. -/
theorem cbd1_sampler_table0 {n : Nat} (entry : Fin 8) :
    (MxxIR.selectInt (entry.val : Int) [0, 0, 0, 0, 0, 1, 0, 1] : ErrorPoly n) =
      binTable (n := n) 8 (cbd1Bin 8) entry := by
  have htable : MxxIR.selectInt (entry.val : Int) [0, 0, 0, 0, 0, 1, 0, 1] =
      if entry.val % 2 = 1 then cbd1Bin 8 (entry.val / 2) else 0 := by
    revert entry
    decide
  rw [htable]
  unfold binTable
  split_ifs <;> simp

/-- A run of the generated coefficient-readout table returns the canonical odd digit.
This connects the table identity to the Program rather than assuming its literal output. -/
theorem smudge_readout_program0 {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_0_0.Params} (outer : Nat) (entry : Fin 8) {value : Int}
    (hprogram : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_4_10
      backend hashModel params outer entry.val () value) :
    value = readoutVal 4 8 entry.val := by
  have hvalue : value = MxxIR.selectInt (entry.val : Int) [1, 1, 3, 3, 5, 5, 7, 7] := by
    simpa only [add_zero] using hprogram.2.2
  exact hvalue.trans (smudge_readout_table0 entry)

/-- A run of the generated CBD table returns its canonical sampler polynomial.
Its odd-entry bin structure therefore follows from the generated Program's constraints. -/
theorem cbd1_sampler_program0 {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_preprocess_0_0.Params} {n : Nat}
    (outer : Nat) (entry : Fin 8) {value : Int}
    (hprogram : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_8_10
      backend hashModel params outer entry.val () value) :
    (value : ErrorPoly n) = binTable (n := n) 8 (cbd1Bin 8) entry := by
  have hvalue : value =
      MxxIR.selectInt (entry.val : Int) [0, 0, 0, 0, 0, 1, 0, 1] := by
    simpa only [add_zero] using hprogram.2.2
  rw [hvalue]
  exact cbd1_sampler_table0 entry

/-- The exported readout table and the online `hwalsh` family discharge all three value
hypotheses of `online0_part_smudge_values`: the residue, oddness, and digit bound seven.
The input exponents and Fourier coefficient identities remain explicit obligations. -/
theorem online0_smudge_digits_of_walsh {q inverse members : Nat}
    (hinverse : (inverse : ExactPoly q 8) * (8 : ExactPoly q 8) = 1)
    (index : Fin 8 → Nat) (hodd : ∀ branch, Odd (index branch))
    (perm : Fin 8 ≃ Fin 8)
    (hindex : ∀ branch, index branch % 16 = 1 + 2 * 1 * (perm branch).val)
    {C : Fin 8 → ExactPoly q 8}
    (hC : ∀ branch, C branch = lutFourierCoefficient 8 (X q ^ 1)
      (inverse : ExactPoly q 8)
      (fun entry : Fin 8 ↦ reducePoly q 8
        (MxxIR.selectInt (entry.val : Int) [1, 1, 3, 3, 5, 5, 7, 7] : ErrorPoly 8))
      (index branch))
    (μ ν : Fin members → ErrorPoly 8) (exponent : Fin members → Nat)
    (hμ : ∀ member, reducePoly q 8 (μ member) =
      reducePoly q 8 (Xe (n := 8) (exponent member)))
    (hwalsh : ∀ member, ∑ branch,
      reducePoly q 8 (substitution (index branch) (hodd branch) (μ member)) * C branch =
        reducePoly q 8 (ν member)) :
    ∀ member, reducePoly q 8 (ν member) = reducePoly q 8
        (smudgeReadoutDigit 8 8 (exponent member) : ErrorPoly 8) ∧
      smudgeReadoutDigit 8 8 (exponent member) % 2 = 1 ∧
      |smudgeReadoutDigit 8 8 (exponent member)| ≤ 7 := by
  have hcanonical : ∀ branch, C branch = lutFourierCoefficient 8 (X q ^ 1)
      (inverse : ExactPoly q 8)
      (fun entry : Fin 8 ↦ reducePoly q 8 (readoutVal 4 8 entry.val : ErrorPoly 8))
      (index branch) := by
    intro branch
    simpa only [smudge_readout_table0] using hC branch
  intro member
  refine ⟨?_, smudgeReadoutDigit_range (by decide) (by decide) (by decide) _⟩
  exact smudge_value_of_walsh (P := 8) (stride := 1) (log := 3)
    (by decide) (by decide) hinverse index hodd perm hindex hcanonical
    (by simpa only [one_mul] using hμ member) (hwalsh member)

/-- The exported sampler table and the online `hwalsh` family give CBD(1) residues and
bound one, exactly the hypotheses of `online0_fresh_value_representatives`.
An arbitrary integer lift `ν` is not itself asserted to have bounded coefficients. -/
theorem online0_cbd1_values_of_walsh {q inverse members width : Nat}
    (hwidth : 0 < width)
    (hinverse : (inverse : ExactPoly q 8) * (8 : ExactPoly q 8) = 1)
    (index : Fin 8 → Nat) (hodd : ∀ branch, Odd (index branch))
    (perm : Fin 8 ≃ Fin 8)
    (hindex : ∀ branch, index branch % 16 = 1 + 2 * 1 * (perm branch).val)
    {C : Fin 8 → ExactPoly q 8}
    (hC : ∀ branch, C branch = lutFourierCoefficient 8 (X q ^ 1)
      (inverse : ExactPoly q 8)
      (fun entry : Fin 8 ↦ reducePoly q 8
        (MxxIR.selectInt (entry.val : Int) [0, 0, 0, 0, 0, 1, 0, 1] : ErrorPoly 8))
      (index branch))
    (μ ν : Fin members → ErrorPoly 8) (exponent : Fin members → Nat)
    (hμ : ∀ member, reducePoly q 8 (μ member) =
      reducePoly q 8 (Xe (n := 8) (readoutExpG 8 width (exponent member))))
    (hwalsh : ∀ member, ∑ branch,
      reducePoly q 8 (substitution (index branch) (hodd branch) (μ member)) * C branch =
        reducePoly q 8 (ν member)) :
    ∀ member, reducePoly q 8 (ν member) = reducePoly q 8
        (samplerVal 8 width (cbd1Bin 8) (exponent member : Int) : ErrorPoly 8) ∧
      (samplerVal 8 width (cbd1Bin 8) (exponent member : Int)).natAbs ≤ 1 := by
  have hcanonical : ∀ branch, C branch = lutFourierCoefficient 8 (X q ^ 1)
      (inverse : ExactPoly q 8)
      (fun entry : Fin 8 ↦ reducePoly q 8 (binTable (n := 8) 8 (cbd1Bin 8) entry))
      (index branch) := by
    intro branch
    simpa only [cbd1_sampler_table0] using hC branch
  intro member
  exact cbd1_value_of_walsh (P := 8) (stride := 1) (log := 3)
    (by decide) (by decide) hwidth (by decide) (by decide)
    hinverse index hodd perm hindex hcanonical
    (by simpa only [one_mul] using hμ member) (hwalsh member)

/-- A fresh part admits a CBD-bounded representative from modular LUT values alone.
Replacing its target by this representative preserves every reduced target in `hf`. -/
theorem online0_part_fresh_value_representatives {q : Nat}
    (ν : Fin 7200 → ErrorPoly 8) (value : Fin 7200 → Int)
    (hν : ∀ member, reducePoly q 8 (ν member) =
      reducePoly q 8 (value member : ErrorPoly 8))
    (hbound : ∀ member, (value member).natAbs ≤ 1) (part : Fin 36) :
    ∃ noise : ErrorMatrix 8 1 25, CoeffBound noise 1 ∧
      (fun _ column ↦ reducePoly q 8 (placedReadout ν (freshPartColumn part column))) =
        reduceMatrix q 8 1 25 noise := by
  exact online0_fresh_value_representatives (columns := 900) (width := 25)
    ν value hν hbound (freshPartColumn part)

/-- At the exported parameters, no part's scalar-offset smudge admits coefficient bound four.
The obstruction holds for every part and prime, without enumerating their families. -/
theorem online0_part_smudge_small_bound_impossible (ν : Fin 36000 → ErrorPoly 8)
    (digit : Fin 36000 → Int)
    (hν : ∀ member, reducePoly W_Stage_online_0.modulus_0 8 (ν member) =
      reducePoly W_Stage_online_0.modulus_0 8 (digit member : ErrorPoly 8))
    (hodd : ∀ member, digit member % 2 = 1)
    (hbound : ∀ member, |digit member| ≤ 7) (part : Fin 36) (prime : Fin 5)
    {noise : ErrorMatrix 8 1 25}
    (hTH : Matrix.of (fun (_ : Fin 1) column ↦
        reducePoly W_Stage_online_0.modulus_0 8
          (placedReadout ν (smudgePartColumn part prime column)) *
            (smudgeHalf0 : ExactPoly W_Stage_online_0.modulus_0 8)) -
        Matrix.of (fun (_ : Fin 1) (_ : Fin 25) ↦
          (smudgeHalf0 : ExactPoly W_Stage_online_0.modulus_0 8)) =
        reduceMatrix W_Stage_online_0.modulus_0 8 1 25 noise)
    (hnoise : CoeffBound noise 4) : False := by
  have hentry := congrArg (fun matrix : ExactMatrix W_Stage_online_0.modulus_0 8 1 25 ↦
    matrix 0 0) hTH
  change reducePoly W_Stage_online_0.modulus_0 8
    (placedReadout ν (smudgePartColumn part prime 0)) * smudgeHalf0 - smudgeHalf0 =
      reducePoly W_Stage_online_0.modulus_0 8 (noise 0 0) at hentry
  rw [placedReadout_reduce (q := W_Stage_online_0.modulus_0) (columns := 4500)
    ν digit hν (smudgePartColumn part prime 0)] at hentry
  exact smudge_scalar_offset_obstruction (P := 8) (bound := 4)
    (by norm_num [W_Stage_online_0.modulus_0]) (by decide) (by decide)
    smudgeHalf0_correct.2 _ (fun _ ↦ hodd _) (fun _ ↦ hbound _)
    ⟨1, by decide⟩ (by decide) hentry (hnoise 0 0)
    (by norm_num [W_Stage_online_0.modulus_0])

end Dio2Proof
