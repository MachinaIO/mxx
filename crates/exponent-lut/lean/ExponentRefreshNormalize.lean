import ExponentRefreshDigits
namespace MxxExponentLut
open Mxx.Primitives
open scoped BigOperators

/-- Coefficient assembly preserves the supplied readout choices. -/
noncomputable def refreshNativeRadixMatrix {q n columns width count : Nat}
    (base : Nat) (native : Fin (2 * width) → ExactPoly q n)
    (selection : Fin 1 → Fin columns → Fin n → Fin count → Fin (2 * width)) :
    ExactMatrix q n 1 columns := fun row col ↦
  ∑ k : Fin n, (∑ d : Fin count, (base : ExactPoly q n) ^ d.val *
    native (selection row col k d)) * AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ k.val

/-- Public offsets occupy every coefficient of every output column. -/
noncomputable def refreshOffsetMatrix {q n columns : Nat} (offset : ExactPoly q n) :
    ExactMatrix q n 1 columns := fun _ _ ↦
  ∑ k : Fin n, offset * AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ k.val

theorem refresh_integer_assembly {q n columns : Nat}
    (digits : Fin 1 → Fin columns → Fin n → Int)
    (native : Fin 1 → Fin columns → Fin n → ExactPoly q n)
    (offset : ExactPoly q n)
    (identity : ∀ row col k, (digits row col k : ExactPoly q n) = native row col k + offset) :
    ∀ row col, reduceMatrix q n 1 columns (refreshFreshLift digits) row col =
      (∑ k : Fin n, native row col k *
        AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ k.val) +
          refreshOffsetMatrix (q := q) (n := n) (columns := columns) offset row col := by
  intro row col
  simp only [reduceMatrix, refreshFreshLift, map_sum, map_mul, map_pow]
  simp only [reducePoly, map_intCast]
  simp_rw [identity, add_mul]
  simp [refreshOffsetMatrix, Finset.sum_add_distrib]

/-- Positive routed native sums and the actual public offsets complete the digits. -/
theorem refresh_high_ideal_factor {q n columns : Nat} (J delta : Int)
    (payload fresh mask : ErrorMatrix n 1 columns)
    (nativeFresh nativeMask freshOffset maskOffset : ExactMatrix q n 1 columns)
    (freshAssembly : reduceMatrix q n 1 columns fresh = nativeFresh + freshOffset)
    (maskAssembly : reduceMatrix q n 1 columns mask = nativeMask + maskOffset) :
    -(J * delta) • reduceMatrix q n 1 columns payload + J • nativeMask + J • maskOffset +
      (J * delta) • nativeFresh + (J * delta) • freshOffset =
    reduceMatrix q n 1 columns (J • (delta • (-payload + fresh) + mask)) := by
  have scalar (a : Int) (v : ErrorMatrix n 1 columns) :
      reduceMatrix q n 1 columns (a • v) = a • reduceMatrix q n 1 columns v := by
    ext row col
    exact map_zsmul (reducePoly q n) a (v row col)
  have neg : reduceMatrix q n 1 columns (-payload) = -reduceMatrix q n 1 columns payload := by
    ext row col
    exact map_neg (reducePoly q n) _
  rw [scalar, reduceMatrix_add, scalar, reduceMatrix_add, neg, freshAssembly, maskAssembly]
  simp only [smul_add, smul_neg, smul_smul, neg_smul]
  abel

/-- The retained decoder correction gives one integer anchor for every modulus. -/
theorem refresh_decoder_ideal {q n inner columns : Nat} (delta : Int)
    (secret : ErrorMatrix n 1 inner) (publicKey : ErrorMatrix n inner columns)
    (payload fresh mask : ErrorMatrix n 1 columns) :
    reduceMatrix q n 1 columns (delta • (-payload + fresh) + mask) +
      reduceMatrix q n 1 inner secret * (delta • reduceMatrix q n inner columns publicKey) =
    reduceMatrix q n 1 columns (delta • (secret * publicKey - payload + fresh) + mask) := by
  have scalar (v : ErrorMatrix n 1 columns) :
      reduceMatrix q n 1 columns (delta • v) = delta • reduceMatrix q n 1 columns v := by
    ext row col
    exact map_zsmul (reducePoly q n) delta (v row col)
  simp only [sub_eq_add_neg, reduceMatrix_add, scalar, reduceMatrix_mul,
    smul_add, smul_neg, Matrix.mul_smul]
  abel
theorem refresh_decoder_ideal_coeff {q n inner columns : Nat}
    (hq : 1 < q) (hn : 0 < n) (delta : Int)
    (secret : ErrorMatrix n 1 inner) (publicKey : ErrorMatrix n inner columns)
    (payload fresh mask : ErrorMatrix n 1 columns) (row col coefficient) :
    ((reduceMatrix q n 1 columns (delta • (-payload + fresh) + mask) +
      reduceMatrix q n 1 inner secret * (delta • reduceMatrix q n inner columns publicKey))
      row col).coeff coefficient =
      (delta * ((secret * publicKey - payload + fresh) row col).coeff coefficient +
        (mask row col).coeff coefficient : Int) := by
  rw [refresh_decoder_ideal]
  simp only [reduceMatrix]
  rw [reducePoly_coeff hq hn]
  have scalar (p : ErrorPoly n) : ((delta : ErrorPoly n) * p).coeff coefficient =
      delta * p.coeff coefficient := Negacyclic.coeff_smul delta p coefficient
  simp only [Matrix.add_apply, Matrix.smul_apply, zsmul_eq_mul, scalar,
    Negacyclic.coeff_add]

/-- Canonical coefficients of the common high public input define one integer
lift whose ordinary reduction is the actual runtime low public matrix. -/
theorem refresh_public_integer_reduce {q p n columns : Nat}
    (hp : 1 < p) (hn : 0 < n) (input : ExactMatrix q n 2 columns) :
    reduceMatrix p n 2 columns
      (fun row col ↦ ∑ k : Fin n,
        (Int.ofNat ((input row col).coeff k).val : ErrorPoly n) *
          AdjoinRoot.root (negacyclicModulus n Int) ^ k.val) =
      MxxRuntime.modulusReduce input := by
  letI : Fact (1 < p) := ⟨hp⟩
  ext row col
  suffices h : ∀ k, (reduceMatrix p n 2 columns
      (fun row col ↦ ∑ k : Fin n, (Int.ofNat ((input row col).coeff k).val : ErrorPoly n) *
        AdjoinRoot.root (negacyclicModulus n Int) ^ k.val) row col).coeff k =
      (MxxRuntime.modulusReduce input row col : ExactPoly p n).coeff k by
    calc
      _ = ∑ k : Fin n, algebraMap (ZMod p) (ExactPoly p n)
          ((MxxRuntime.modulusReduce input row col : ExactPoly p n).coeff k) *
            AdjoinRoot.root (negacyclicModulus n (ZMod p)) ^ k.val := by
        simp_rw [← h]
        exact Negacyclic.expansion hn _
      _ = _ := (Negacyclic.expansion hn _).symm
  intro k
  simp only [reduceMatrix, MxxRuntime.modulusReduce]
  rw [reducePoly_coeff hp hn, MxxRuntime.polynomialOfCoefficients_coeff hp hn]
  have coeff := refreshFreshLift_coeff (columns := columns) hn
    (fun _ col k ↦ Int.ofNat ((input row col).coeff k).val) 0 col k
  exact congrArg (fun z : Int ↦ (z : ZMod p)) coeff

theorem refresh_public_integer_source {q n columns : Nat}
    (hq : 1 < q) (hn : 0 < n) (input : ExactMatrix q n 2 columns) :
    reduceMatrix q n 2 columns
      (fun row col ↦ ∑ k : Fin n,
        (Int.ofNat ((input row col).coeff k).val : ErrorPoly n) *
          AdjoinRoot.root (negacyclicModulus n Int) ^ k.val) = input := by
  ext row col
  exact congrArg (fun matrix : ExactMatrix q n 1 columns ↦ matrix 0 col)
    (refresh_anchor_lift hq hn (fun _ col ↦ input row col))

/-- Constant public columns remain constant through the actual concat chain. -/
theorem refresh_concat_constant {q n rows left right : Nat}
    {a : ExactMatrix q n rows left} {b : ExactMatrix q n rows right}
    {out : ExactMatrix q n rows (left + right)} (value : ExactPoly q n)
    (runs : MxxRuntime.concatColumns a b out)
    (ha : ∀ row col, a row col = value) (hb : ∀ row col, b row col = value) :
    ∀ row col, out row col = value := by
  intro row col
  rw [runs row col]
  split_ifs <;> simp only [ha, hb]

end MxxExponentLut
