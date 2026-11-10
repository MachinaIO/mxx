import TfhePhase

/-!
# The phase of the blind rotation

`brSeq` with bootstrapping keys of the GSW form `gswOf` keeps the phase `acc₁ - z acc₀` of its
accumulator equal to the rotated lookup table plus an integer error `errSeq`: each external
product multiplies the phase by `X^{s_i r_i}` and adds the bootstrapping-key errors weighted by
the step's digits.
-/

set_option maxRecDepth 16384

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime MxxSecurity

/-! ## The gadget decomposition -/

/-- The first six digits rebuild row zero and the last six rebuild row one. -/
theorem gadget_reconstruct (diff : ExactMatrix Q N 2 1) :
    (∑ c : Fin 6, gadgetRow 0 c * decompose diff (Fin.castAdd 6 c) 0) = diff 0 0 ∧
    (∑ c : Fin 6, gadgetRow 0 c * decompose diff (Fin.natAdd 6 c) 0) = diff 1 0 := by
  have hrec := regularGadgetMatrix_reconstruct Backend.layout0 diff (by decide) (by decide)
    layout_exact
  refine ⟨?_, ?_⟩ <;>
  · rw [← congrFun (congrFun hrec _) 0, Matrix.mul_apply]
    simp only [gadgetRow, decompose, castMatrixColumns_apply, castMatrixRows_apply]
    rw [← (finCongr (by rw [layout_digitCount] : 6 + 6 = 2 * Backend.layout0.digitCount)).sum_comp]
    rw [Fin.sum_univ_add]
    simp only [regularGadgetMatrix_two_rows Backend.layout0 layout_exact, finCongr_apply,
      Fin.val_cast, Fin.val_castAdd, Fin.val_natAdd]
    have hlow (x : Fin 6) : (x : Nat) / Backend.layout0.digitCount = 0 := by
      rw [layout_digitCount]; exact Nat.div_eq_of_lt x.isLt
    have hhigh (x : Fin 6) : (6 + (x : Nat)) / Backend.layout0.digitCount = 1 := by
      rw [layout_digitCount]; omega
    simp only [hlow, hhigh, Fin.val_zero, Fin.val_one, if_true, if_false, zero_ne_one,
      one_ne_zero, zero_mul, Finset.sum_const_zero, add_zero, zero_add]
    apply Finset.sum_congr rfl
    intro x _
    congr 2
    apply Fin.ext
    simp [layout_digitCount, Nat.mod_eq_of_lt x.isLt]

/-- The centered integer digits. -/
noncomputable def digitInts (diff : ExactMatrix Q N 2 1) : ErrorMatrix N 12 1 :=
  fun r c ↦ centeredIntLift (decompose diff r c)

theorem decompose_eq_reduce (diff : ExactMatrix Q N 2 1) (r : Fin 12) (c : Fin 1) :
    decompose diff r c = reducePoly Q N (digitInts diff r c) :=
  (reducePoly_centeredIntLift (by decide) (by decide) _).symm

/-- Every digit is within `32`. -/
theorem digitInts_bound (diff : ExactMatrix Q N 2 1) (r : Fin 12) (c : Fin 1) (k : Fin N) :
    |(digitInts diff r c).coeff k| ≤ 32 := by
  have hbound := regularDecomposeMatrix_bounded Backend.layout0 diff (by decide) (by decide)
  obtain ⟨witness, hwitness, hb⟩ : PreimageWithin (decompose diff) (Backend.layout0.base / 2) :=
    preimageWithin_castMatrixRows _ hbound
  have hbase : Backend.layout0.base / 2 = 32 := by decide
  rw [hbase] at hb
  have hentry : decompose diff r c = reducePoly Q N (witness r c) := by
    rw [show decompose diff = _ from hwitness]
    rfl
  have hsmall : ∀ i, 2 * ((witness r c).coeff i).natAbs < Q := fun i ↦ by
    have := hb r c i; unfold Q; omega
  rw [digitInts, hentry, centeredIntLift_reducePoly (by decide) (by decide) _ hsmall,
    Int.abs_eq_natAbs]
  exact_mod_cast hb r c k


/-! ## The accumulator -/

noncomputable abbrev rootQ : ExactPoly Q N := AdjoinRoot.root (negacyclicModulus N (ZMod Q))
noncomputable abbrev rootZ : ErrorPoly N := AdjoinRoot.root (negacyclicModulus N Int)

theorem reducePoly_rootZ_pow (k : Nat) : reducePoly Q N (rootZ ^ k) = rootQ ^ k := by
  rw [map_pow]
  simp [reducePoly]

section Accumulator

variable (A E : ExactMatrix Q N 1 8640) (Eint : ErrorMatrix N 1 8640) (zM : ExactMatrix Q N 1 1)
  (s : Fin 720 → Int) (rot : Fin 720 → Int) (init : ExactMatrix Q N 2 1)

/-- The accumulator after `n` external products with the keys `gswOf A E zM (s i) i`. -/
noncomputable def accOf (n : Nat) : ExactMatrix Q N 2 1 :=
  brSeq (fun i ↦ (gswOf A E zM (s i) i).1) (fun i ↦ (gswOf A E zM (s i) i).2.1) rot init n

/-- The difference that step `i` decomposes. -/
noncomputable def diffOf (i : Fin 720) (acc : ExactMatrix Q N 2 1) : ExactMatrix Q N 2 1 :=
  matrixSub (multiplyMonomial acc (rot i)) acc

/-- The integer digits of step `i`. -/
noncomputable def stepDigits (i : Fin 720) : ErrorMatrix N 12 1 :=
  digitInts (diffOf rot i (accOf A E zM s rot init i))

/-- The key error added by step `i`. -/
noncomputable def stepNoise (i : Fin 720) : ErrorPoly N :=
  ∑ c : Fin 12, Eint 0 ⟨12 * i.val + c.val, by omega⟩ * stepDigits A E zM s rot init i c 0

/-- The rotation applied at step `i`. -/
def stepShift (i : Fin 720) : Nat := (s i).toNat * (rot i).toNat

/-- The integer error of the accumulator phase. -/
noncomputable def errSeq : Nat → ErrorPoly N
  | 0 => 0
  | n + 1 => if h : n < 720 then
      rootZ ^ stepShift s rot ⟨n, h⟩ * errSeq n + stepNoise A E Eint zM s rot init ⟨n, h⟩
      else errSeq n

/-- The total rotation after `n` steps. -/
def expSeq (n : Nat) : Nat :=
  ∑ j ∈ Finset.range n, if h : j < 720 then stepShift s rot ⟨j, h⟩ else 0

end Accumulator

/-- The phase of every accumulator: the rotated initial phase plus the reduced integer error. -/
theorem accOf_phase (A E : ExactMatrix Q N 1 8640) (Eint : ErrorMatrix N 1 8640)
    (hE : E = reduceMatrix Q N 1 8640 Eint) (zM : ExactMatrix Q N 1 1) (s : Fin 720 → Int)
    (hs : ∀ j, s j = 0 ∨ s j = 1) (rot : Fin 720 → Int) (hrot : ∀ i, 0 ≤ rot i ∧ rot i < 2048)
    (init : ExactMatrix Q N 2 1) (n : Nat) :
    accOf A E zM s rot init n 1 0 - zM 0 0 * accOf A E zM s rot init n 0 0 =
      rootQ ^ expSeq s rot n * (init 1 0 - zM 0 0 * init 0 0) +
        reducePoly Q N (errSeq A E Eint zM s rot init n) := by
  set z := zM 0 0
  induction n with
  | zero =>
    simp [accOf, brSeq, errSeq, expSeq]
  | succ n ih =>
    by_cases hn : n < 720
    · set i : Fin 720 := ⟨n, hn⟩
      set acc := accOf A E zM s rot init n
      set e := rot i
      have hri := hrot i
      have he : (e % (2 * (N : Int))).toNat = e.toNat := by
        rw [Int.emod_eq_of_lt hri.1 (by simp only [N, Nat.cast_ofNat]; omega)]
      obtain ⟨hrow0, hrow1⟩ := gadget_reconstruct (diffOf rot i acc)
      set D := decompose (diffOf rot i acc)
      set Dw := stepDigits A E zM s rot init i
      have hD (c : Fin 12) : D c 0 = reducePoly Q N (Dw c 0) := decompose_eq_reduce _ c 0
      have hdiff (r : Fin 2) : diffOf rot i acc r 0 = acc r 0 * rootQ ^ e.toNat - acc r 0 := by
        show acc r 0 * rootQ ^ (e % (2 * (N : Int))).toNat - acc r 0 = _
        rw [he]
      rw [hdiff] at hrow0 hrow1
      set M := concatRowsOf (gswOf A E zM (s i) i).1 (gswOf A E zM (s i) i).2.1
      have hM0 (c : Fin 12) : M 0 c = A 0 ⟨12 * i.val + c.val, by omega⟩ +
          if h : c.val < 6 then (s i : ExactPoly Q N) * gadgetRow 0 ⟨c.val, h⟩ else 0 := by
        simp only [M, concatRowsOf, gswOf, matrixAdd, concatOf, matrixMulScalarLeft, liftInteger,
          Matrix.add_apply, Fin.val_zero, Nat.lt_one_iff, dite_true, Matrix.zero_apply]
        rfl
      have hM1 (c : Fin 12) : M 1 c = z * A 0 ⟨12 * i.val + c.val, by omega⟩ +
          reducePoly Q N (Eint 0 ⟨12 * i.val + c.val, by omega⟩) +
          if h : 6 ≤ c.val then (s i : ExactPoly Q N) * gadgetRow 0 ⟨c.val - 6, by omega⟩
            else 0 := by
        simp only [M, concatRowsOf, gswOf, matrixAdd, concatOf, matrixMulScalarLeft, liftInteger,
          Matrix.add_apply, hE, reduceMatrix, Matrix.zero_apply]
        by_cases hc : c.val < 6
        · simp only [hc, show ¬ (6 ≤ c.val) by omega, dite_true, dite_false, add_zero]
          rfl
        · simp only [hc, show 6 ≤ c.val by omega, dite_true, dite_false, zero_add]
          rfl
      have hext : (matrixMul M D) 1 0 - z * (matrixMul M D) 0 0 =
          reducePoly Q N (stepNoise A E Eint zM s rot init i) +
            (s i : ExactPoly Q N) * ((rootQ ^ e.toNat - 1) * (acc 1 0 - z * acc 0 0)) := by
        have hsum (r : Fin 2) : (matrixMul M D) r 0 = ∑ c : Fin 12, M r c * D c 0 :=
          Matrix.mul_apply
        have hcast (c : Fin 6) : M 1 (Fin.castAdd 6 c) * D (Fin.castAdd 6 c) 0 -
            z * (M 0 (Fin.castAdd 6 c) * D (Fin.castAdd 6 c) 0) =
            reducePoly Q N (Eint 0 ⟨12 * i.val + c.val, by omega⟩ * Dw (Fin.castAdd 6 c) 0) -
              z * (s i : ExactPoly Q N) * (gadgetRow 0 c * D (Fin.castAdd 6 c) 0) := by
          rw [hM0, hM1]
          simp only [Fin.val_castAdd, c.isLt, dif_pos, show ¬ (6 ≤ c.val) by omega, dif_neg,
            Fin.eta, map_mul, ← hD, not_false_eq_true, dite_false, add_zero]
          ring
        have hnat (c : Fin 6) : M 1 (Fin.natAdd 6 c) * D (Fin.natAdd 6 c) 0 -
            z * (M 0 (Fin.natAdd 6 c) * D (Fin.natAdd 6 c) 0) =
            reducePoly Q N (Eint 0 ⟨12 * i.val + (6 + c.val), by omega⟩ * Dw (Fin.natAdd 6 c) 0) +
              (s i : ExactPoly Q N) * (gadgetRow 0 c * D (Fin.natAdd 6 c) 0) := by
          rw [hM0, hM1]
          simp only [Fin.val_natAdd, show ¬ (6 + c.val < 6) by omega, dif_neg,
            show 6 ≤ 6 + c.val by omega, dif_pos, map_mul, ← hD, not_false_eq_true, dite_false,
            add_zero, mul_zero]
          have hc : (⟨6 + c.val - 6, by omega⟩ : Fin 6) = c := Fin.ext (by simp)
          rw [hc]
          ring
        have hsplit (f : Fin 12 → ExactPoly Q N) : ∑ c, f c =
            ∑ c : Fin 6, f (Fin.castAdd 6 c) + ∑ c : Fin 6, f (Fin.natAdd 6 c) :=
          Fin.sum_univ_add (a := 6) (b := 6) f
        have hnoise : reducePoly Q N (stepNoise A E Eint zM s rot init i) =
            ∑ c : Fin 6, reducePoly Q N (Eint 0 ⟨12 * i.val + c.val, by omega⟩ *
              Dw (Fin.castAdd 6 c) 0) +
            ∑ c : Fin 6, reducePoly Q N (Eint 0 ⟨12 * i.val + (6 + c.val), by omega⟩ *
              Dw (Fin.natAdd 6 c) 0) := by
          rw [stepNoise, map_sum, hsplit]
          rfl
        rw [hsum, hsum, Finset.mul_sum, ← Finset.sum_sub_distrib, hsplit, hnoise]
        simp only [hcast, hnat, Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
          hrow0, hrow1]
        ring
      have hstep : accOf A E zM s rot init (n + 1) =
          matrixAdd acc (matrixMul M (decompose (diffOf rot i acc))) := by
        simp only [accOf, brSeq, dif_pos hn]
        rfl
      have hsplit : (matrixAdd acc (matrixMul M D)) 1 0 - z * (matrixAdd acc (matrixMul M D)) 0 0 =
          (acc 1 0 - z * acc 0 0) + ((matrixMul M D) 1 0 - z * (matrixMul M D) 0 0) := by
        simp only [matrixAdd, Matrix.add_apply]
        ring
      have herr : errSeq A E Eint zM s rot init (n + 1) =
          rootZ ^ stepShift s rot i * errSeq A E Eint zM s rot init n +
            stepNoise A E Eint zM s rot init i := by
        simp only [errSeq, dif_pos hn]
        rfl
      have hexp : expSeq s rot (n + 1) = expSeq s rot n + stepShift s rot i := by
        simp only [expSeq, Finset.sum_range_succ, dif_pos hn]
        rfl
      rw [hstep, hsplit, hext, ih, herr, hexp, map_add, map_mul, reducePoly_rootZ_pow]
      simp only [stepShift]
      rcases hs i with h0 | h1
      · rw [h0]
        simp only [Int.cast_zero, zero_mul, add_zero, Int.toNat_zero, pow_zero, one_mul]
        ring
      · rw [h1]
        simp only [Int.cast_one, one_mul, Int.toNat_one]
        ring
    · have hstep : accOf A E zM s rot init (n + 1) = accOf A E zM s rot init n := by
        simp [accOf, brSeq, hn]
      have herr : errSeq A E Eint zM s rot init (n + 1) = errSeq A E Eint zM s rot init n := by
        simp [errSeq, hn]
      have hexp : expSeq s rot (n + 1) = expSeq s rot n := by
        simp [expSeq, Finset.sum_range_succ, hn]
      rw [hstep, herr, hexp]
      exact ih

end MxxFheTfhe
