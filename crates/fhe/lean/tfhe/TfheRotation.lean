import TfheStages

/-!
The blind rotation as explicit functions of the sampling tape. `accSeq` is the accumulator after
each external product, and `errSeq` the integer error of its phase: the error is rotated with the
accumulator and gains the digit-weighted key error of each step. Every execution of the stage
computes exactly `accSeq`.
-/

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime

/-! ## The gadget decomposition as a function -/

/-- The digits of a two-row column under the ring gadget. -/
noncomputable def decompose (diff : ExactMatrix Q N 2 1) : ExactMatrix Q N 12 1 :=
  castMatrixRows (by rw [layout2_digitCount]) (regularDecomposeMatrix FheBackend.layout2 diff)

theorem gadgetDecomposeRuns_eq {diff : ExactMatrix Q N 2 1} {D : ExactMatrix Q N 12 1}
    (hD : gadgetDecomposeRuns FheBackend.backend 64 6 diff D) : D = decompose diff := by
  obtain ⟨layout, hl, _, _, _, hdeq, _⟩ := hD
  rw [backend_layout] at hl
  cases hl
  rw [hdeq]
  rfl

/-- The first six digits rebuild row zero and the last six rebuild row one. -/
theorem gadget_reconstruct (diff : ExactMatrix Q N 2 1) :
    (∑ c : Fin 6, gadgetRow 0 c * decompose diff (Fin.castAdd 6 c) 0) = diff 0 0 ∧
    (∑ c : Fin 6, gadgetRow 0 c * decompose diff (Fin.natAdd 6 c) 0) = diff 1 0 := by
  have hrec := regularGadgetMatrix_reconstruct FheBackend.layout2 diff (by decide) (by decide)
    layout2_exact
  refine ⟨?_, ?_⟩ <;>
  · rw [← congrFun (congrFun hrec _) 0, Matrix.mul_apply]
    simp only [gadgetRow, decompose, castMatrixColumns_apply, castMatrixRows_apply]
    rw [← (finCongr (by rw [layout2_digitCount] : 6 + 6 = 2 * FheBackend.layout2.digitCount)).sum_comp]
    rw [Fin.sum_univ_add]
    simp only [regularGadgetMatrix_two_rows FheBackend.layout2 layout2_exact, finCongr_apply,
      Fin.val_cast, Fin.val_castAdd, Fin.val_natAdd]
    have hlow (x : Fin 6) : (x : Nat) / FheBackend.layout2.digitCount = 0 := by
      rw [layout2_digitCount]; exact Nat.div_eq_of_lt x.isLt
    have hhigh (x : Fin 6) : (6 + (x : Nat)) / FheBackend.layout2.digitCount = 1 := by
      rw [layout2_digitCount]; omega
    simp only [hlow, hhigh, Fin.val_zero, Fin.val_one, if_true, if_false, zero_ne_one,
      one_ne_zero, zero_mul, Finset.sum_const_zero, add_zero, zero_add]
    apply Finset.sum_congr rfl
    intro x _
    congr 2
    apply Fin.ext
    simp [layout2_digitCount, Nat.mod_eq_of_lt x.isLt]

/-- The centered integer digits. -/
noncomputable def digitInts (diff : ExactMatrix Q N 2 1) : ErrorMatrix N 12 1 :=
  fun r c ↦ centeredIntLift (decompose diff r c)

theorem decompose_eq_reduce (diff : ExactMatrix Q N 2 1) (r : Fin 12) (c : Fin 1) :
    decompose diff r c = reducePoly Q N (digitInts diff r c) :=
  (reducePoly_centeredIntLift (by decide) (by decide) _).symm

/-- Every digit is within `32`. -/
theorem digitInts_bound (diff : ExactMatrix Q N 2 1) (r : Fin 12) (c : Fin 1) (k : Fin N) :
    |(digitInts diff r c).coeff k| ≤ 32 := by
  have hbound := regularDecomposeMatrix_bounded FheBackend.layout2 diff (by decide) (by decide)
  obtain ⟨witness, hwitness, hb⟩ : PreimageWithin (decompose diff) (FheBackend.layout2.base / 2) :=
    preimageWithin_castMatrixRows _ hbound
  have hbase : FheBackend.layout2.base / 2 = 32 := by decide
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

/-- The values of one run fixed by the hash model and the external inputs: the encryption masks
and messages. -/
structure World where
  maskL : Fin lweN → Int
  maskR : Fin lweN → Int
  messageL : Int
  messageR : Int

noncomputable abbrev rootQ : ExactPoly Q N := AdjoinRoot.root (negacyclicModulus N (ZMod Q))
noncomputable abbrev rootZ : ErrorPoly N := AdjoinRoot.root (negacyclicModulus N Int)

/-- The lookup-table coefficient `floor(Q / 8)`. -/
abbrev lutValue : Int := 654329600

noncomputable def lutPoly : ExactPoly Q N := reducePoly Q N (intPoly fun _ ↦ lutValue)

/-- An encryption body. -/
def body (mask : Fin lweN → Int) (message : Int) (t : SampleTape) (stage : Nat) : Int :=
  ((∑ j, mask j * lweSecret t j) + (message * 2 - 1) * Δ + encError t stage) % q

/-- The body of `Δ - left - right`. -/
def combinedBody (W : World) (t : SampleTape) : Int :=
  ((0 - (body W.maskL W.messageL t 1 + body W.maskR W.messageR t 2) % q) % q + Δ) % q

/-- The mask of `Δ - left - right`. -/
def combinedMask (W : World) (j : Fin lweN) : Int := (0 - (W.maskL j + W.maskR j) % q) % q

/-- Rounding from `q` to `2 N`. -/
def roundRing (x : Int) : Int := ((x * 2048 + 2147483648) / 4294967296) % 2048

def rotation (W : World) (j : Fin lweN) : Int := roundRing (combinedMask W j)

def initialExp (W : World) (t : SampleTape) : Int := (0 - roundRing (combinedBody W t)) % 2048

theorem roundRing_range (x : Int) : 0 ≤ roundRing x ∧ roundRing x < 2048 := by
  unfold roundRing; omega

theorem initialExp_range (W : World) (t : SampleTape) :
    0 ≤ initialExp W t ∧ initialExp W t < 2048 := by
  unfold initialExp; omega

noncomputable def accInit (W : World) (t : SampleTape) : ExactMatrix Q N 2 1 :=
  fun r _ ↦ if r = 0 then 0 else lutPoly * rootQ ^ (initialExp W t).toNat

/-- Bootstrapping key `i` as one two-row matrix. -/
noncomputable def bskRows (t : SampleTape) (i : Fin lweN) : ExactMatrix Q N 2 12 :=
  fun r c ↦ if r = 0 then gswA t i 0 c else gswB t i 0 c

noncomputable def diffOf (W : World) (i : Fin lweN) (acc : ExactMatrix Q N 2 1) :
    ExactMatrix Q N 2 1 :=
  matrixSub (multiplyMonomial acc (rotation W i)) acc

noncomputable def accStep (W : World) (t : SampleTape) (i : Fin lweN) (acc : ExactMatrix Q N 2 1) :
    ExactMatrix Q N 2 1 :=
  matrixAdd acc (matrixMul (bskRows t i) (decompose (diffOf W i acc)))

noncomputable def accSeq (W : World) (t : SampleTape) : Nat → ExactMatrix Q N 2 1
  | 0 => accInit W t
  | n + 1 => if h : n < lweN then accStep W t ⟨n, h⟩ (accSeq W t n) else accSeq W t n

/-- The integer digits of step `i`. -/
noncomputable def stepDigits (W : World) (t : SampleTape) (i : Fin lweN) : ErrorMatrix N 12 1 :=
  digitInts (diffOf W i (accSeq W t i))

/-- The key error added by step `i`. -/
noncomputable def stepNoise (W : World) (t : SampleTape) (i : Fin lweN) : ErrorPoly N :=
  ∑ c : Fin 12, bskError t i 0 c * stepDigits W t i c 0

/-- The rotation applied at step `i`. -/
def stepShift (W : World) (t : SampleTape) (i : Fin lweN) : Nat :=
  (lweSecret t i).toNat * (rotation W i).toNat

noncomputable def errSeq (W : World) (t : SampleTape) : Nat → ErrorPoly N
  | 0 => 0
  | n + 1 => if h : n < lweN then
      rootZ ^ stepShift W t ⟨n, h⟩ * errSeq W t n + stepNoise W t ⟨n, h⟩ else errSeq W t n

def expSeq (W : World) (t : SampleTape) (n : Nat) : Nat :=
  (initialExp W t).toNat + ∑ j ∈ Finset.range n,
    if h : j < lweN then stepShift W t ⟨j, h⟩ else 0

theorem reducePoly_rootZ_pow (k : Nat) : reducePoly Q N (rootZ ^ k) = rootQ ^ k := by
  rw [map_pow]
  simp [reducePoly]

theorem multiplyMonomial_apply {rows columns : Nat} (input : ExactMatrix Q N rows columns)
    (exponent : Int) (row : Fin rows) (column : Fin columns) :
    multiplyMonomial input exponent row column =
      input row column * rootQ ^ (exponent % (2 * (N : Int))).toNat := rfl

/-- The phase of every accumulator: the rotated lookup table plus the reduced integer error. -/
theorem accSeq_phase (W : World) (t : SampleTape) (hs : ∀ j, lweSecret t j = 0 ∨ lweSecret t j = 1)
    (n : Nat) :
    accSeq W t n 1 0 - reducePoly Q N (ringSecret t) * accSeq W t n 0 0 =
      rootQ ^ expSeq W t n * lutPoly + reducePoly Q N (errSeq W t n) := by
  set z := reducePoly Q N (ringSecret t)
  induction n with
  | zero =>
    simp [accSeq, accInit, errSeq, expSeq, mul_comm]
  | succ n ih =>
    by_cases hn : n < lweN
    · set i : Fin lweN := ⟨n, hn⟩
      set acc := accSeq W t n
      set e := rotation W i
      have he : 0 ≤ e ∧ e < 2048 := roundRing_range (combinedMask W i)
      have he' : (e % (2 * (N : Int))).toNat = e.toNat := by
        rw [Int.emod_eq_of_lt he.1 (by simp only [N, Nat.cast_ofNat]; omega)]
      obtain ⟨hrow0, hrow1⟩ := gadget_reconstruct (diffOf W i acc)
      set D := decompose (diffOf W i acc)
      set Dw := stepDigits W t i
      have hD (c : Fin 12) : D c 0 = reducePoly Q N (Dw c 0) := decompose_eq_reduce _ c 0
      have hdiff (r : Fin 2) : diffOf W i acc r 0 = acc r 0 * rootQ ^ e.toNat - acc r 0 := by
        show acc r 0 * rootQ ^ (e % (2 * (N : Int))).toNat - acc r 0 = _
        rw [he']
      rw [hdiff] at hrow0 hrow1
      -- The external product's phase.
      have hext : (matrixMul (bskRows t i) D) 1 0 - z * (matrixMul (bskRows t i) D) 0 0 =
          reducePoly Q N (stepNoise W t i) +
            (lweSecret t i : ExactPoly Q N) * ((rootQ ^ e.toNat - 1) * (acc 1 0 - z * acc 0 0)) := by
        have hsum (r : Fin 2) : (matrixMul (bskRows t i) D) r 0 = ∑ c : Fin 12, bskRows t i r c * D c 0 :=
          Matrix.mul_apply
        have hcast (c : Fin 6) : bskRows t i 1 (Fin.castAdd 6 c) * D (Fin.castAdd 6 c) 0 -
            z * (bskRows t i 0 (Fin.castAdd 6 c) * D (Fin.castAdd 6 c) 0) =
            reducePoly Q N (bskError t i 0 (Fin.castAdd 6 c) * Dw (Fin.castAdd 6 c) 0) -
              z * (lweSecret t i : ExactPoly Q N) * (gadgetRow 0 c * D (Fin.castAdd 6 c) 0) := by
          simp only [bskRows, gswA, gswB, Fin.val_castAdd, c.isLt, dif_pos, Fin.eta, if_pos,
            one_ne_zero, if_false, if_true]
          rw [dif_neg (show ¬ (6 ≤ c.val) by omega), map_mul, ← hD]
          ring
        have hnat (c : Fin 6) : bskRows t i 1 (Fin.natAdd 6 c) * D (Fin.natAdd 6 c) 0 -
            z * (bskRows t i 0 (Fin.natAdd 6 c) * D (Fin.natAdd 6 c) 0) =
            reducePoly Q N (bskError t i 0 (Fin.natAdd 6 c) * Dw (Fin.natAdd 6 c) 0) +
              (lweSecret t i : ExactPoly Q N) * (gadgetRow 0 c * D (Fin.natAdd 6 c) 0) := by
          simp only [bskRows, gswA, gswB, Fin.val_natAdd, one_ne_zero, if_false, if_true]
          rw [dif_neg (show ¬ (6 + c.val < 6) by omega), dif_pos (show 6 ≤ 6 + c.val by omega),
            map_mul, ← hD]
          have hc : (⟨6 + c.val - 6, by omega⟩ : Fin 6) = c := Fin.ext (by simp)
          rw [hc]
          ring
        have hsplit (f : Fin 12 → ExactPoly Q N) : ∑ c, f c =
            ∑ c : Fin 6, f (Fin.castAdd 6 c) + ∑ c : Fin 6, f (Fin.natAdd 6 c) :=
          Fin.sum_univ_add (a := 6) (b := 6) f
        rw [hsum, hsum, Finset.mul_sum, ← Finset.sum_sub_distrib, hsplit,
          show stepNoise W t i = ∑ c : Fin 12, bskError t i 0 c * Dw c 0 from rfl, map_sum,
          hsplit (fun c ↦ reducePoly Q N (bskError t i 0 c * Dw c 0))]
        simp only [hcast, hnat, Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
          hrow0, hrow1]
        ring
      have hstep : accSeq W t (n + 1) = accStep W t i acc := by
        simp only [accSeq, dif_pos hn]
        rfl
      have hsplit : accStep W t i acc 1 0 - z * accStep W t i acc 0 0 =
          (acc 1 0 - z * acc 0 0) +
            ((matrixMul (bskRows t i) D) 1 0 - z * (matrixMul (bskRows t i) D) 0 0) := by
        simp only [accStep, matrixAdd, Matrix.add_apply]
        ring
      have herr : errSeq W t (n + 1) = rootZ ^ stepShift W t i * errSeq W t n + stepNoise W t i := by
        simp only [errSeq, dif_pos hn]
        rfl
      have hexp : expSeq W t (n + 1) = expSeq W t n + stepShift W t i := by
        simp only [expSeq, Finset.sum_range_succ, dif_pos hn]
        ring
      rw [hstep, hsplit, hext, ih, herr, hexp, map_add, map_mul, reducePoly_rootZ_pow]
      simp only [stepShift]
      rcases hs i with h0 | h1
      · rw [h0]
        simp only [Int.cast_zero, zero_mul, add_zero, Int.toNat_zero, pow_zero, one_mul]
        ring
      · rw [h1]
        simp only [Int.cast_one, one_mul, Int.toNat_one]
        ring
    · have hstep : accSeq W t (n + 1) = accSeq W t n := by simp [accSeq, hn]
      have herr : errSeq W t (n + 1) = errSeq W t n := by simp [errSeq, hn]
      have hexp : expSeq W t (n + 1) = expSeq W t n := by simp [expSeq, Finset.sum_range_succ, hn]
      rw [hstep, herr, hexp]
      exact ih

/-- Every execution of the blind-rotation scope computes `accSeq`. -/
theorem blind_rotation_runs {W : World} {t : SampleTape} {tape : SampleTape} {path : List Nat}
    {A B : Fin lweN → ExactMatrix Q N 1 12} (hA : A = gswA t) (hB : B = gswB t)
    {P : ExactMatrix Q N 1 1} (hP : P 0 0 = lutPoly) {e0 : Int} (he0 : e0 = initialExp W t)
    {masks : Fin lweN → Int} (hmasks : masks = combinedMask W) {outputs}
    (h : Stage_nand.scope_tfhe_blind_rotation FheBackend.backend tape path { unit := () }
      (0, multiplyMonomial P e0, masks, A, B, ()) outputs) :
    outputs.1 0 0 = accSeq W t lweN 0 0 ∧ outputs.2.1 0 0 = accSeq W t lweN 1 0 := by
  obtain ⟨witness, hcat, _, hround, _, hiter, _, _, _, _, _, _, hs8, _, _, _, _, _, _, hs9, hout⟩ := h
  subst hout
  have hr (i : Fin lweN) : witness.w_6_0 i = rotation W i := by
    obtain ⟨_, _, hi⟩ := hround i
    rw [hi, hmasks]
    rfl
  have hfinal := MxxIR.IterRuns.invariant
    (Invariant := fun i (acc : ExactMatrix Q N 2 1) ↦ acc = accSeq W t i) ?_ ?_ hiter
  · dsimp only at hfinal ⊢
    rw [sliceMatrix_entry hs8 (by decide) (by decide), sliceMatrix_entry hs9 (by decide) (by decide),
      hfinal]
    exact ⟨rfl, rfl⟩
  · obtain ⟨h0, h1⟩ := concatRows_one_one hcat 0
    have he0' : (initialExp W t % (2 * (N : Int))).toNat = (initialExp W t).toNat := by
      have := initialExp_range W t
      rw [Int.emod_eq_of_lt this.1 (by simp only [N, Nat.cast_ofNat]; omega)]
    funext r c
    have hc : c = 0 := Subsingleton.elim _ _
    subst hc
    fin_cases r
    · show witness.w_2_0 0 0 = _
      rw [h0]
      rfl
    · show witness.w_2_0 1 0 = _
      rw [h1]
      show P 0 0 * rootQ ^ (e0 % (2 * (N : Int))).toNat = _
      rw [he0, he0', hP]
      rfl
  · intro i current next hcurrent hstep
    obtain ⟨w3, w5, w6, w8, w11, _, _, hAi, _, _, hBi, hC, _, _, he, hD, hnext⟩ := hstep
    obtain ⟨pA, hpA, rfl⟩ := hAi
    obtain ⟨pB, hpB, rfl⟩ := hBi
    obtain ⟨pe, hpe, rfl⟩ := he
    simp only [Int.ofNat_eq_natCast, Nat.cast_inj] at hpA hpB hpe
    have hi : i < lweN := hpA ▸ pA.isLt
    have hpA' : pA = ⟨i, hi⟩ := Fin.ext hpA
    have hpB' : pB = ⟨i, hi⟩ := Fin.ext hpB
    have hpe' : pe = ⟨i, hi⟩ := Fin.ext hpe
    subst hpA' hpB' hpe' hnext hcurrent
    have hCeq : w6 = bskRows t ⟨i, hi⟩ := by
      funext r c
      obtain ⟨h0, h1⟩ := concatRows_one_one hC c
      fin_cases r
      · show w6 0 c = _
        rw [h0, hA]; rfl
      · show w6 1 c = _
        rw [h1, hB]; rfl
    dsimp only at hD ⊢
    rw [gadgetDecomposeRuns_eq hD, hCeq, hr]
    simp only [accSeq, dif_pos hi]
    rfl

end MxxFheTfhe
