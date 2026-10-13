import ExponentRuntime
import ExponentScale
import ExponentDecode

/-!
# The decode stage's CRT recombination for any parameter set

The decode stage of the refresh of every CRT slot lifts the residue of each of `k + 1` slots
`q_0, …, q_k` (centered) to the decoding modulus `Q = ∏ q_i`, recombines the lifts with a CRT
basis `K_i ≡ [i = j] (mod q_j)`, subtracts the expected row and threshold-decodes every coefficient.
This file states these steps for any ring dimension `n`, slot count, column count `m` and modulus,
without enumerating slots, columns or coefficients:

* `crt_int_gen`: integers congruent to `V` modulo each `q_j` recombine to `V` modulo `∏ q_j`.
* `crtSum`: the left fold `((l_0 K_0 + l_1 K_1) + …) + l_k K_k` the stage builds, and
  `crt_matrix_gen`: it recombines the lifts of the residues of `v` to `v`.
* `residue_sub_gen`, `decode_small_gen`, `column_zero_gen`: the subtraction of the expected row and
  the decode of a small difference, at any modulus and column count.
* `sigmaFold`: the left fold `Σ_o s_o X^o` the stage builds σ with, and `sigmaFold_eq`.

The toy instance's lemmas (`TopCrt`, `TopDecode`, `TopSigma`) follow in `TopCrtGenericToy`.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

variable {n : Nat}

/-! ## The integer CRT -/

/-- Integers congruent to `V` modulo each of the pairwise coprime `q_j` recombine with a CRT basis
`K` (`K_i ≡ [i = j] (mod q_j)`) to `V` modulo `∏ q_j`. -/
theorem crt_int_gen {k : Nat} (q : Fin k → Nat) (K : Fin k → Int)
    (hK : ∀ i j, K i ≡ (if i = j then 1 else 0) [ZMOD (q j : Int)])
    (hcop : ∀ i j, i ≠ j → Nat.Coprime (q i) (q j)) {x : Fin k → Int} {V : Int}
    (hx : ∀ j, x j ≡ V [ZMOD (q j : Int)]) :
    ∑ i, x i * K i ≡ V [ZMOD (∏ i, (q i : Int))] := by
  have hj : ∀ j, ∑ i, x i * K i ≡ V [ZMOD (q j : Int)] := by
    intro j
    refine Int.ModEq.trans ?_ (hx j)
    rw [← ZMod.intCast_eq_intCast_iff]
    push_cast
    have hKj : ∀ i, ((K i : Int) : ZMod (q j)) = if i = j then 1 else 0 := by
      intro i
      have := (ZMod.intCast_eq_intCast_iff _ _ _).2 (hK i j)
      rw [this]
      split <;> simp
    simp only [hKj, mul_ite, mul_one, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, if_true]
  apply Int.modEq_of_dvd
  apply Finset.prod_dvd_of_coprime
  · intro i _ j _ hij
    exact Nat.isCoprime_iff_coprime.2 (hcop i j hij)
  · intro i _
    exact (hj i).dvd

/-! ## Residues and their centered lifts -/

/-- `res` is the residue of `v` modulo its prime, coefficient by coefficient. -/
def ResidueAt {q Q m : Nat} (res : ExactMatrix q n 1 m) (v : ExactMatrix Q n 1 m) : Prop :=
  ∀ c i, (res 0 c).coeff i = ((((v 0 c).coeff i).val : Nat) : ZMod q)

/-- `l` is the centered lift to `Q` of a residue of `v` modulo `q` (a slot's lift in the decode
stage). Packing the residue existentially keeps the slot family non-dependent. -/
def LiftOf {Q m : Nat} (q : Nat) (l v : ExactMatrix Q n 1 m) : Prop :=
  ∃ r : ExactMatrix q n 1 m, ResidueAt r v ∧ centeredRebaseRuns r l

theorem liftOf_intro {q Q m : Nat} {r : ExactMatrix q n 1 m} {l v : ExactMatrix Q n 1 m}
    (h : ResidueAt r v) (hl : centeredRebaseRuns r l) : LiftOf q l v := ⟨r, h, hl⟩

/-- A lift's coefficient is an integer congruent to the value modulo the slot's prime. -/
theorem liftOf_coeff {q Q m : Nat} (hn : 0 < n) (hq : 1 < q) {l v : ExactMatrix Q n 1 m}
    (h : LiftOf q l v) (c : Fin m) (i : Fin n) :
    ∃ x : Int, (l 0 c).coeff i = (x : ZMod Q) ∧ x ≡ (((v 0 c).coeff i).val : Int) [ZMOD (q : Int)] := by
  obtain ⟨r, hr, hl⟩ := h
  have hQ : 1 < Q := hl.1
  refine ⟨(centeredIntLift (r 0 c)).coeff i, ?_, ?_⟩
  · rw [centeredRebaseRuns_reduce hl, reducePoly_coeff_eq hQ hn]
  · rw [← ZMod.intCast_eq_intCast_iff]
    have e := congrArg (fun p : ExactPoly q n ↦ p.coeff i) (reducePoly_centeredIntLift hq hn (r 0 c))
    simp only at e
    rw [reducePoly_coeff_eq hq hn] at e
    rw [e, hr c i, Int.cast_natCast]

/-! ## The matrix-level recombination -/

/-- The decode stage's recombination `((l_0 K_0 + l_1 K_1) + …) + l_k K_k` of `k + 1` lifts. -/
noncomputable def crtSum {Q m : Nat} :
    (k : Nat) → (Fin (k + 1) → ExactMatrix Q n 1 m) → (Fin (k + 1) → Int) → ExactMatrix Q n 1 m
  | 0, l, K => matrixMulScalarRight (l 0) (matrixPolynomial [K 0])
  | k + 1, l, K => matrixAdd (crtSum k (fun i ↦ l i.castSucc) (fun i ↦ K i.castSucc))
      (matrixMulScalarRight (l (Fin.last (k + 1))) (matrixPolynomial [K (Fin.last (k + 1))]))

theorem crtSum_apply {Q m : Nat} (k : Nat) (l : Fin (k + 1) → ExactMatrix Q n 1 m)
    (K : Fin (k + 1) → Int) (r : Fin 1) (c : Fin m) :
    crtSum k l K r c = ∑ i, (K i : ExactPoly Q n) * l i r c := by
  induction k with
  | zero =>
    simp [crtSum, matrixMulScalarRight, MxxRuntime.matrixPolynomial_single, mul_comm]
  | succ k ih =>
    rw [crtSum, matrixAdd, Matrix.add_apply, ih]
    conv_rhs => rw [Fin.sum_univ_castSucc]
    simp [matrixMulScalarRight, MxxRuntime.matrixPolynomial_single, mul_comm]

theorem coeff_intCast_mul_gen {q : Nat} (K : Int) (p : ExactPoly q n) (i : Fin n) :
    ((K : ExactPoly q n) * p).coeff i = (K : ZMod q) * p.coeff i := by
  rw [← map_intCast (algebraMap (ZMod q) (ExactPoly q n)) K, Negacyclic.coeff_smul]

/-- The decode stage's recombination of the lifts of the residues of `v` modulo the pairwise
coprime slots `q_0, …, q_k` of `Q = ∏ q_i` is `v`. -/
theorem crt_matrix_gen {Q m k : Nat} (hn : 0 < n) (q : Fin (k + 1) → Nat) (hq : ∀ i, 1 < q i)
    (K : Fin (k + 1) → Int) (hK : ∀ i j, K i ≡ (if i = j then 1 else 0) [ZMOD (q j : Int)])
    (hcop : ∀ i j, i ≠ j → Nat.Coprime (q i) (q j)) (hprod : Q = ∏ i, q i)
    {v : ExactMatrix Q n 1 m} {l : Fin (k + 1) → ExactMatrix Q n 1 m}
    (hl : ∀ i, LiftOf (q i) (l i) v) : crtSum k l K = v := by
  funext row c
  obtain rfl : row = 0 := Subsingleton.elim _ _
  have hQ : 1 < Q := (hl 0).choose_spec.2.1
  haveI : NeZero Q := ⟨by omega⟩
  haveI : Fact (1 < Q) := ⟨hQ⟩
  rw [crtSum_apply]
  apply negacyclic_ext hn
  intro j
  rw [Negacyclic.coeff_sum]
  simp only [coeff_intCast_mul_gen]
  choose x hx hxq using fun i ↦ liftOf_coeff hn (hq i) (hl i) c j
  simp only [hx]
  have e := crt_int_gen q K hK hcop hxq
  rw [← Nat.cast_prod, ← hprod, ← ZMod.intCast_eq_intCast_iff] at e
  push_cast at e
  rw [ZMod.natCast_zmod_val] at e
  rw [← e]
  exact Finset.sum_congr rfl fun i _ ↦ mul_comm _ _

/-! ## The subtraction of the expected row and the decode -/

/-- The difference of the residue of `Y + e'` modulo `q` and `Y` reduced to `q` is `e'`. -/
theorem residue_sub_gen {Q q m : Nat} (hn : 0 < n) (hQ : 1 < Q) {Y : ExactMatrix Q n 1 m}
    {Yr res : ExactMatrix q n 1 m} (E : ErrorMatrix n 1 m)
    (h : ResidueAt res (Y + reduceMatrix Q n 1 m E)) (hY : modulusReduceRuns Y Yr) :
    matrixSub res Yr = reduceMatrix q n 1 m E := by
  haveI : NeZero Q := ⟨by omega⟩
  obtain ⟨hq, hdvd, rfl⟩ := hY
  haveI : Fact (1 < q) := ⟨hq⟩
  funext row c
  obtain rfl : row = 0 := Subsingleton.elim _ _
  apply negacyclic_ext hn
  intro i
  have hc : ∀ x : ZMod Q, ((x.val : ℕ) : ZMod q) = ZMod.cast x := fun x ↦ (ZMod.cast_eq_val x).symm
  simp only [matrixSub, Matrix.sub_apply, Negacyclic.coeff_sub, reduceMatrix, modulusReduce]
  rw [h c i, polynomialOfCoefficients_coeff hq hn, reducePoly_coeff_eq hq hn]
  simp only [Matrix.add_apply, Negacyclic.coeff_add, reduceMatrix, reducePoly_coeff_eq hQ hn]
  rw [hc, ZMod.cast_add hdvd, ZMod.cast_intCast hdvd]
  rw [Int.ofNat_eq_natCast, Int.cast_natCast, hc]
  ring

/-- A coefficient `e` with `4 |e| + 4 < q` threshold-decodes to zero, at any length. -/
theorem decode_small_gen {q : Nat} (hn : 0 < n) (hq : 1 < q) (hodd : q % 2 = 1)
    {D : ExactMatrix q n 1 1} {e : ErrorPoly n} (hD : D 0 0 = reducePoly q n e)
    (he : ∀ i, 4 * |e.coeff i| + 4 < q) {length position out : Int}
    (h : thresholdDecode 2 length position D out) : out = 0 := by
  obtain ⟨_, _, _, _, index, _, _, hout⟩ := h
  rw [hout, hD, reducePoly_coeff_eq hq hn]
  have := threshold_decode_value hq hodd (b := 0) (Or.inl rfl) (he index)
  simpa using this

theorem slice_entry_gen {q columns : Nat} {D : ExactMatrix q n 1 columns}
    {x : ExactMatrix q n 1 1} {a b : Int} (h : sliceMatrix D 0 1 a b x) (ha : a.toNat < columns) :
    x 0 0 = D 0 ⟨a.toNat, ha⟩ := by
  obtain ⟨_, _, _, _, _, _, _, _, hx⟩ := h
  rw [hx 0 0 (by simp) (by simpa using ha)]
  simp

/-- A column of a small difference decodes to zero, at any odd modulus and column count. -/
theorem column_zero_gen {Q m : Nat} (hn : 0 < n) (hQ : 1 < Q) (hodd : Q % 2 = 1)
    (E : ErrorMatrix n 1 m) (hE : ∀ c i, 4 * |(E 0 c).coeff i| + 4 < (Q : Int))
    {x : ExactMatrix Q n 1 1} {a b : Int} (h : sliceMatrix (reduceMatrix Q n 1 m E) 0 1 a b x)
    (ha : a.toNat < m) {length position out : Int} (hd : thresholdDecode 2 length position x out) :
    out = 0 :=
  decode_small_gen hn hQ hodd (slice_entry_gen h ha) (fun i ↦ hE _ i) hd

/-! ## σ from its coefficients -/

/-- σ as the decode stage builds it from `k + 1` coefficients:
`((s_0 X^0 + s_1 X^1) + …) + s_k X^k`. -/
noncomputable def sigmaFold {Q : Nat} : (k : Nat) → (Fin (k + 1) → Int) → ExactMatrix Q n 1 1
  | 0, s => multiplyMonomial (liftInteger (s 0)) 0
  | k + 1, s => matrixAdd (sigmaFold k fun i ↦ s i.castSucc)
      (multiplyMonomial (liftInteger (s (Fin.last (k + 1)))) ((k + 1 : Nat) : Int))

/-- σ is the reduction of `Σ_o s_o X^o` when its degree stays below `2 n`. -/
theorem sigmaFold_eq {Q : Nat} (k : Nat) (hk : k < 2 * n) (s : Fin (k + 1) → Int) :
    (sigmaFold k s : ExactMatrix Q n 1 1) =
      fun _ _ ↦ reducePoly Q n (∑ o : Fin (k + 1), ((s o : Int) : ErrorPoly n) * Xe (n := n) o) := by
  induction k with
  | zero =>
    funext r c
    simp [sigmaFold, multiplyMonomial_liftInteger, message, Xe]
  | succ k ih =>
    funext r c
    rw [sigmaFold, matrixAdd, Matrix.add_apply, ih (by omega), multiplyMonomial_liftInteger]
    conv_rhs => rw [Fin.sum_univ_castSucc, map_add]
    congr 1
    have he : (((k + 1 : Nat) : Int) % ((2 * n : Nat) : Int)).toNat = k + 1 := by
      rw [Int.emod_eq_of_lt (by omega) (by omega)]
      rfl
    simp only [message, he, Xe, Fin.val_last]

end RefreshTopProof
