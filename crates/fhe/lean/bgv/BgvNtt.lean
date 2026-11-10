import MxxRuntime
import RuntimeLemmas
import Mathlib.RingTheory.RootsOfUnity.PrimitiveRoots
import Mathlib.Algebra.Polynomial.Roots
import Mathlib.Tactic.NormNum.Prime

/-!
# The plaintext NTT is a bijection

Over a prime field, evaluating a polynomial of degree below `n` at the `n` odd powers of a
primitive `2n`-th root of unity is injective, hence bijective: the inverse transform of the BGV
slot encoding exists and is unique. The plaintext modulus `t = 1032193` has the primitive root
`94`.
-/

namespace BgvSecurity

open Mxx.Primitives MxxRuntime Polynomial

abbrev n : Nat := 8192
abbrev t : Nat := 1032193

instance : Fact (Nat.Prime t) := ⟨by norm_num⟩

/-- The odd-power evaluations of a coefficient vector. -/
def oddEval {K : Type} [CommRing K] {n : Nat} (ω : K) (c : Fin n → K) : Fin n → K :=
  fun i ↦ ∑ k : Fin n, c k * ω ^ ((2 * i.val + 1) * k.val)

theorem oddEval_injective {K : Type} [Field K] {n : Nat} {ω : K}
    (hω : IsPrimitiveRoot ω (2 * n)) : Function.Injective (oddEval (n := n) ω) := by
  intro c c' h
  by_cases hn : n = 0
  · subst hn
    funext i
    exact i.elim0
  let p : K[X] := ∑ k : Fin n, C (c k - c' k) * X ^ k.val
  have hdeg : p.natDegree < n := by
    refine lt_of_le_of_lt (natDegree_sum_le_of_forall_le _ _ (n := n - 1) fun k _ ↦ ?_) (by omega)
    refine (natDegree_C_mul_le _ _).trans ?_
    rw [natDegree_X_pow]
    omega
  have hpoints : Function.Injective fun i : Fin n ↦ ω ^ (2 * i.val + 1) := by
    intro i j hij
    exact Fin.ext (by have := hω.pow_inj (by omega) (by omega) hij; omega)
  have hzero : p = 0 := by
    refine eq_zero_of_natDegree_lt_card_of_eval_eq_zero p hpoints (fun i ↦ ?_) (by simpa using hdeg)
    have hi := congrFun h i
    simp only [oddEval] at hi
    simp only [p, eval_finset_sum, eval_mul, eval_C, eval_pow, eval_X, sub_mul,
      Finset.sum_sub_distrib, ← pow_mul]
    rw [hi, sub_self]
  funext k
  have hk := congrArg (fun p : K[X] ↦ p.coeff k.val) hzero
  simp only [p, finset_sum_coeff, coeff_C_mul, coeff_X_pow, coeff_zero] at hk
  rw [Finset.sum_eq_single k (fun j _ hj ↦ by rw [if_neg (fun h ↦ hj (Fin.ext h.symm)), mul_zero])
    (by simp), if_pos rfl, mul_one] at hk
  exact sub_eq_zero.mp hk

theorem oddEval_bijective {p : Nat} [Fact p.Prime] {n : Nat} {ω : ZMod p}
    (hω : IsPrimitiveRoot ω (2 * n)) : Function.Bijective (oddEval (n := n) ω) :=
  (Finite.injective_iff_bijective).mp (oddEval_injective hω)

/-- `94` is a primitive `2n`-th root of unity modulo `t`. -/
theorem root94_primitive : IsPrimitiveRoot ((94 : Nat) : ZMod t) (2 * n) := by
  have hneg : ((94 : Nat) : ZMod t) ^ 2 ^ 13 = -1 := by
    rw [← Nat.cast_pow, ← ZMod.natCast_mod]
    rw [show 94 ^ 2 ^ 13 % t = t - 1 by decide +kernel]
    rw [Nat.cast_sub (by decide), ZMod.natCast_self, Nat.cast_one, zero_sub]
  have hnot : ¬ ((94 : Nat) : ZMod t) ^ 2 ^ 13 = 1 := by
    rw [hneg]
    intro h
    have : ((2 : Nat) : ZMod t) = 0 := by
      have := congrArg (· + 1) h
      simp only [neg_add_cancel] at this
      rw [Nat.cast_ofNat]
      linear_combination -this
    rw [ZMod.natCast_eq_zero_iff] at this
    exact absurd (Nat.le_of_dvd (by decide) this) (by decide)
  have hfin : ((94 : Nat) : ZMod t) ^ 2 ^ (13 + 1) = 1 := by
    rw [pow_succ, pow_mul, hneg]
    norm_num
  have horder := orderOf_eq_prime_pow hnot hfin
  rw [show 2 * n = 2 ^ (13 + 1) by norm_num, ← horder]
  exact IsPrimitiveRoot.orderOf _

/-- A Nat-level primitive root is a primitive root in `ZMod`. -/
theorem isPrimitiveRoot_of_ntt {p : Nat} [Fact p.Prime] {n root : Nat} (hn : 0 < n)
    (h : nttPrimitiveRoot p n root) : IsPrimitiveRoot ((root : Nat) : ZMod p) (2 * n) := by
  obtain ⟨_, _, hfull, hlower⟩ := h
  refine IsPrimitiveRoot.mk_of_lt _ (by omega) ?_ ?_
  · rw [← Nat.cast_pow, ← ZMod.natCast_mod, hfull, Nat.cast_one]
  · intro l hl hlt heq
    apply hlower l hl hlt
    have : ((root ^ l : Nat) : ZMod p) = ((1 : Nat) : ZMod p) := by push_cast; exact heq
    rw [ZMod.natCast_eq_natCast_iff'] at this
    rw [this]
    exact Nat.one_mod_eq_one.mpr (Fact.out : p.Prime).one_lt.ne'

theorem nttPrimitiveRoot_94 : nttPrimitiveRoot t n 94 := by
  refine ⟨by decide, by decide, by decide +kernel, fun e he het ↦ ?_⟩
  intro heq
  have := root94_primitive.pow_eq_one_iff_dvd e |>.mp (by
    rw [← Nat.cast_pow, ← ZMod.natCast_mod, heq, Nat.cast_one])
  exact absurd (Nat.le_of_dvd he this) (by omega)

/-- The native root of the plaintext ring is a primitive `2n`-th root. -/
theorem native_primitive : nttPrimitiveRoot t n (nativeNttRoot t n) := by
  unfold nativeNttRoot
  exact Nat.sInf_mem (s := {root | nttPrimitiveRoot t n root}) ⟨94, nttPrimitiveRoot_94⟩

theorem native_lt : nativeNttRoot t n < t := native_primitive.2.1

theorem native_root_primitive : IsPrimitiveRoot ((nativeNttRoot t n : Nat) : ZMod t) (2 * n) :=
  isPrimitiveRoot_of_ntt (by decide) native_primitive

set_option maxRecDepth 100000 in
theorem bitReverse_lt (k : Nat) (hk : k < n) : nttBitReverse 13 k < n := by
  have := checkBelow_spec (test := fun k ↦ decide (bitReverseFast 13 k 0 < 8192))
    (bound := 8192) (by decide +kernel) k hk
  simpa only [bitReverseFast_eq, zero_mul, zero_add, decide_eq_true_eq] using this

theorem zmod_val_cast {x : ZMod t} : (((x.val : Nat) : Int) : ZMod t) = x := by
  rw [Int.cast_natCast, ZMod.natCast_zmod_val]

/-- The coefficients of a plaintext polynomial, as elements of `ZMod t`. -/
noncomputable def coeffsOf (o : ExactMatrix t n 1 1) : Fin n → ZMod t := fun i ↦ (o 0 0).coeff i

theorem fromValues_coeffs {input : Fin n → Int} {o : ExactMatrix t n 1 1}
    (h : polynomialFromValues true input o) :
    oddEval ((nativeNttRoot t n : Nat) : ZMod t) (coeffsOf o) =
      fun index ↦ ((input ⟨nttBitReverse 13 index.val, bitReverse_lt _ index.isLt⟩ : Int) : ZMod t) := by
  have h' : polynomialNttRuns t true input (fun i ↦ (((o 0 0).coeff i).val : Int)) := h
  obtain ⟨_, _, bits, hbits, heval⟩ := polynomialNttRuns_eval h'
  have hb : bits = 13 := by
    have : 2 ^ bits = 2 ^ 13 := by rw [← hbits]; norm_num
    exact Nat.pow_right_injective (le_refl 2) this
  subst hb
  funext index
  obtain ⟨slot, hslot, hvalue⟩ := heval index
  simp only [if_true] at hvalue
  have : slot = ⟨nttBitReverse 13 index.val, bitReverse_lt _ index.isLt⟩ := Fin.ext hslot
  subst this
  rw [hvalue]
  simp only [oddEval, coeffsOf, zmod_val_cast]

theorem matrix_one_one_ext {q : Nat} [Fact (1 < q)] {o o' : ExactMatrix q n 1 1}
    (h : ∀ i, (o 0 0).coeff i = (o' 0 0).coeff i) : o = o' := by
  funext r c
  rw [Subsingleton.elim r 0, Subsingleton.elim c 0]
  exact negacyclic_ext (by decide) h

instance : Fact (1 < t) := ⟨by decide⟩

/-- The inverse transform has at most one output. -/
theorem fromValues_unique {input : Fin n → Int} {o o' : ExactMatrix t n 1 1}
    (h : polynomialFromValues true input o) (h' : polynomialFromValues true input o') : o = o' := by
  have hc := oddEval_injective native_root_primitive
    ((fromValues_coeffs h).trans (fromValues_coeffs h').symm)
  exact matrix_one_one_ext fun i ↦ congrFun hc i

/-- The inverse transform has an output. -/
theorem fromValues_exists (input : Fin n → Int) :
    ∃ o : ExactMatrix t n 1 1, polynomialFromValues true input o := by
  obtain ⟨c, hc⟩ := (oddEval_bijective native_root_primitive).2
    (fun index : Fin n ↦ ((input ⟨nttBitReverse 13 index.val, bitReverse_lt _ index.isLt⟩ : Int) :
      ZMod t))
  refine ⟨fun _ _ ↦ polynomialOfCoefficients fun k ↦ ((c k).val : Int), ?_⟩
  have hcoeff (k : Fin n) : ((polynomialOfCoefficients (q := t) fun k ↦ ((c k).val : Int)).coeff k)
      = c k := by
    rw [polynomialOfCoefficients_coeff (by decide) (by decide), zmod_val_cast]
  show polynomialNttRuns t true input _
  refine ⟨by decide, fun prime hprime hdvd ↦ ?_, 13, nativeNttRoot t n, by norm_num, native_lt,
    fun prime hprime hdvd ↦ ?_, fun i ↦ ⟨by positivity,
      Int.ofNat_lt.mpr (ZMod.val_lt _)⟩, ?_⟩
  · have hp : prime ∣ t := dvd_trans (dvd_mul_right prime prime) hdvd
    have hpt := (Nat.prime_dvd_prime_iff_eq hprime (Fact.out : t.Prime)).mp hp
    subst hpt
    exact absurd (Nat.le_of_dvd (by decide) hdvd) (by decide)
  · have hpt := (Nat.prime_dvd_prime_iff_eq hprime (Fact.out : t.Prime)).mp hdvd
    subst hpt
    exact ⟨by decide, by rw [Nat.mod_eq_of_lt native_lt]; exact native_primitive,
      Nat.mod_eq_of_lt native_lt⟩
  · intro index
    refine ⟨⟨nttBitReverse 13 index.val, bitReverse_lt _ index.isLt⟩, rfl, ?_⟩
    rw [← ZMod.intCast_eq_intCast_iff']
    have hi := congrFun hc index
    simp only [oddEval] at hi
    rw [← hi]
    push_cast
    refine Finset.sum_congr rfl fun k _ ↦ ?_
    rw [hcoeff k]
    simp only [ZMod.natCast_val, ZMod.cast_id', id]

end BgvSecurity
