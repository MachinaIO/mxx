import ExponentReadoutSetup
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

namespace MxxExponentLut
open Mxx.Primitives
open scoped BigOperators

/-- Trace on the strided native root, indexed by each odd residue once. -/
def oddTrace {R : Type} [CommRing R] (width : Nat) (root : R) (exponent : Nat) : R :=
  ∑ j ∈ Finset.range width, root ^ ((2 * j + 1) * exponent)

private theorem oddTrace_double {R : Type} [CommRing R] (width exponent : Nat) (root : R)
    (hroot : root ^ (2 * width) = -1) :
    oddTrace (2 * width) root exponent =
      oddTrace width root exponent + oddTrace width root exponent * (-1) ^ exponent := by
  unfold oddTrace
  rw [show 2 * width = width + width by omega, Finset.sum_range_add]
  congr 1
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro j hj
  rw [show (2 * (width + j) + 1) * exponent =
    (2 * j + 1) * exponent + (2 * width) * exponent by ring, pow_add, pow_mul root (2 * width) exponent, hroot]

private theorem oddTrace_even {R : Type} [CommRing R] (width exponent : Nat) (root : R) :
    oddTrace width root (2 * exponent) = oddTrace width (root ^ 2) exponent := by
  unfold oddTrace
  apply Finset.sum_congr rfl
  intro j hj
  rw [← pow_mul]
  congr 1
  ring

/-- Exact odd-residue trace for every power-of-two width. Cancellation is
proved inside the ring; no field or integral-domain hypothesis is required. -/
theorem oddTrace_power_two {R : Type} [CommRing R] (log exponent : Nat) (root : R)
    (hroot : root ^ (2 ^ log) = -1) (hexponent : exponent < 2 * 2 ^ log) :
    oddTrace (2 ^ log) root exponent =
      if exponent = 0 then ((2 ^ log : Nat) : R)
      else if exponent = 2 ^ log then -((2 ^ log : Nat) : R) else 0 := by
  induction log generalizing root exponent with
  | zero =>
    have he : exponent = 0 ∨ exponent = 1 := by omega
    rcases he with rfl | rfl
    · simp [oddTrace]
    · simpa [oddTrace] using hroot
  | succ log ih =>
    have hwidth : 2 ^ (log + 1) = 2 * 2 ^ log := by rw [pow_succ]; ring
    rw [hwidth] at hroot hexponent ⊢
    rw [oddTrace_double _ _ _ hroot]
    rcases Nat.even_or_odd exponent with heven | hodd
    · obtain ⟨half, rfl⟩ := heven
      have htwice : half + half = 2 * half := by omega
      rw [htwice, oddTrace_even]
      have hroot' : (root ^ 2) ^ (2 ^ log) = -1 := by rw [← pow_mul]; exact hroot
      rw [ih half (root ^ 2) hroot' (by omega)]
      have heven : Even (2 * half) := by exact even_two_mul half
      rw [heven.neg_one_pow]
      split_ifs <;> simp_all <;> ring
    · rw [hodd.neg_one_pow]
      have hparity := Nat.odd_iff.mp hodd
      have hnotzero : exponent ≠ 0 := by omega
      have hnotwidth : exponent ≠ 2 * 2 ^ log := by omega
      simp [hnotzero, hnotwidth]

private theorem power_mod_period {R : Type} [CommRing R] (root : R) (period exponent : Nat)
    (hperiod : root ^ period = 1) : root ^ exponent = root ^ (exponent % period) := by
  conv_lhs => rw [← Nat.div_add_mod exponent period]
  rw [pow_add, pow_mul, hperiod]
  simp

private theorem oddTrace_mod {R : Type} [CommRing R] (width exponent : Nat) (root : R)
    (hroot : root ^ width = -1) :
    oddTrace width root exponent = oddTrace width root (exponent % (2 * width)) := by
  have hp : root ^ (2 * width) = 1 := by rw [Nat.mul_comm, pow_mul, hroot]; ring
  unfold oddTrace
  apply Finset.sum_congr rfl
  intro j hj
  rw [Nat.mul_comm (2 * j + 1) exponent, pow_mul,
    power_mod_period root (2 * width) exponent hp]
  rw [← pow_mul, Nat.mul_comm]

theorem oddTrace_difference {R : Type} [CommRing R] (log exponent index : Nat) (root : R)
    (hroot : root ^ (2 ^ log) = -1)
    (he : exponent < 2 * 2 ^ log) (hi : index < 2 ^ log) :
    oddTrace (2 ^ log) root (exponent + 2 * 2 ^ log - index) =
      if exponent = index then ((2 ^ log : Nat) : R)
      else if exponent = index + 2 ^ log then -((2 ^ log : Nat) : R) else 0 := by
  have hw : 0 < 2 ^ log := by positivity
  rw [oddTrace_mod _ _ _ hroot, oddTrace_power_two log _ root hroot (Nat.mod_lt _ (by omega))]
  have hmod : (exponent + 2 * 2 ^ log - index) % (2 * 2 ^ log) =
      if index ≤ exponent then exponent - index else exponent + 2 * 2 ^ log - index := by
    split_ifs with h
    · rw [show exponent + 2 * 2 ^ log - index = (exponent - index) + 2 * 2 ^ log by omega,
        Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    · rw [Nat.mod_eq_of_lt (by omega)]
  have hz : (exponent + 2 * 2 ^ log - index) % (2 * 2 ^ log) = 0 ↔ exponent = index := by
    rw [hmod]
    split_ifs <;> omega
  have hn : (exponent + 2 * 2 ^ log - index) % (2 * 2 ^ log) = 2 ^ log ↔
      exponent = index + 2 ^ log := by
    rw [hmod]
    split_ifs <;> omega
  simp only [hz, hn]

/-- The exact two-dimensional sum obtained by tracing `X^(stride*e) * T'`.
The negative target exponents are represented modulo the native period. -/
def tracedTarget {R : Type} [CommRing R] (width : Nat) (root : R)
    (table : Fin width → R) (exponent : Nat) : R :=
  ∑ k : Fin width, (2 * table k - 1) * oddTrace width root (exponent + 2 * width - k.val)

/-- Before normalization, the traced target selects the table coefficient
with the native antiperiodic sign, including the negative-monomial half. -/
theorem tracedTarget_value {R : Type} [CommRing R] (log exponent : Nat) (root : R)
    (table : Fin (2 ^ log) → R) (hroot : root ^ (2 ^ log) = -1)
    (he : exponent < 2 * 2 ^ log) :
    tracedTarget (2 ^ log) root table exponent =
      if h : exponent < 2 ^ log then
        ((2 ^ log : Nat) : R) * (2 * table ⟨exponent, h⟩ - 1)
      else -((2 ^ log : Nat) : R) * (2 * table ⟨exponent - 2 ^ log, by omega⟩ - 1) := by
  unfold tracedTarget
  simp_rw [oddTrace_difference log exponent _ root hroot he (Fin.isLt _)]
  split_ifs with h
  · rw [Finset.sum_eq_single (⟨exponent, h⟩ : Fin (2 ^ log))]
    · simp [mul_comm]
    · intro k hk hne
      have hk0 : exponent ≠ k.val := by intro hke; exact hne (Fin.ext hke.symm)
      have hk1 : exponent ≠ k.val + 2 ^ log := by omega
      simp [hk0, hk1]
    · simp
  · let chosen : Fin (2 ^ log) := ⟨exponent - 2 ^ log, by omega⟩
    rw [Finset.sum_eq_single chosen]
    · have hw : 0 < 2 ^ log := by positivity
      have h0 : exponent ≠ chosen.val := by dsimp [chosen]; omega
      have h1 : exponent = chosen.val + 2 ^ log := by dsimp [chosen]; omega
      simp only [if_neg h0, if_pos h1]
      dsimp [chosen]
      ring
    · intro k hk hne
      have hk0 : exponent ≠ k.val := by have := k.isLt; omega
      have hk1 : exponent ≠ k.val + 2 ^ log := by
        intro hke
        apply hne
        apply Fin.ext
        dsimp [chosen]
        omega
      simp [hk0, hk1]
    · simp

/-- The modular inverse supplied by setup normalizes the exact trace. The
separate `half` parameter permits composite-modulus rings without division. -/
theorem normalized_tracedTarget_value {R : Type} [CommRing R]
    (log exponent : Nat) (root inverse half : R) (table : Fin (2 ^ log) → R)
    (hroot : root ^ (2 ^ log) = -1) (he : exponent < 2 * 2 ^ log)
    (hinverse : inverse * ((2 * 2 ^ log : Nat) : R) = 1) (hhalf : 2 * half = 1) :
    inverse * tracedTarget (2 ^ log) root table exponent =
      if h : exponent < 2 ^ log then table ⟨exponent, h⟩ - half
      else half - table ⟨exponent - 2 ^ log, by omega⟩ := by
  have hi : inverse * (2 * ((2 ^ log : Nat) : R)) = 1 := by
    simpa only [Nat.cast_mul, Nat.cast_ofNat] using hinverse
  have hs : inverse * ((2 ^ log : Nat) : R) = half := by
    linear_combination half * hi - inverse * ((2 ^ log : Nat) : R) * hhalf
  rw [tracedTarget_value log exponent root table hroot he]
  split_ifs with h
  · calc
      _ = (inverse * ((2 ^ log : Nat) : R)) * (2 * table ⟨exponent, h⟩ - 1) := by ring
      _ = _ := by rw [hs]; linear_combination table ⟨exponent, h⟩ * hhalf
  · calc
      _ = -(inverse * ((2 ^ log : Nat) : R)) *
          (2 * table ⟨exponent - 2 ^ log, by omega⟩ - 1) := by ring
      _ = _ := by rw [hs]; linear_combination -table ⟨exponent - 2 ^ log, by omega⟩ * hhalf

/-- Concrete runtime indices are admitted through their residue permutation,
which can be proved from subset enumeration or checked on generated integers.
This hypothesis contains no trace or functional-output equation. -/
theorem trace_index_permutation {R : Type} [CommRing R] (width exponent : Nat) (root : R)
    (index : Fin width → Nat) (permutation : Fin width ≃ Fin width)
    (hindex : ∀ i, index i % (2 * width) = 2 * (permutation i).val + 1)
    (hroot : root ^ width = -1) :
    (∑ i, root ^ (index i * exponent)) = oddTrace width root exponent := by
  have hp : root ^ (2 * width) = 1 := by rw [Nat.mul_comm, pow_mul, hroot]; ring
  calc
    _ = ∑ i, root ^ ((2 * (permutation i).val + 1) * exponent) := by
      apply Finset.sum_congr rfl
      intro i hi
      rw [pow_mul, power_mod_period root (2 * width) (index i) hp, hindex i, ← pow_mul]
    _ = ∑ i : Fin width, root ^ ((2 * i.val + 1) * exponent) := permutation.sum_comp (fun i : Fin width => root ^ ((2 * i.val + 1) * exponent))
    _ = oddTrace width root exponent := by
      exact Fin.sum_univ_eq_sum_range (fun j => root ^ ((2 * j + 1) * exponent)) width

/-- The actual target declaration uses negative strided monomials modulo the
native period, preserving `ExpLayout::mono`'s negacyclic sign. -/
noncomputable def nativeTraceTarget {q n width stride : Nat} (table : Fin width → Int) :
    ExactPoly q n :=
  ∑ k, ((2 * table k - 1 : Int) : ExactPoly q n) *
    AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * (2 * width - k.val))

/-- Substituting the actual runtime automorphisms into the target product
produces the proved odd-residue trace sum, rather than an assumed trace rule. -/
theorem nativeTraceTarget_trace {q n width stride : Nat}
    (table : Fin width → Int) (exponent : Nat)
    (index : Fin width → Nat) (hodd : ∀ i, Odd (index i))
    (permutation : Fin width ≃ Fin width)
    (hindex : ∀ i, index i % (2 * width) = 2 * (permutation i).val + 1)
    (hdimension : n = stride * width) :
    (∑ i, readoutAutomorphismHom (index i) (hodd i)
      (AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * exponent) *
        nativeTraceTarget (q := q) (n := n) (stride := stride) table)) =
      tracedTarget width (AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ stride)
        (fun k => (table k : ExactPoly q n)) exponent := by
  let root : ExactPoly q n := AdjoinRoot.root (negacyclicModulus n (ZMod q))
  have hr : (root ^ stride) ^ width = -1 := by
    rw [← pow_mul, ← hdimension]
    exact Negacyclic.root_pow_n
  simp only [nativeTraceTarget, map_mul, map_sum, map_intCast, map_pow, Finset.mul_sum]
  rw [Finset.sum_comm]
  unfold tracedTarget
  apply Finset.sum_congr rfl
  intro k hk
  have he (i : Fin width) :
      readoutAutomorphismHom (index i) (hodd i) (AdjoinRoot.root (negacyclicModulus n (ZMod q))) =
        root ^ index i := by simp [readoutAutomorphismHom, root]
  simp_rw [he]
  have hterm (i : Fin width) :
      (root ^ index i) ^ (stride * exponent) *
          (((2 * table k - 1 : Int) : ExactPoly q n) *
            (root ^ index i) ^ (stride * (2 * width - k.val))) =
        ((2 * table k - 1 : Int) : ExactPoly q n) *
          (root ^ stride) ^ (index i * (exponent + 2 * width - k.val)) := by
    have hkbound := k.isLt
    have hsum : exponent + (2 * width - k.val) = exponent + 2 * width - k.val := by omega
    rw [← pow_mul, ← pow_mul, ← pow_mul]
    rw [mul_left_comm, ← pow_add]
    congr 2
    rw [← Nat.mul_add, ← Nat.mul_add, hsum]
    ring
  simp_rw [hterm]
  rw [← Finset.mul_sum, trace_index_permutation width _ (root ^ stride) index permutation hindex hr]
  simp [root]

/-- Functional native readout, including its negative-monomial half, from
the actual target expression and a concrete residue permutation of indices. -/
theorem normalized_nativeTraceTarget {q n stride log : Nat}
    (table : Fin (2 ^ log) → Int) (exponent : Nat)
    (index : Fin (2 ^ log) → Nat) (hodd : ∀ i, Odd (index i))
    (permutation : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * 2 ^ log) = 2 * (permutation i).val + 1)
    (hdimension : n = stride * 2 ^ log) (he : exponent < 2 * 2 ^ log)
    (inverse half : ExactPoly q n)
    (hinverse : inverse * ((2 * 2 ^ log : Nat) : ExactPoly q n) = 1)
    (hhalf : 2 * half = 1) :
    inverse * (∑ i, readoutAutomorphismHom (index i) (hodd i)
      (AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * exponent) *
        nativeTraceTarget (q := q) (n := n) (stride := stride) table)) =
      if h : exponent < 2 ^ log then (table ⟨exponent, h⟩ : ExactPoly q n) - half
      else half - (table ⟨exponent - 2 ^ log, by omega⟩ : ExactPoly q n) := by
  rw [nativeTraceTarget_trace table exponent index hodd permutation hindex hdimension]
  apply normalized_tracedTarget_value log exponent _ inverse half _ _ he hinverse hhalf
  rw [← pow_mul, ← hdimension]
  exact Negacyclic.root_pow_n

/-- The same functional value for the actual exported scalar automorphism
executions. The permutation premise can be checked on generated indices; the
conclusion follows from runtime equations and the proved trace computation. -/
theorem normalized_native_trace_runs {q n stride log : Nat} (hq : 1 < q) (hn : 0 < n)
    (table : Fin (2 ^ log) → Int) (exponent : Nat)
    (index : Fin (2 ^ log) → Nat) (hodd : ∀ i, Odd (index i))
    (permutation : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * 2 ^ log) = 2 * (permutation i).val + 1)
    (hdimension : n = stride * 2 ^ log) (he : exponent < 2 * 2 ^ log)
    (inverse half : ExactPoly q n)
    (hinverse : inverse * ((2 * 2 ^ log : Nat) : ExactPoly q n) = 1)
    (hhalf : 2 * half = 1)
    (raw : Fin (2 ^ log) → ExactMatrix q n 1 1)
    (hrun : ∀ i, MxxRuntime.ringAutomorphismRuns (index i : Int)
      (fun _ _ => AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * exponent) *
        nativeTraceTarget (q := q) (n := n) (stride := stride) table) (raw i)) :
    inverse * (∑ i, raw i 0 0) =
      if h : exponent < 2 ^ log then (table ⟨exponent, h⟩ : ExactPoly q n) - half
      else half - (table ⟨exponent - 2 ^ log, by omega⟩ : ExactPoly q n) := by
  have hraw (i : Fin (2 ^ log)) : raw i 0 0 =
      readoutAutomorphismHom (index i) (hodd i)
        (AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * exponent) *
          nativeTraceTarget (q := q) (n := n) (stride := stride) table) := by
    obtain ⟨_, _, _, hout⟩ := hrun i
    rw [hout]
    simp only [Int.toNat_natCast]
    rw [readoutAutomorphismHom_runtime hq hn (index i) (hodd i)]
  simp_rw [hraw]
  exact normalized_nativeTraceTarget table exponent index hodd permutation hindex
    hdimension he inverse half hinverse hhalf

end MxxExponentLut
