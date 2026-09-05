import ExponentTrace
import ExponentMonomial
import Mathlib.Data.Nat.ModEq

namespace MxxExponentLut
open Mxx.Primitives
open scoped BigOperators

private theorem power_mod {R : Type} [CommRing R] (root : R) (period exponent : Nat)
    (hperiod : root ^ period = 1) : root ^ exponent = root ^ (exponent % period) := by
  conv_lhs => rw [← Nat.div_add_mod exponent period]
  rw [pow_add, pow_mul, hperiod]
  simp

private def geometricTrace {R : Type} [CommRing R] (width : Nat) (root : R) (exponent : Nat) : R :=
  ∑ j ∈ Finset.range width, root ^ (j * exponent)

/-- Power-of-two Fourier orthogonality is proved by exact cancellation in
arbitrary commutative rings, including the composite CRT modulus used by Rust. -/
private theorem geometricTrace_power_two {R : Type} [CommRing R]
    (log exponent : Nat) (root : R) (hroot : root ^ (2 ^ log) = -1) :
    geometricTrace (2 ^ log) (root ^ 2) exponent =
      if exponent % (2 ^ log) = 0 then ((2 ^ log : Nat) : R) else 0 := by
  have hw : 0 < 2 ^ log := by positivity
  have hp : root ^ (2 * 2 ^ log) = 1 := by rw [Nat.mul_comm, pow_mul, hroot]; ring
  have hp2 : (root ^ 2) ^ (2 ^ log) = 1 := by rw [← pow_mul]; exact hp
  have hm : geometricTrace (2 ^ log) (root ^ 2) exponent =
      geometricTrace (2 ^ log) (root ^ 2) (exponent % (2 ^ log)) := by
    unfold geometricTrace
    apply Finset.sum_congr rfl
    intro j hj
    rw [Nat.mul_comm j exponent, pow_mul, power_mod (root ^ 2) (2 ^ log) exponent hp2,
      ← pow_mul, Nat.mul_comm]
  rw [hm]
  apply ((IsUnit.of_pow_eq_one hp (by omega)).pow (exponent % (2 ^ log))).mul_left_cancel
  have htrace : root ^ (exponent % (2 ^ log)) *
      geometricTrace (2 ^ log) (root ^ 2) (exponent % (2 ^ log)) =
      oddTrace (2 ^ log) root (exponent % (2 ^ log)) := by
    unfold geometricTrace oddTrace
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j hj
    rw [← pow_mul, ← pow_add]
    congr 1
    ring
  rw [htrace, oddTrace_power_two log _ root hroot (by have := Nat.mod_lt exponent hw; omega)]
  have hne : exponent % (2 ^ log) ≠ 2 ^ log := Nat.ne_of_lt (Nat.mod_lt _ hw)
  by_cases he : exponent % (2 ^ log) = 0 <;> simp [he, hne]

/-- On the strided root, the width-W subgroup permutes `1+(2M/W)j`, not
necessarily all odd native residues when the LUT width is smaller than M. -/
theorem lut_subgroup_trace {R : Type} [CommRing R]
    (log nativeWidth quotient exponent : Nat) (root : R)
    (hnative : nativeWidth = quotient * 2 ^ log) (hroot : root ^ nativeWidth = -1)
    (index : Fin (2 ^ log) → Nat) (permutation : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * nativeWidth) = 1 + 2 * quotient * (permutation i).val) :
    (∑ i, root ^ (index i * exponent)) =
      root ^ exponent * (if exponent % (2 ^ log) = 0 then ((2 ^ log : Nat) : R) else 0) := by
  have hp : root ^ (2 * nativeWidth) = 1 := by rw [Nat.mul_comm, pow_mul, hroot]; ring
  have hr : (root ^ quotient) ^ (2 ^ log) = -1 := by rw [← pow_mul, ← hnative]; exact hroot
  calc
    _ = ∑ i, root ^ ((1 + 2 * quotient * (permutation i).val) * exponent) := by
      apply Finset.sum_congr rfl
      intro i hi
      rw [pow_mul, power_mod root (2 * nativeWidth) (index i) hp, hindex i, ← pow_mul]
    _ = ∑ i : Fin (2 ^ log), root ^ ((1 + 2 * quotient * i.val) * exponent) :=
      permutation.sum_comp (fun i => root ^ ((1 + 2 * quotient * i.val) * exponent))
    _ = root ^ exponent * geometricTrace (2 ^ log) ((root ^ quotient) ^ 2) exponent := by
      unfold geometricTrace
      rw [← Fin.sum_univ_eq_sum_range, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i hi
      rw [← pow_mul, ← pow_mul, ← pow_add]
      congr 1
      ring
    _ = _ := by rw [geometricTrace_power_two log exponent (root ^ quotient) hr]

private theorem difference_divisible {width nativeWidth exponent index : Nat}
    (hnpos : 0 < nativeWidth) (hi : index < width) (hn : width ∣ nativeWidth) :
    (exponent + 2 * nativeWidth - index) % width = 0 ↔ exponent % width = index := by
  have hwn : width ≤ nativeWidth := Nat.le_of_dvd hnpos hn
  have hperiod : 2 * nativeWidth ≡ 0 [MOD width] :=
    Nat.modEq_zero_iff_dvd.mpr (dvd_mul_of_dvd_right hn 2)
  have htotal : (exponent + 2 * nativeWidth - index) + index ≡ exponent [MOD width] := by
    rw [Nat.sub_add_cancel (by omega)]
    simpa using (show exponent ≡ exponent [MOD width] from Nat.ModEq.rfl).add hperiod
  have hequiv : (exponent + 2 * nativeWidth - index) ≡ 0 [MOD width] ↔ exponent ≡ index [MOD width] := by
    constructor
    · intro h
      exact htotal.symm.trans (by simpa using h.add_right index)
    · intro h
      exact Nat.ModEq.add_right_cancel' index (by simpa using htotal.trans h)
  simpa only [Nat.ModEq, Nat.zero_mod, Nat.mod_eq_of_lt hi] using hequiv

/-- Exactly `encoding.rs::lut_coefficient`: W inverse times the sum of
`mono(L(k)) * sigma(mono(-k))`, retaining the declared table order. -/
def lutFourierCoefficient {R : Type} [CommRing R] {width : Nat}
    (nativeWidth : Nat) (root inverse : R) (table : Fin width → R) (index : Nat) : R :=
  inverse * ∑ k, table k * root ^ (index * (2 * nativeWidth - k.val))

private theorem lut_fourier_double_sum {R : Type} [CommRing R] {width : Nat}
    (nativeWidth exponent : Nat) (root inverse : R) (table : Fin width → R)
    (index : Fin width → Nat) (hwidth : width ≤ nativeWidth) :
    (∑ i, root ^ (index i * exponent) * lutFourierCoefficient nativeWidth root inverse table (index i)) =
      inverse * ∑ k, table k * (∑ i, root ^ (index i * (exponent + 2 * nativeWidth - k.val))) := by
  unfold lutFourierCoefficient
  simp_rw [← mul_assoc, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k hk
  apply Finset.sum_congr rfl
  intro i hi
  have hsum : index i * exponent + index i * (2 * nativeWidth - k.val) =
      index i * (exponent + 2 * nativeWidth - k.val) := by
    rw [← Nat.mul_add]
    have hkbound := k.isLt
    congr 1
    omega
  rw [← hsum, pow_add]
  ring

/-- Actual LUT evaluation on the full native exponent domain. Outside its
width-W lookup domain the implementation retains a monomial factor; this
faithful extension still preserves signed-monomial plaintexts. -/
theorem lut_fourier_interpolation {R : Type} [CommRing R]
    (log nativeWidth quotient exponent : Nat) (root inverse : R)
    (table : Fin (2 ^ log) → R) (hnative : nativeWidth = quotient * 2 ^ log)
    (hnpos : 0 < nativeWidth) (hroot : root ^ nativeWidth = -1)
    (hinverse : inverse * ((2 ^ log : Nat) : R) = 1)
    (index : Fin (2 ^ log) → Nat) (permutation : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * nativeWidth) = 1 + 2 * quotient * (permutation i).val) :
    (∑ i, root ^ (index i * exponent) * lutFourierCoefficient nativeWidth root inverse table (index i)) =
      root ^ (exponent - exponent % (2 ^ log)) * table ⟨exponent % (2 ^ log), Nat.mod_lt _ (by positivity)⟩ := by
  have hw : 0 < 2 ^ log := by positivity
  have hdvd : 2 ^ log ∣ nativeWidth := by rw [hnative]; exact dvd_mul_left _ _
  have hwidth : 2 ^ log ≤ nativeWidth := Nat.le_of_dvd hnpos hdvd
  rw [lut_fourier_double_sum nativeWidth exponent root inverse table index hwidth]
  simp_rw [lut_subgroup_trace log nativeWidth quotient _ root hnative hroot index permutation hindex]
  let chosen : Fin (2 ^ log) := ⟨exponent % (2 ^ log), Nat.mod_lt _ hw⟩
  rw [Finset.sum_eq_single chosen]
  · have hselected : (exponent + 2 * nativeWidth - chosen.val) % (2 ^ log) = 0 :=
      (difference_divisible hnpos chosen.isLt hdvd).mpr rfl
    rw [if_pos hselected]
    have hexp : exponent + 2 * nativeWidth - chosen.val =
        (exponent - chosen.val) + 2 * nativeWidth := by
      have := Nat.mod_le exponent (2 ^ log)
      dsimp [chosen]
      omega
    have hp : root ^ (2 * nativeWidth) = 1 := by rw [Nat.mul_comm, pow_mul, hroot]; ring
    rw [hexp, pow_add, hp, mul_one]
    calc
      _ = (inverse * ((2 ^ log : Nat) : R)) * (root ^ (exponent - chosen.val) * table chosen) := by ring
      _ = _ := by rw [hinverse, one_mul]
  · intro k hk hne
    have hnot : (exponent + 2 * nativeWidth - k.val) % (2 ^ log) ≠ 0 := by
      intro h
      apply hne
      apply Fin.ext
      exact ((difference_divisible hnpos k.isLt hdvd).mp h).symm
    rw [if_neg hnot, mul_zero, mul_zero]
  · simp

/-- In the declared lookup domain the monomial factor is one, giving the
actual table entry rather than an assumed interpolation axiom. -/
theorem lut_fourier_table_value {R : Type} [CommRing R]
    (log nativeWidth quotient : Nat) (root inverse : R)
    (table : Fin (2 ^ log) → R) (input : Fin (2 ^ log))
    (hnative : nativeWidth = quotient * 2 ^ log) (hnpos : 0 < nativeWidth)
    (hroot : root ^ nativeWidth = -1) (hinverse : inverse * ((2 ^ log : Nat) : R) = 1)
    (index : Fin (2 ^ log) → Nat) (permutation : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * nativeWidth) = 1 + 2 * quotient * (permutation i).val) :
    (∑ i, root ^ (index i * input.val) * lutFourierCoefficient nativeWidth root inverse table (index i)) =
      table input := by
  rw [lut_fourier_interpolation log nativeWidth quotient input.val root inverse table
    hnative hnpos hroot hinverse index permutation hindex]
  simp [Nat.mod_eq_of_lt input.isLt]

/-- Ring expression emitted for one Rust LUT coefficient: the table output
is not automorphed, while the negative input monomial is automorphed. -/
noncomputable def nativeLutFourierCoefficient {q n width stride : Nat}
    (nativeWidth : Nat) (inverse : ExactPoly q n) (table : Fin width → ExactPoly q n)
    (index : Nat) (hodd : Odd index) : ExactPoly q n :=
  inverse * ∑ k, table k * readoutAutomorphismHom index hodd
    (AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * (2 * nativeWidth - k.val)))

private theorem nativeCoefficient_eq {q n width stride : Nat}
    (nativeWidth : Nat) (inverse : ExactPoly q n) (table : Fin width → ExactPoly q n)
    (index : Nat) (hodd : Odd index) :
    nativeLutFourierCoefficient (stride := stride) nativeWidth inverse table index hodd =
      lutFourierCoefficient nativeWidth
        (AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ stride) inverse table index := by
  unfold nativeLutFourierCoefficient lutFourierCoefficient
  congr 1
  apply Finset.sum_congr rfl
  intro k hk
  congr 1
  rw [map_pow]
  have hr : readoutAutomorphismHom index hodd (AdjoinRoot.root (negacyclicModulus n (ZMod q))) =
      AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ index := by simp [readoutAutomorphismHom]
  rw [hr, ← pow_mul, ← pow_mul]
  congr 1
  ring

/-- Actual native Fourier expressions interpolate with their full-domain
monomial extension, retaining the operation width independently of the native width. -/
theorem native_lut_fourier_interpolation {q n stride log : Nat}
    (nativeWidth quotient exponent : Nat) (inverse : ExactPoly q n)
    (table : Fin (2 ^ log) → ExactPoly q n)
    (hdimension : n = stride * nativeWidth) (hnative : nativeWidth = quotient * 2 ^ log)
    (hnpos : 0 < nativeWidth) (hinverse : inverse * ((2 ^ log : Nat) : ExactPoly q n) = 1)
    (index : Fin (2 ^ log) → Nat) (hodd : ∀ i, Odd (index i))
    (permutation : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * nativeWidth) = 1 + 2 * quotient * (permutation i).val) :
    (∑ i, readoutAutomorphismHom (index i) (hodd i)
      (AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * exponent)) *
      nativeLutFourierCoefficient (stride := stride) nativeWidth inverse table (index i) (hodd i)) =
    (AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ stride) ^ (exponent - exponent % (2 ^ log)) *
      table ⟨exponent % (2 ^ log), Nat.mod_lt _ (by positivity)⟩ := by
  have hr (i : Fin (2 ^ log)) :
      readoutAutomorphismHom (index i) (hodd i)
        (AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * exponent)) =
      (AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ stride) ^ (index i * exponent) := by
    rw [map_pow]
    have hroot : readoutAutomorphismHom (index i) (hodd i)
        (AdjoinRoot.root (negacyclicModulus n (ZMod q))) =
        AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ index i := by simp [readoutAutomorphismHom]
    rw [hroot, ← pow_mul, ← pow_mul]
    congr 1
    ring
  simp_rw [hr, nativeCoefficient_eq]
  apply lut_fourier_interpolation log nativeWidth quotient exponent _ inverse table hnative hnpos _
    hinverse index permutation hindex
  rw [← pow_mul, ← hdimension]
  exact Negacyclic.root_pow_n

/-- Exponent-valued output tables preserve the exact strided input contract
on the full native domain, enabling subsequent nonlinear Program operations. -/
theorem native_lut_exponent_closure {q n stride log : Nat}
    (nativeWidth quotient exponent : Nat) (inverse : ExactPoly q n)
    (table : Fin (2 ^ log) → Fin (2 * nativeWidth))
    (hdimension : n = stride * nativeWidth) (hnative : nativeWidth = quotient * 2 ^ log)
    (hnpos : 0 < nativeWidth) (hinverse : inverse * ((2 ^ log : Nat) : ExactPoly q n) = 1)
    (index : Fin (2 ^ log) → Nat) (hodd : ∀ i, Odd (index i))
    (permutation : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * nativeWidth) = 1 + 2 * quotient * (permutation i).val) :
    ∃ output : Fin (2 * nativeWidth),
      (∑ i, readoutAutomorphismHom (index i) (hodd i)
        (AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * exponent)) *
        nativeLutFourierCoefficient (stride := stride) nativeWidth inverse
          (fun k => AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * (table k).val))
          (index i) (hodd i)) =
      AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * output.val) := by
  let selected : Fin (2 ^ log) := ⟨exponent % (2 ^ log), Nat.mod_lt _ (by positivity)⟩
  let total := exponent - exponent % (2 ^ log) + (table selected).val
  refine ⟨⟨total % (2 * nativeWidth), Nat.mod_lt _ (by omega)⟩, ?_⟩
  rw [native_lut_fourier_interpolation nativeWidth quotient exponent inverse _ hdimension hnative
    hnpos hinverse index hodd permutation hindex]
  let root : ExactPoly q n := AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ stride
  have hp : root ^ (2 * nativeWidth) = 1 := by
    have hr : root ^ nativeWidth = -1 := by
      dsimp [root]
      rw [← pow_mul, ← hdimension]
      exact Negacyclic.root_pow_n
    rw [Nat.mul_comm, pow_mul, hr]
    ring
  change root ^ (exponent - exponent % (2 ^ log)) *
    AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * (table selected).val) = _
  rw [pow_mul]
  change root ^ (exponent - exponent % (2 ^ log)) * root ^ (table selected).val = _
  rw [← pow_add]
  change root ^ total = _
  rw [power_mod root (2 * nativeWidth) total hp]
  exact (pow_mul _ _ _).symm

/-- The derived strided exponent witness supplies the gate-level signed
monomial contract with its exponent strictly below the full negacyclic period. -/
theorem strided_exponent_signedMonomial {q n stride nativeWidth : Nat}
    (hn : 0 < n) (hdimension : n = stride * nativeWidth) {value : ExactPoly q n}
    (hvalue : ∃ output : Fin (2 * nativeWidth), value =
      AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * output.val)) :
    SignedMonomial value := by
  obtain ⟨output, hout⟩ := hvalue
  have hs : 0 < stride := by nlinarith
  have hbound : stride * output.val < 2 * n := by
    have h := Nat.mul_lt_mul_of_pos_left output.isLt hs
    nlinarith
  exact ⟨⟨stride * output.val, hbound⟩, hout⟩

/-- Subsequent private/public/LUT gates obtain an actual unit-L1 integer
message witness from Fourier interpolation, without assuming output closure. -/
theorem native_lut_integer_lift {q n stride log : Nat} (hn : 0 < n)
    (nativeWidth quotient exponent : Nat) (inverse : ExactPoly q n)
    (table : Fin (2 ^ log) → Fin (2 * nativeWidth))
    (hdimension : n = stride * nativeWidth) (hnative : nativeWidth = quotient * 2 ^ log)
    (hnpos : 0 < nativeWidth) (hinverse : inverse * ((2 ^ log : Nat) : ExactPoly q n) = 1)
    (index : Fin (2 ^ log) → Nat) (hodd : ∀ i, Odd (index i))
    (permutation : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * nativeWidth) = 1 + 2 * quotient * (permutation i).val) :
    ∃ lift : ErrorPoly n,
      reducePoly q n lift =
        (∑ i, readoutAutomorphismHom (index i) (hodd i)
          (AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * exponent)) *
          nativeLutFourierCoefficient (stride := stride) nativeWidth inverse
            (fun k => AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * (table k).val))
            (index i) (hodd i)) ∧ polynomialL1 lift ≤ 1 := by
  apply signedMonomial_integer_lift hn
  apply strided_exponent_signedMonomial hn hdimension
  exact native_lut_exponent_closure nativeWidth quotient exponent inverse table hdimension hnative
    hnpos hinverse index hodd permutation hindex

/-- Each negative-monomial automorphism in the actual coefficient producer
can be consumed directly, connecting its generated matrix sum to the Fourier
expression above. This premise is execution, not an interpolation equation. -/
theorem native_lut_coefficient_runs {q n stride width : Nat} (hq : 1 < q) (hn : 0 < n)
    (nativeWidth index : Nat) (hodd : Odd index) (inverse : ExactPoly q n)
    (table : Fin width → ExactPoly q n) (rotated : Fin width → ExactMatrix q n 1 1)
    (hrun : ∀ k, MxxRuntime.ringAutomorphismRuns (index : Int)
      (fun _ _ => AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^
        (stride * (2 * nativeWidth - k.val))) (rotated k)) :
    inverse * (∑ k, table k * rotated k 0 0) =
      nativeLutFourierCoefficient (stride := stride) nativeWidth inverse table index hodd := by
  unfold nativeLutFourierCoefficient
  congr 1
  apply Finset.sum_congr rfl
  intro k hk
  congr 1
  obtain ⟨_, _, _, hout⟩ := hrun k
  rw [hout]
  simp only [Int.toNat_natCast]
  rw [readoutAutomorphismHom_runtime hq hn index hodd]

end MxxExponentLut
