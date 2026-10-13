import ExponentRuntime
import ExponentWalsh
import ExponentAutomorphism

/-!
# The PRF readout and the sampling LUTs for any parameter set

The sparse-LWR PRF of exponent modulus `Q' = 2 W'` (native width `W'`, stride `n / W'`) reads its
accumulated state `X^((n / W') z)` with the readout LUT of width `W'` whose table is
`X^(s_r τ(k))`, `τ(k) = 2 f(k) - 1`, `f(k) = ⌊p' k / W'⌋ + 1` (`SparseLwrPrfProfile::
canonical_readout`), in the readout layout of width `P = 2 p'` and stride `s_r = n / P`
(`PrfSampler::build`). Through the negacyclic wrap the output is
`(-1)^⌊z / W'⌋ X^(s_r τ(z mod W')) = X^(s_r readoutExpG z)`.

The sampling LUT of width `P` reads `X^(s_r e)` with the table `binTable g` (`g(j)` at the odd
entry `2 j + 1`, zero at the even ones): for `e = readoutExpG z` the sample is
`(-1)^⌊z / W'⌋ g(⌊p' (z mod W') / W'⌋)`. The binary sampler (`g ≡ h`, `2 h ≡ 1`) plus its offset
(the same LUT at `z = 0`) gives `σ = 1 - ⌊(z mod 2W') / W'⌋` modulo the decoding prime.

The LUT evaluation itself (`lut_gen`, `lut_gen_value`) is `lut_fourier_interpolation` at the root
`X^s` of the LUT's layout, for any width `2^log`, stride `s` and ring dimension `n`.

Nothing here fixes `n`, `W'`, `P`, `Q'` or the tables; the toy (`N = 8`, `W' = P = 4`) follows in
`GenToy`.
-/

namespace RefreshTopProof.Gen

open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

variable {n : Nat}

/-! ## The canonical readout -/

/-- The canonical sparse-LWR value `f(k) = ⌊p' k / W'⌋ + 1`. -/
def canonF (p' W' k : Nat) : Nat := p' * k / W' + 1

/-- The readout exponent `τ(k) = 2 f(k) - 1`. -/
def readoutVal (p' W' k : Nat) : Nat := 2 * canonF p' W' k - 1

theorem readoutVal_eq (p' W' k : Nat) : readoutVal p' W' k = 2 * (p' * k / W') + 1 := by
  unfold readoutVal canonF; omega

theorem readoutVal_odd (p' W' k : Nat) : readoutVal p' W' k % 2 = 1 := by
  rw [readoutVal_eq]; omega

/-- The bin of the readout: `(τ(k) - 1) / 2 = ⌊p' k / W'⌋`. -/
theorem readoutVal_bin (p' W' k : Nat) : readoutVal p' W' k / 2 = p' * k / W' := by
  rw [readoutVal_eq]; omega

theorem bin_lt {p' W' k : Nat} (hp : 0 < p') (hk : k < W') : p' * k / W' < p' := by
  rw [Nat.div_lt_iff_lt_mul (by omega)]
  exact Nat.mul_lt_mul_of_pos_left hk hp

/-- `τ(k) < P = 2 p'` on the native width. -/
theorem readoutVal_lt {p' W' k : Nat} (hp : 0 < p') (hk : k < W') : readoutVal p' W' k < 2 * p' := by
  have := bin_lt hp hk
  rw [readoutVal_eq]; omega

/-- The readout exponent of the state `X^((n / W') z)` in the readout layout of width `P`:
`τ(z mod W') + P ⌊z / W'⌋`. -/
def readoutExpG (P W' z : Nat) : Nat := readoutVal (P / 2) W' (z % W') + P * (z / W')

theorem readoutExpG_div {P W' z : Nat} (hP : 0 < P / 2) (hP2 : P % 2 = 0) (hW : 0 < W') :
    readoutExpG P W' z / P = z / W' ∧ readoutExpG P W' z % P = readoutVal (P / 2) W' (z % W') := by
  have hlt := readoutVal_lt (W' := W') (k := z % W') hP (Nat.mod_lt _ hW)
  have hP' : 2 * (P / 2) = P := by omega
  rw [hP'] at hlt
  unfold readoutExpG
  constructor
  · rw [Nat.add_mul_div_left _ _ (by omega), Nat.div_eq_of_lt hlt, Nat.zero_add]
  · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hlt]

/-- The offset: the encoding of one reads `τ(0) = 1`. -/
theorem readoutExpG_zero (P W' : Nat) : readoutExpG P W' 0 = 1 := by
  simp [readoutExpG, readoutVal_eq]

/-- The readout LUT's value through the negacyclic wrap: with `n = s_r P`,
`(-1)^⌊z / W'⌋ X^(s_r τ(z mod W')) = X^(s_r readoutExpG z)`. -/
theorem readout_value_gen {sr P W' : Nat} (hn : n = sr * P) (z : Nat) :
    ((((-1 : Int) ^ (z / W') : Int) : ErrorPoly n) * Xe (n := n) (sr * readoutVal (P / 2) W' (z % W'))) =
      Xe (n := n) (sr * readoutExpG P W' z) := by
  have hroot : AdjoinRoot.root (negacyclicModulus n Int) ^ (sr * P) = -1 := by
    rw [← hn]; exact Negacyclic.root_pow_n
  have hk : sr * readoutExpG P W' z = sr * readoutVal (P / 2) W' (z % W') + (sr * P) * (z / W') := by
    unfold readoutExpG; ring
  rw [hk]
  simp only [Xe, pow_add, pow_mul _ (sr * P), hroot]
  push_cast
  ring

/-! ## The sampling LUT -/

/-- A sampled value of a negacyclic LUT of width `P` with table `T`: `(-1)^⌊e / P⌋ T(e mod P)`. -/
noncomputable def sampledG (P : Nat) [NeZero P] (T : Fin P → ErrorPoly n) (e : Nat) : ErrorPoly n :=
  (((-1 : Int) ^ (e / P) : Int) : ErrorPoly n) * T ⟨e % P, Nat.mod_lt _ (NeZero.pos P)⟩

/-- The sampling table of a bin table `g` on the `P / 2` bins: `g(j)` at `2 j + 1`, zero at the
even entries (`PrfSampler::build`). -/
noncomputable def binTable (P : Nat) (g : Nat → Int) (k : Fin P) : ErrorPoly n :=
  if k.val % 2 = 1 then ((g (k.val / 2) : Int) : ErrorPoly n) else 0

/-- The sampling LUT on the PRF readout: `(-1)^⌊z / W'⌋ T(τ(z mod W'))`. -/
theorem sampled_readout {P W' : Nat} [NeZero P] (hP : 0 < P / 2) (hP2 : P % 2 = 0) (hW : 0 < W')
    (T : Fin P → ErrorPoly n) (z : Nat) :
    sampledG P T (readoutExpG P W' z) = (((-1 : Int) ^ (z / W') : Int) : ErrorPoly n) *
      T ⟨readoutVal (P / 2) W' (z % W'), by
        have := readoutVal_lt (W' := W') (k := z % W') hP (Nat.mod_lt _ hW); omega⟩ := by
  obtain ⟨hdiv, hmod⟩ := readoutExpG_div (z := z) hP hP2 hW
  unfold sampledG
  rw [hdiv]
  congr 2
  exact Fin.ext hmod

/-- **The sampler's member value.** The sampling LUT of the bin table `g` on the PRF readout of
`z` is `(-1)^⌊z / W'⌋ g(⌊p' (z mod W') / W'⌋)`. -/
theorem sampled_bin {P W' : Nat} [NeZero P] (hP : 0 < P / 2) (hP2 : P % 2 = 0) (hW : 0 < W')
    (g : Nat → Int) (z : Nat) :
    sampledG P (binTable (n := n) P g) (readoutExpG P W' z) =
      (((-1 : Int) ^ (z / W') * g ((P / 2) * (z % W') / W') : Int) : ErrorPoly n) := by
  rw [sampled_readout hP hP2 hW, binTable, if_pos (readoutVal_odd _ _ _), readoutVal_bin]
  push_cast
  ring

/-- The binary sampler's offset: the LUT at the readout of one is `h`. -/
theorem sampled_bin_one {P : Nat} [NeZero P] (hP : 2 ≤ P) (h : Int) :
    sampledG P (binTable (n := n) P fun _ ↦ h) 1 = ((h : Int) : ErrorPoly n) := by
  unfold sampledG binTable
  have h1 : 1 % P = 1 := Nat.mod_eq_of_lt (by omega)
  have h2 : 1 / P = 0 := Nat.div_eq_of_lt (by omega)
  simp [h1, h2]

/-- **The mask's σ.** A binary member (`g ≡ h`, `2 h ≡ 1 (mod q_c)`) plus the offset is
`σ = 1 - ⌊(z mod 2W') / W'⌋` modulo `q_c`. -/
theorem member_sigma_gen {P W' qc : Nat} [NeZero P] (hP : 0 < P / 2) (hP2 : P % 2 = 0)
    (hW : 0 < W') {h : Int} (h2 : ((2 * h : Int) : ZMod qc) = 1) (z : Nat) :
    reducePoly qc n (sampledG P (binTable P fun _ ↦ h) (readoutExpG P W' z) +
        sampledG P (binTable P fun _ ↦ h) 1) =
      reducePoly qc n (((1 - ((z : Int) % (2 * W')) / W' : Int)) : ErrorPoly n) := by
  rw [sampled_bin hP hP2 hW, sampled_bin_one (by omega), ← Int.cast_add, reducePoly_intCast,
    reducePoly_intCast, intCast_eq_algebraMap, intCast_eq_algebraMap]
  congr 1
  have hW' : ((z : Int) % (2 * W')) / W' = ((z / W' : Nat) : Int) % 2 := by
    have h1 : (z : Int) % (2 * (W' : Int)) = ((z % (2 * W') : Nat) : Int) := by push_cast; rfl
    have h2 : (z % (2 * W')) / W' = z / W' % 2 := by
      rw [Nat.mul_comm 2 W', Nat.mod_mul_right_div_self]
    rw [h1, ← Int.natCast_ediv, h2]
    push_cast; rfl
  rw [hW']
  rcases Nat.even_or_odd (z / W') with ⟨m, hm⟩ | ⟨m, hm⟩
  · rw [hm, show ((m + m : Nat) : Int) % 2 = 0 by omega, Even.neg_one_pow ⟨m, rfl⟩]
    push_cast at h2 ⊢
    linear_combination h2
  · rw [hm, show ((2 * m + 1 : Nat) : Int) % 2 = 1 by omega, Odd.neg_one_pow ⟨m, rfl⟩]
    push_cast
    ring

/-! ## The LUT evaluation at any layout -/

/-- A LUT of width `2^log` in the layout of stride `s` (`n = s · quotient · 2^log`) on the state
`X^(s z)`: `Σ_i σ_i(X^(s z)) C_i = X^(s (z - z mod 2^log)) T(z mod 2^log)`. -/
theorem lut_gen {q inverse s log quotient : Nat} (hn : n = s * (quotient * 2 ^ log)) (hnpos : 0 < n)
    (T : Fin (2 ^ log) → ErrorPoly n)
    (hinverse : (inverse : ExactPoly q n) * ((2 ^ log : Nat) : ExactPoly q n) = 1)
    (index : Fin (2 ^ log) → Nat) (hodd : ∀ i, Odd (index i)) (perm : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * (quotient * 2 ^ log)) = 1 + 2 * quotient * (perm i).val)
    (z : Nat) :
    ∑ i, reducePoly q n (substitution (index i) (hodd i) (Xe (n := n) (s * z))) *
        lutFourierCoefficient (quotient * 2 ^ log) (X q ^ s) (inverse : ExactPoly q n)
          (fun k ↦ reducePoly q n (T k)) (index i) =
      reducePoly q n (Xe (n := n) (s * (z - z % 2 ^ log)) * T ⟨z % 2 ^ log, Nat.mod_lt _ (by positivity)⟩) := by
  have hroot : (X q ^ s : ExactPoly q n) ^ (quotient * 2 ^ log) = -1 := by
    rw [← pow_mul, ← hn]; exact Negacyclic.root_pow_n
  have h := lut_fourier_interpolation (R := ExactPoly q n) log (quotient * 2 ^ log) quotient z
    (X q ^ s) (inverse : ExactPoly q n) (fun k ↦ reducePoly q n (T k)) rfl
    (by rw [hn] at hnpos; exact Nat.pos_of_mul_pos_left hnpos) hroot hinverse index perm hindex
  rw [map_mul, reducePoly_Xe, pow_mul, ← h]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  rw [substitution_root_pow, reducePoly_Xe, ← pow_mul,
    Nat.mul_comm (index i) z, ← Nat.mul_assoc]

/-- The full-period case (`quotient = 1`, `n = s 2^log`): the LUT on `X^(s z)` is
`(-1)^⌊z / 2^log⌋ T(z mod 2^log)`. -/
theorem lut_gen_value {q inverse s log : Nat} (hn : n = s * 2 ^ log) (hnpos : 0 < n)
    (T : Fin (2 ^ log) → ErrorPoly n)
    (hinverse : (inverse : ExactPoly q n) * ((2 ^ log : Nat) : ExactPoly q n) = 1)
    (index : Fin (2 ^ log) → Nat) (hodd : ∀ i, Odd (index i)) (perm : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * 2 ^ log) = 1 + 2 * 1 * (perm i).val) (z : Nat) :
    ∑ i, reducePoly q n (substitution (index i) (hodd i) (Xe (n := n) (s * z))) *
        lutFourierCoefficient (2 ^ log) (X q ^ s) (inverse : ExactPoly q n)
          (fun k ↦ reducePoly q n (T k)) (index i) =
      reducePoly q n ((((-1 : Int) ^ (z / 2 ^ log) : Int) : ErrorPoly n) *
        T ⟨z % 2 ^ log, Nat.mod_lt _ (by positivity)⟩) := by
  have h := lut_gen (q := q) (inverse := inverse) (s := s) (quotient := 1) (by rw [one_mul]; exact hn) hnpos
    T hinverse index hodd perm (by simpa using hindex) z
  simp only [one_mul] at h
  rw [h]
  congr 2
  have hroot : AdjoinRoot.root (negacyclicModulus n Int) ^ (s * 2 ^ log) = -1 := by
    rw [← hn]; exact Negacyclic.root_pow_n
  have hdm := Nat.div_add_mod z (2 ^ log)
  rw [show s * (z - z % 2 ^ log) = (s * 2 ^ log) * (z / 2 ^ log) by
    rw [show z - z % 2 ^ log = 2 ^ log * (z / 2 ^ log) by omega, Nat.mul_assoc]]
  rw [Xe, pow_mul, hroot]
  push_cast; rfl

end RefreshTopProof.Gen
