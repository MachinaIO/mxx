import Mathlib
import ExponentEncoding
import RuntimeModulusSwitch
import RuntimeSubGaussian
import PrimitivesCrtRadix
import ExponentH2

/-!
# The refresh to the top level

Program- and parameter-independent lemmas of the refresh to the top level
(`crate::refresh::top`): one decoding chain per prime `q` of the top modulus
`Q₀` re-encrypts a level-`L` encoding of `x` as the level-0 encoding
`y = σ A' - Δ₀ r x G^st₀ + e'` under the new mask `σ = (σ₀, 1)`.

* `target_weight_congr`: the target action. On chain `q`, with decoding modulus
  `Q = Δ_L M` (`M` the plaintext modulus, `q ∣ M`) and `Q = q Q_q`, a weight
  `R` with `Δ_L R ≡ Q_q T (mod q)` and `M/q ∣ R` satisfies
  `Δ_L (R + M K) ≡ Q_q T (mod Q)` for every carry `K` of the gadget
  decomposition.
* `mask_projection_congr`: the mask child's payload `(M_c + 1) σ₀` acts as `σ₀`
  after the scale `Q_q` when `q ∣ M_c`.
* `crt_recombine`: the residues of every chain, each lifted and weighted by
  any CRT idempotent of its prime, recombine to the value modulo `Q₀`.
-/

namespace Mxx.ExponentLut.Refresh

/-! ## The target action -/

/-- The target weight `R` of prime `q` maps the level's payload scale to the reference format:
`Δ (R + M K) ≡ Q_q T (mod Q)` for `Q = Δ M = q Q_q`, when `Δ R ≡ Q_q T (mod q)`, `M / q ∣ R`
(`M = q M'`) and `q` is coprime to `Q_q` (the primes are distinct). -/
theorem target_weight_congr {Q Δ M M' q Qq : ℤ} (hQ : Q = Δ * M) (hM : M = q * M')
    (hQq : Q = q * Qq) (hq0 : q ≠ 0) (hcop : IsCoprime q Qq) {R T : ℤ} (hq : Δ * R ≡ Qq * T [ZMOD q])
    (hrest : M' ∣ R) (K : ℤ) : Δ * (R + M * K) ≡ Qq * T [ZMOD Q] := by
  -- `Δ M K = Q K` vanishes modulo `Q`.
  have hcarry : Δ * (R + M * K) ≡ Δ * R [ZMOD Q] := by
    have : Δ * (R + M * K) = Δ * R + Q * K := by rw [hQ]; ring
    rw [this]
    exact Int.modEq_iff_dvd.mpr ⟨-K, by ring⟩
  refine hcarry.trans ?_
  rw [hQq]
  -- Both sides agree modulo `q` and modulo `Q_q`, which are coprime.
  have hQq' : Δ * R ≡ Qq * T [ZMOD Qq] := by
    obtain ⟨r, hr⟩ := hrest
    have hΔ : Δ * M' = Qq := by
      have h : q * (Δ * M') = q * Qq := by rw [← hQq, hQ, hM]; ring
      exact mul_left_cancel₀ hq0 h
    rw [hr, ← mul_assoc, hΔ]
    exact Int.modEq_iff_dvd.mpr ⟨T - r, by ring⟩
  exact (Int.modEq_and_modEq_iff_modEq_mul (Int.isCoprime_iff_gcd_eq_one.mp hcop)).mp ⟨hq, hQq'⟩

/-! ## The mask projection -/

/-- In a ring where `Q = q Q_q` vanishes, the mask child's payload `(M + 1) σ` with `q ∣ M` acts as
`σ` after the scale `Q_q`. -/
theorem mask_projection {R : Type*} [CommRing R] {Q q Qq M : ℕ} (hQ : Q = q * Qq) (hM : q ∣ M)
    (hzero : (Q : R) = 0) (σ : R) : (Qq : R) * (((M : R) + 1) * σ) = (Qq : R) * σ := by
  obtain ⟨k, rfl⟩ := hM
  have : (Qq : R) * ((q * k : ℕ) : R) = 0 := by
    rw [show (Qq : R) * ((q * k : ℕ) : R) = (Q : R) * k by rw [hQ]; push_cast; ring, hzero,
      zero_mul]
  linear_combination σ * this

/-! ## The recombination of the chains -/

open scoped Function in
/-- The residues of every chain recombine to the value: for pairwise coprime moduli `m`, any
integers `z i` representing the residues `y mod m i`, weighted by the CRT idempotents `w i`
(`w i ≡ 1` modulo `m i` and `≡ 0` modulo the others), sum to `y`. -/
theorem crt_recombine {ι : Type*} [Fintype ι] [DecidableEq ι] (m : ι → ℕ)
    (hco : Pairwise (Nat.Coprime on m)) (y : ZMod (∏ i, m i)) (z : ι → ℤ)
    (hz : ∀ i, (z i : ZMod (m i)) = ZMod.prodEquivPi m hco y i) (w : ι → ZMod (∏ i, m i))
    (hw : ∀ i, ZMod.prodEquivPi m hco (w i) = Pi.single i 1) :
    ∑ i, w i * (z i : ZMod (∏ i, m i)) = y := by
  apply (ZMod.prodEquivPi m hco).injective
  funext j
  simp only [map_sum, map_mul, map_intCast, Finset.sum_apply, Pi.mul_apply, hw, Pi.intCast_apply]
  rw [Finset.sum_eq_single j]
  · simp [hz]
  · intro i _ hij
    simp [Pi.single_apply, Ne.symm hij]
  · simp

open scoped Function in
/-- An integer `c` that is `1` modulo `m i` and `0` modulo the other moduli is the CRT idempotent of
`i`. -/
theorem crt_idempotent {ι : Type*} [Fintype ι] [DecidableEq ι] (m : ι → ℕ)
    (hco : Pairwise (Nat.Coprime on m)) (i : ι) (c : ℤ)
    (hc : ∀ j, (c : ZMod (m j)) = if j = i then 1 else 0) :
    ZMod.prodEquivPi m hco (c : ZMod (∏ i, m i)) = Pi.single i 1 := by
  funext j
  rw [map_intCast, Pi.intCast_apply, hc j]
  by_cases h : j = i
  · subst h; simp
  · simp [h]

/-! ## The decoded value

Rows are `1 × m` matrices over a commutative ring `R` (the decoding ring `R_Q`), keys pairs of
rows (`K 0`, `K 1`) under the mask `(s₀, 1)`: a row `c` with key `K`, target `t` and error `e`
satisfies `c = s₀ K 0 + K 1 - t + e`. -/

section Decoded

variable {R : Type*} [CommRing R]

/-- `s₀ K 0 + K 1`: the mask part of a row with key `K`. -/
def maskPart {a b : Type*} [Fintype a] (s₀ : R) (K : Fin 2 → Matrix a b R) : Matrix a b R :=
  s₀ • K 0 + K 1

theorem maskPart_mul {a b c : Type*} [Fintype a] [Fintype b] (s₀ : R) (K : Fin 2 → Matrix a b R)
    (D : Matrix b c R) : maskPart s₀ K * D = maskPart s₀ (fun k ↦ K k * D) := by
  simp [maskPart, Matrix.add_mul, Matrix.smul_mul]

theorem maskPart_add {a b : Type*} [Fintype a] (s₀ : R) (K L : Fin 2 → Matrix a b R) :
    maskPart s₀ K + maskPart s₀ L = maskPart s₀ (K + L) := by
  simp [maskPart, smul_add]; abel

theorem maskPart_sub {a b : Type*} [Fintype a] (s₀ : R) (K L : Fin 2 → Matrix a b R) :
    maskPart s₀ K - maskPart s₀ L = maskPart s₀ (K - L) := by
  simp [maskPart, smul_sub]; abel

/-- The refresh combination of one prime and its unmasking (`crate::rns_bgg::refresh` and
`crate::decoder::unmask`). The rows are the mask carrier `g` (target `β G`, `β` the mask child's
payload), the public one encoding's second block `c₁` (target `G`), the target component `cₓ`
(target `P`), and the fresh term `w = [f_k z]_k` (target `Q_q e'`, the fresh carriers' action); the
actions satisfy `G X₀ = Q_q A'₀`, `G X₁ = Q_q A'₁` and `P Y = Q_q Tₓ`; the hint is `b = s B + e_b`
and the preimage `B Kpre = K_d + K_sm`, with `K_d` the combination's key; the smudge carrier is
`c_sm = s K_sm - (M + H) + e_sm`. Then the unmasked value `ζ = b Kpre - d - c_sm - H` is
`Q_q y + M + N` with `y = β A'₀ + A'₁ - Tₓ + e'` and `N = e_b Kpre - E - e_sm`,
`E = e_g X₀ + e₁ X₁ - eₓ Y + e_w` the combination's error. -/
theorem decoded_value {m w columns hint : ℕ}
    (s₀ β Qq : R) (G : Matrix (Fin 1) (Fin m) R) (P : Matrix (Fin 1) (Fin w) R)
    (Kg K₁ : Fin 2 → Matrix (Fin 1) (Fin m) R) (Kx : Fin 2 → Matrix (Fin 1) (Fin w) R)
    (Kw : Fin 2 → Matrix (Fin 1) (Fin columns) R)
    (eg e₁ : Matrix (Fin 1) (Fin m) R) (ex : Matrix (Fin 1) (Fin w) R)
    (ew : Matrix (Fin 1) (Fin columns) R)
    (X₀ X₁ : Matrix (Fin m) (Fin columns) R) (Y : Matrix (Fin w) (Fin columns) R)
    (A₀ A₁ Tx e' : Matrix (Fin 1) (Fin columns) R)
    (hX₀ : G * X₀ = Qq • A₀) (hX₁ : G * X₁ = Qq • A₁) (hY : P * Y = Qq • Tx)
    (B : Fin 2 → Matrix (Fin 1) (Fin hint) R) (eb : Matrix (Fin 1) (Fin hint) R)
    (Kpre : Matrix (Fin hint) (Fin columns) R) (Ksm : Fin 2 → Matrix (Fin 1) (Fin columns) R)
    (M H esm : Matrix (Fin 1) (Fin columns) R)
    (hpre : ∀ k, B k * Kpre = (Kg k * X₀ + K₁ k * X₁ - Kx k * Y + Kw k) + Ksm k) :
    let g := maskPart s₀ Kg - β • G + eg
    let c₁ := maskPart s₀ K₁ - G + e₁
    let cₓ := maskPart s₀ Kx - P + ex
    let fw := maskPart s₀ Kw - Qq • e' + ew
    let d := g * X₀ + c₁ * X₁ - cₓ * Y + fw
    let b := maskPart s₀ B + eb
    let csm := maskPart s₀ Ksm - (M + H) + esm
    b * Kpre - d - csm - H =
      Qq • (β • A₀ + A₁ - Tx + e') + M + (eb * Kpre - (eg * X₀ + e₁ * X₁ - ex * Y + ew) - esm) := by
  intro g c₁ cₓ fw d b csm
  have hB : maskPart s₀ B * Kpre = maskPart s₀ (fun k ↦ B k * Kpre) := maskPart_mul _ _ _
  simp only [b, d, g, c₁, cₓ, fw, csm, Matrix.add_mul, Matrix.sub_mul, Matrix.smul_mul, hB,
    maskPart_mul, hX₀, hX₁, hY]
  simp only [maskPart, hpre, smul_add, smul_sub, smul_smul]
  module

end Decoded

/-! ## The rounding of one decoder -/

/-- The integer rounding behind the decoder's modulus switch: if `Q_q r` is within `Q_q / 2` of
`Q_q a + ν` and `2 |ν| < Q_q` (`Q_q` odd), then `r = a`. -/
theorem round_eq {Qq r a ν : ℤ} (hQq : 0 < Qq) (hodd : Qq % 2 = 1)
    (hlo : -(Qq / 2) ≤ Qq * r - (Qq * a + ν)) (hhi : Qq * r - (Qq * a + ν) ≤ Qq / 2)
    (hν : 2 * |ν| < Qq) : r = a := by
  have hν' := abs_lt.mp (show |2 * ν| < Qq by rw [abs_mul]; simpa using hν)
  by_contra hne
  rcases lt_or_gt_of_ne hne with h | h
  · have : Qq * r ≤ Qq * a - Qq := by nlinarith
    omega
  · have : Qq * a + Qq ≤ Qq * r := by nlinarith
    omega

open MxxRuntime Mxx.Primitives in
/-- The decoder's modulus switch from `Q = Q_q q` to `q` returns `ŷ mod q` for a coefficient
`Q_q ŷ + ν` (modulo `Q`) with `2 |ν| < Q_q`. -/
theorem modulusSwitch_residue {Q q Qq n rows columns : ℕ} (hn : 0 < n) (hQq : 0 < Qq)
    (hQ : Q = Qq * q) (input : ExactMatrix Q n rows columns) (output : ExactMatrix q n rows columns)
    (runs : modulusSwitchRuns input output) (row : Fin rows) (column : Fin columns) (i : Fin n)
    (yhat ν : ℤ)
    (hrel : Int.ofNat ((input row column).coeff i).val ≡ (Qq : ℤ) * yhat + ν [ZMOD (Q : ℤ)])
    (hν : 2 * |ν| < Qq) : (output row column).coeff i = (yhat : ZMod q) := by
  obtain ⟨r, hout, hlo, hhi⟩ :=
    modulusSwitchRuns_rounding_error hn hQq hQ input output runs row column i
  have hodd : (Qq : ℤ) % 2 = 1 := by
    have ho := runs.2.2.1
    have hp := runs.2.2.2.1
    rw [hQ, Nat.mul_mod, hp, Nat.mul_one] at ho
    omega
  obtain ⟨t, ht⟩ := (Int.modEq_iff_dvd.mp hrel.symm)
  set v := Int.ofNat ((input row column).coeff i).val
  have hv : v = (Qq : ℤ) * (yhat + q * t) + ν := by
    have : (Q : ℤ) = Qq * q := by exact_mod_cast hQ
    rw [this] at ht
    linarith
  rw [hv] at hlo hhi
  have hr := round_eq (by exact_mod_cast hQq) hodd hlo hhi hν
  rw [hout, hr]
  push_cast
  simp

/-! ## The recombined output -/

open scoped Function in
open MxxRuntime Mxx.Primitives in
/-- The output of the refresh to the top level, coefficient by coefficient. Chain `c` (output
prime `m c`, decoding modulus `Q c = Q_q c · m c`) switches its decoded row to its prime; its
decoded coefficient is `Q_q c ŷ + ν` (modulo `Q c`) with `ŷ` representing the reference value
`y` modulo `m c` (the chain's target, mask and `A'` agree with the reference modulo its prime:
`target_weight_congr`, `mask_projection`) and `2 |ν| < Q_q c`. Then the centered residues of
every chain, weighted by any CRT idempotents `w`, recombine to the coefficient of `y`. -/
theorem top_refresh_coefficient {ι : Type*} [Fintype ι] [DecidableEq ι] (m : ι → ℕ)
    (hco : Pairwise (Nat.Coprime on m)) (hm : ∀ c, 0 < m c) {n rows columns : ℕ} (hn : 0 < n)
    (Q Qq : ι → ℕ) (hQq : ∀ c, 0 < Qq c) (hQ : ∀ c, Q c = Qq c * m c)
    (input : ∀ c, ExactMatrix (Q c) n rows columns)
    (output : ∀ c, ExactMatrix (m c) n rows columns)
    (runs : ∀ c, modulusSwitchRuns (input c) (output c))
    (row : Fin rows) (column : Fin columns) (i : Fin n) (y : ZMod (∏ c, m c))
    (yhat ν : ι → ℤ)
    (hrel : ∀ c, Int.ofNat ((input c row column).coeff i).val ≡
      (Qq c : ℤ) * yhat c + ν c [ZMOD (Q c : ℤ)])
    (hmargin : ∀ c, 2 * |ν c| < Qq c)
    (hlink : ∀ c, (yhat c : ZMod (m c)) = ZMod.prodEquivPi m hco y c)
    (w : ι → ZMod (∏ c, m c)) (hw : ∀ c, ZMod.prodEquivPi m hco (w c) = Pi.single c 1) :
    ∑ c, w c * (centeredLift (m c) ((output c row column).coeff i) : ZMod (∏ c, m c)) = y := by
  refine crt_recombine m hco y _ (fun c ↦ ?_) w hw
  rw [centeredLift_cast _ (hm c),
    modulusSwitch_residue hn (hQq c) (hQ c) (input c) (output c) (runs c) row column i (yhat c)
      (ν c) (hrel c) (hmargin c), hlink c]

/-! ## The average case -/

section Tail

open MeasureTheory MxxRuntime Function

variable {ι : Type} [DecidableEq ι] (μ : ι → Measure ℤ) [∀ i, IsProbabilityMeasure (μ i)]

/-- The tail of every decoder of a refresh. The source errors `X` (the encryption errors, the
child carriers' errors, the hint and smudge errors) form a coordinatewise sub-Gaussian vector of
proxies `v` over the block `B`; decoded coefficient `j` has the noise `N_j = Σ_i m_ji X_i` with
coefficients reading only `T`, disjoint from `B` (the digit actions and the preimages), without
any independence among the terms (`CSubG.combine`), and the smudge `M_j` with `|M_j| ≤ S`. Then
some decoded coefficient leaves the margin, `Q_q ≤ 2 |M_j + N_j|`, with probability at most
`Σ_j 2 exp (-(Q_q/2 - S)² / (2 w_j²))` for any `w_j ≥ Σ_i |m_ji| √(v_i)`. -/
theorem refresh_noise_tail {κ κ' : Type} [Fintype κ] [Nonempty κ] [Fintype κ']
    {X : (ι → ℤ) → κ → ℝ} {B : Finset ι} {v : κ → ℝ} (hX : CSubG μ X B v)
    (hv : ∀ i, 0 < v i) (R : Finset ι) (hXdep : DependsOn X R)
    {m : (ι → ℤ) → κ' → κ → ℝ} (T : Finset ι) (hT : Disjoint T B) (hm : DependsOn m T)
    {w : κ' → ℝ} (hw : ∀ ω j, ∑ i, |m ω j i| * Real.sqrt (v i) ≤ w j) (hw0 : ∀ j, 0 < w j)
    (M : κ' → ℝ) {Qq S : ℝ} (hM : ∀ j, |M j| ≤ S) (hS : S < Qq / 2) :
    Measure.infinitePi μ {ω | ∃ j, Qq ≤ 2 * |M j + ∑ i, m ω j i * X ω i|} ≤
      ∑ j, ENNReal.ofReal (2 * Real.exp (-(Qq / 2 - S) ^ 2 / (2 * w j ^ 2))) := by
  have hN := hX.combine hv R hXdep T hT hm hw
  have hNdep : DependsOn (fun ω j ↦ ∑ i, m ω j i * X ω i) ((R ∪ T : Finset ι) : Set ι) := by
    intro ω ω' h
    have h1 := hXdep (fun x hx ↦ h x (by simp [Finset.mem_coe.mp hx]))
    have h2 := hm (fun x hx ↦ h x (by simp [Finset.mem_coe.mp hx]))
    simp only [h1, h2]
  have hsub : {ω | ∃ j, Qq ≤ 2 * |M j + ∑ i, m ω j i * X ω i|} ⊆
      ⋃ j, {ω | Qq / 2 - S ≤ |∑ i, m ω j i * X ω i|} := by
    intro ω ⟨j, hj⟩
    refine Set.mem_iUnion.mpr ⟨j, ?_⟩
    have := abs_add_le (M j) (∑ i, m ω j i * X ω i)
    have := hM j
    simp only [Set.mem_setOf_eq]
    linarith
  calc Measure.infinitePi μ {ω | ∃ j, Qq ≤ 2 * |M j + ∑ i, m ω j i * X ω i|}
      ≤ Measure.infinitePi μ (⋃ j, {ω | Qq / 2 - S ≤ |∑ i, m ω j i * X ω i|}) :=
        measure_mono hsub
    _ ≤ ∑ j, Measure.infinitePi μ {ω | Qq / 2 - S ≤ |∑ i, m ω j i * X ω i|} :=
        measure_iUnion_fintype_le _ _
    _ ≤ ∑ j, ENNReal.ofReal (2 * Real.exp (-(Qq / 2 - S) ^ 2 / (2 * w j ^ 2))) := by
        refine Finset.sum_le_sum fun j _ ↦ ?_
        exact hN.tail (R ∪ T) hNdep j (pow_pos (hw0 j) 2) (by linarith)

open scoped Function in
open Mxx.Primitives in
/-- The average-case correctness of the refresh to the top level. On every tape `ω`, chain `c`
runs its modulus switch, and its decoded coefficient `j = (row, column, i)` is
`Q_q c ŷ + M c j + N c j ω` (modulo `Q c`) with the smudge `|M c j| ≤ S c` and the noise
`N c j ω = Σ_k m c ω j k X ω k`, a combination (without independence) of one coordinatewise
sub-Gaussian vector `X` of all source errors whose coefficients read only `T c`, disjoint from its
block. Then the recombined output differs from the reference value `y` in some coefficient with
probability at most `Σ_c Σ_j 2 exp (-(Q_q c / 2 - S c)² / (2 w c j²))`. -/
theorem top_refresh_average {Ch κ : Type} [Fintype Ch] [DecidableEq Ch] [Fintype κ] [Nonempty κ]
    (m : Ch → ℕ) (hco : Pairwise (Nat.Coprime on m)) (hm : ∀ c, 0 < m c)
    {n rows columns : ℕ} (hn : 0 < n) (Q Qq : Ch → ℕ) (hQq : ∀ c, 0 < Qq c)
    (hQ : ∀ c, Q c = Qq c * m c)
    (input : (ι → ℤ) → ∀ c, ExactMatrix (Q c) n rows columns)
    (output : (ι → ℤ) → ∀ c, ExactMatrix (m c) n rows columns)
    (runs : ∀ ω c, MxxRuntime.modulusSwitchRuns (input ω c) (output ω c))
    (y : Fin rows → Fin columns → Fin n → ZMod (∏ c, m c))
    (yhat : Ch → Fin rows × Fin columns × Fin n → ℤ)
    (hlink : ∀ c j, (yhat c j : ZMod (m c)) = ZMod.prodEquivPi m hco (y j.1 j.2.1 j.2.2) c)
    {X : (ι → ℤ) → κ → ℝ} {B : Finset ι} {v : κ → ℝ} (hX : CSubG μ X B v) (hv : ∀ k, 0 < v k)
    (R : Finset ι) (hXdep : DependsOn X R)
    (mc : Ch → (ι → ℤ) → Fin rows × Fin columns × Fin n → κ → ℤ) (T : Ch → Finset ι)
    (hT : ∀ c, Disjoint (T c) B) (hmdep : ∀ c, DependsOn (fun ω j k ↦ (mc c ω j k : ℝ)) (T c))
    (Xint : (ι → ℤ) → κ → ℤ) (hXint : ∀ ω k, (Xint ω k : ℝ) = X ω k)
    (M : Ch → Fin rows × Fin columns × Fin n → ℤ) (S : Ch → ℝ) (hM : ∀ c j, |(M c j : ℝ)| ≤ S c)
    (hS : ∀ c, S c < Qq c / 2)
    (hrel : ∀ ω c (j : Fin rows × Fin columns × Fin n),
      Int.ofNat ((input ω c j.1 j.2.1).coeff j.2.2).val ≡
        (Qq c : ℤ) * yhat c j + (M c j + ∑ k, mc c ω j k * Xint ω k) [ZMOD (Q c : ℤ)])
    (w : Ch → Fin rows × Fin columns × Fin n → ℝ)
    (hw : ∀ c ω j, ∑ k, |(mc c ω j k : ℝ)| * Real.sqrt (v k) ≤ w c j) (hw0 : ∀ c j, 0 < w c j)
    (crt : Ch → ZMod (∏ c, m c)) (hcrt : ∀ c, ZMod.prodEquivPi m hco (crt c) = Pi.single c 1) :
    Measure.infinitePi μ {ω | ∃ j : Fin rows × Fin columns × Fin n,
        ∑ c, crt c * (centeredLift (m c) ((output ω c j.1 j.2.1).coeff j.2.2) : ZMod (∏ c, m c)) ≠
          y j.1 j.2.1 j.2.2} ≤
      ∑ c, ∑ j, ENNReal.ofReal (2 * Real.exp (-((Qq c : ℝ) / 2 - S c) ^ 2 / (2 * w c j ^ 2))) := by
  -- A wrong output needs a decoded coefficient outside its margin.
  have hsub : {ω | ∃ j : Fin rows × Fin columns × Fin n,
        ∑ c, crt c * (centeredLift (m c) ((output ω c j.1 j.2.1).coeff j.2.2) : ZMod (∏ c, m c)) ≠
          y j.1 j.2.1 j.2.2} ⊆
      ⋃ c, {ω | ∃ j, (Qq c : ℝ) ≤
        2 * |(M c j : ℝ) + ∑ k, (mc c ω j k : ℝ) * X ω k|} := by
    intro ω ⟨j, hj⟩
    by_contra hnone
    simp only [Set.mem_iUnion, Set.mem_setOf_eq, not_exists, not_le] at hnone
    apply hj
    refine top_refresh_coefficient m hco hm hn Q Qq hQq hQ (input ω) (output ω) (runs ω)
      j.1 j.2.1 j.2.2 (y j.1 j.2.1 j.2.2) (yhat · j) (fun c ↦ M c j + ∑ k, mc c ω j k * Xint ω k)
      (fun c ↦ hrel ω c j) (fun c ↦ ?_) (fun c ↦ hlink c j) crt hcrt
    have h : (2 : ℝ) * |(M c j : ℝ) + ∑ k, (mc c ω j k : ℝ) * X ω k| < Qq c := hnone c j
    have e : ((2 * |M c j + ∑ k, mc c ω j k * Xint ω k| : ℤ) : ℝ) =
        2 * |(M c j : ℝ) + ∑ k, (mc c ω j k : ℝ) * X ω k| := by
      push_cast
      simp [hXint]
    have h' : ((2 * |M c j + ∑ k, mc c ω j k * Xint ω k| : ℤ) : ℝ) < ((Qq c : ℤ) : ℝ) := by
      rw [e]
      exact_mod_cast h
    exact_mod_cast h'
  calc _ ≤ Measure.infinitePi μ (⋃ c, {ω | ∃ j, (Qq c : ℝ) ≤
          2 * |(M c j : ℝ) + ∑ k, (mc c ω j k : ℝ) * X ω k|}) := measure_mono hsub
    _ ≤ ∑ c, Measure.infinitePi μ {ω | ∃ j, (Qq c : ℝ) ≤
          2 * |(M c j : ℝ) + ∑ k, (mc c ω j k : ℝ) * X ω k|} := measure_iUnion_fintype_le _ _
    _ ≤ _ := Finset.sum_le_sum fun c _ ↦
        refresh_noise_tail μ hX hv R hXdep (T c) (hT c) (hmdep c) (hw c) (hw0 c)
          (fun j ↦ (M c j : ℝ)) (hM c) (hS c)

/-- The failure bound at the parameters of `tests/refresh_top.rs` (four chains, two components of
six columns over the ring of dimension eight, decoding margins `Q_q ≥ 2^59` from two 30-bit
primes, smudges `P^d / 2 ≤ 2^35` for `P = 4`, `d = 18`): with every noise proxy `w ≤ 2^34`, the
probability of a wrong output is below `2^-100`. -/
theorem refresh_top_test_tail (Qq S : Fin 4 → ℝ) (w : Fin 4 → Fin 2 × Fin 6 × Fin 8 → ℝ)
    (hQq : ∀ c, (2 : ℝ) ^ 59 ≤ Qq c) (hS : ∀ c, S c ≤ 2 ^ 35) (hS0 : ∀ c, 0 ≤ S c)
    (hw : ∀ c j, w c j ≤ 2 ^ 34) (hw0 : ∀ c j, 0 < w c j) :
    ∑ c, ∑ j, ENNReal.ofReal (2 * Real.exp (-(Qq c / 2 - S c) ^ 2 / (2 * w c j ^ 2))) ≤
      (2 : ENNReal)⁻¹ ^ 100 := by
  have hterm : ∀ c j, Real.exp (-(Qq c / 2 - S c) ^ 2 / (2 * w c j ^ 2)) ≤ (10 / 27 : ℝ) ^ 128 := by
    intro c j
    refine MxxExponentLut.exp_le_of_le_neg_nat 128 ?_
    have hw2 : 0 < 2 * w c j ^ 2 := by have := hw0 c j; positivity
    have hgap : (2 : ℝ) ^ 57 ≤ Qq c / 2 - S c := by
      have := hQq c; have := hS c; nlinarith
    have hsq : ((2 : ℝ) ^ 57) ^ 2 ≤ (Qq c / 2 - S c) ^ 2 :=
      pow_le_pow_left₀ (by positivity) hgap 2
    have hden : 2 * w c j ^ 2 ≤ 2 * ((2 : ℝ) ^ 34) ^ 2 := by
      have := hw c j; have := hw0 c j; nlinarith
    rw [neg_div, neg_le_neg_iff, le_div_iff₀ hw2]
    push_cast
    nlinarith
  calc ∑ c, ∑ j, ENNReal.ofReal (2 * Real.exp (-(Qq c / 2 - S c) ^ 2 / (2 * w c j ^ 2)))
      ≤ ∑ _c : Fin 4, ∑ _j : Fin 2 × Fin 6 × Fin 8, ENNReal.ofReal (2 * (10 / 27 : ℝ) ^ 128) := by
        gcongr with c _ j _
        exact hterm c j
    _ = ((384 : ℕ) : ENNReal) * ENNReal.ofReal (2 * (10 / 27 : ℝ) ^ 128) + ENNReal.ofReal 0 := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, Fintype.card_prod,
          nsmul_eq_mul, ENNReal.ofReal_zero, add_zero]
        push_cast
        ring
    _ ≤ (2 : ENNReal)⁻¹ ^ 100 :=
        MxxExponentLut.ennreal_tail_le (by positivity) le_rfl (by norm_num)

end Tail

end Mxx.ExponentLut.Refresh
