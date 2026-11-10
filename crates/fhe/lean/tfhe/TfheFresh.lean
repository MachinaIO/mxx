import TfheRotation

/-!
# The phase of a fresh encryption

The ring ciphertext of `Enc` has the constant phase `(2 m - 1) ⌊Q/8⌋ + freshNoise`, where
`freshNoise` is the constant coefficient of `u e + e₂ - e₁ z` over the integers, so `ks_phase`
gives the phase of the fresh LWE ciphertext.
-/

set_option maxRecDepth 16384

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime MxxSecurity

/-- The integer polynomial sampled from `law` at site `s`. -/
noncomputable def ip (law : SamplerLaw) (s : Nat) (ω : Key → ℤ) : ErrorPoly N :=
  (smpInt law s ω : ErrorMatrix N 1 1) 0 0

theorem ip_coeff (law : SamplerLaw) (s : Nat) (ω : Key → ℤ) (k : Fin N) :
    (ip law s ω).coeff k = law.clamp (ω (s, k.val)) := by
  rw [ip, smpInt_coeff (by decide)]
  congr 3

theorem smp_entry (law : SamplerLaw) (s : Nat) (ω : Key → ℤ) :
    (smp law s ω : ExactMatrix Q N 1 1) 0 0 = reducePoly Q N (ip law s ω) := by
  rw [smp_eq_reduce]
  rfl

theorem bit_clamp (x : Int) : bitLaw.clamp x = 0 ∨ bitLaw.clamp x = 1 := by
  have := SamplerLaw.clamp_supports bitLaw x
  simp only [SamplerLaw.Supports] at this
  norm_num at this
  omega

theorem ring_clamp (x : Int) : |ringLaw.clamp x| ≤ 2496 := by
  have := SamplerLaw.clamp_supports ringLaw x
  simp only [SamplerLaw.Supports] at this
  norm_num at this
  exact this

theorem lwe_clamp (x : Int) : |lweLaw.clamp x| ≤ 524288 := by
  have := SamplerLaw.clamp_supports lweLaw x
  simp only [SamplerLaw.Supports] at this
  norm_num at this
  exact this

/-- The constant coefficient of a product of a binary polynomial with a bounded one. -/
theorem coeff_zero_mul_bound {x y : ErrorPoly N} (hx : ∀ k, x.coeff k = 0 ∨ x.coeff k = 1)
    {b : Int} (hb : 0 ≤ b) (hy : ∀ k, |y.coeff k| ≤ b) :
    |(x * y).coeff 0| ≤ 1024 * b := by
  rw [show (0 : Fin N) = ⟨0, by decide⟩ from rfl, coeff_zero_mul (by decide)]
  refine (abs_sum_le_card _ b fun j ↦ ?_).trans (by simp)
  rw [abs_mul, abs_mul]
  have hs : |(if j.val = 0 then (1 : Int) else -1)| = 1 := by split_ifs <;> simp
  rw [hs, one_mul]
  rcases hx ⟨(N - j.val) % N, Nat.mod_lt _ (by decide)⟩ with h | h <;> rw [h] <;> simp [hy j, hb]

theorem coeff_zero_mul_bound' {x y : ErrorPoly N} (hy : ∀ k, y.coeff k = 0 ∨ y.coeff k = 1)
    {b : Int} (hb : 0 ≤ b) (hx : ∀ k, |x.coeff k| ≤ b) :
    |(x * y).coeff 0| ≤ 1024 * b := by
  rw [mul_comm]
  exact coeff_zero_mul_bound hy hb hx

/-- The integer noise of the fresh ring ciphertext at sites `base`, `base + 1`, `base + 2`:
the constant coefficient of `u e + e₂ - e₁ z`. -/
noncomputable def freshNoise (ω : Key → ℤ) (base : Nat) : Int :=
  (ip bitLaw base ω * ip ringLaw 14 ω + ip ringLaw (base + 2) ω -
    ip ringLaw (base + 1) ω * ip bitLaw 2 ω).coeff 0

theorem freshNoise_bound (ω : Key → ℤ) (base : Nat) : |freshNoise ω base| ≤ 5114304 := by
  have hu : ∀ k, (ip bitLaw base ω).coeff k = 0 ∨ (ip bitLaw base ω).coeff k = 1 :=
    fun k ↦ by rw [ip_coeff]; exact bit_clamp _
  have hz : ∀ k, (ip bitLaw 2 ω).coeff k = 0 ∨ (ip bitLaw 2 ω).coeff k = 1 :=
    fun k ↦ by rw [ip_coeff]; exact bit_clamp _
  have he (s : Nat) : ∀ k, |(ip ringLaw s ω).coeff k| ≤ 2496 :=
    fun k ↦ by rw [ip_coeff]; exact ring_clamp _
  have h1 := coeff_zero_mul_bound hu (by norm_num) (he 14)
  have h2 := coeff_zero_mul_bound' hz (by norm_num) (he (base + 1))
  have h3 := he (base + 2) 0
  unfold freshNoise
  rw [Negacyclic.coeff_sub, Negacyclic.coeff_add]
  rw [abs_le] at h1 h2 h3 ⊢
  constructor <;> linarith [h1.1, h1.2, h2.1, h2.2, h3.1, h3.2]

/-- The key-switching error `i`: the centered coefficient `i % N` of error polynomial `i / N`. -/
noncomputable def kskErr (ω : Key → ℤ) (i : Fin 8192) : Int :=
  centerQ (kskErrs ω ⟨i.val / 1024, by omega⟩ ⟨i.val % 1024, by omega⟩)

/-- The ring secret's coefficients. -/
noncomputable def ringSecret (ω : Key → ℤ) : Fin 1024 → Int := coeffVals (smp bitLaw 2 ω)

theorem coeffVals_smp_bit (s : Nat) (ω : Key → ℤ) (k : Fin 1024) :
    coeffVals (smp bitLaw s ω) k = bitLaw.clamp (ω (s, k.val)) := by
  rw [coeffVals, smp_entry, reducePoly_coeff (by decide) (by decide), ip_coeff]
  have := bit_clamp (ω (s, k.val))
  rcases this with h | h <;> rw [h] <;> rfl

theorem ringSecret_binary (ω : Key → ℤ) (k : Fin 1024) :
    ringSecret ω k = 0 ∨ ringSecret ω k = 1 := by
  rw [ringSecret, coeffVals_smp_bit]
  exact bit_clamp _

theorem coeff_intCast_zero (μ : Int) : ((μ : ExactPoly Q N)).coeff 0 = (μ : ZMod Q) := by
  letI : Fact (1 < Q) := ⟨by decide⟩
  have h : (μ : ExactPoly Q N) = algebraMap (ZMod Q) (ExactPoly Q N) (μ : ZMod Q) *
      AdjoinRoot.root (negacyclicModulus N (ZMod Q)) ^ ((0 : Fin N) : Nat) := by
    simp
  rw [h, Negacyclic.coeff_smul, Negacyclic.coeff_root_pow (by decide)]
  simp

theorem fresh_phase (ω : Key → ℤ) (base : Nat) (m : Int) :
    ∃ η : Int, (((freshOf ω base m).2.1 -
        ∑ j, (freshOf ω base m).1 j * secretOf ω j : Int) : ZMod q) = η ∧
      |(η : ℝ) - ((4294967296 / 5234636801 : ℝ) * ((freshNoise ω base +
        (m * 2 - 1) * 654329600 : Int) : ℝ) -
        ∑ i, (encDigits (encInputsOf ω m) (smp bitLaw base ω) (smp ringLaw (base + 1) ω) i : ℝ) *
          kskErr ω i)| ≤ allowance := by
  letI : Fact (1 < Q) := ⟨by decide⟩
  set u : ExactMatrix Q N 1 1 := smp bitLaw base ω
  set e1 : ExactMatrix Q N 1 1 := smp ringLaw (base + 1) ω
  set e2 : ExactMatrix Q N 1 1 := smp ringLaw (base + 2) ω
  have hzp : ∀ k, ((smp bitLaw 2 ω : ExactMatrix Q N 1 1) 0 0).coeff k = (ringSecret ω k : ZMod Q) := by
    intro k
    simp [ringSecret, coeffVals]
  have hy : (((freshNoise ω base + (m * 2 - 1) * 654329600 : Int)) : ZMod Q) =
      (encC1 (pkBOf ω) u e2 m 0 0).coeff 0 -
        (encC0 (smp (.residue Q) 13 ω) u e1 0 0 * (smp bitLaw 2 ω : ExactMatrix Q N 1 1) 0 0).coeff 0 := by
    have hc : encC1 (pkBOf ω) u e2 m 0 0 -
        encC0 (smp (.residue Q) 13 ω) u e1 0 0 * (smp bitLaw 2 ω : ExactMatrix Q N 1 1) 0 0 =
        reducePoly Q N (ip bitLaw base ω * ip ringLaw 14 ω + ip ringLaw (base + 2) ω -
          ip ringLaw (base + 1) ω * ip bitLaw 2 ω) + (((m * 2 - 1) * 654329600 : Int) : ExactPoly Q N) := by
      simp only [encC1, encC0, pkBOf, matrixAdd, matrixMulScalarLeft, liftInteger, Matrix.add_apply,
        map_add, map_sub, map_mul, ← smp_entry, u, e1, e2]
      ring
    rw [← Negacyclic.coeff_sub, hc, Negacyclic.coeff_add, coeff_intCast_zero,
      reducePoly_coeff (by decide) (by decide)]
    push_cast
    rfl
  exact ks_phase (s := secretOf ω) (ringSecret_binary ω) (A := kskAOf ω) (B := kskBOf ω)
    (ε := kskErr ω) (fun i ↦ rfl) (encC0 (smp (.residue Q) 13 ω) u e1)
    (encC1 (pkBOf ω) u e2 m) hzp hy

end MxxFheTfhe
