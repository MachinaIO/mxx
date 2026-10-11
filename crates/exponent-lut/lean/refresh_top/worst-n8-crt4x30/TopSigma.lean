import Part_0_0_MaskCarrier
import TopDecode

/-!
# The mask's value modulo `q₀`

The mask carrier of chain `0` encodes `β = Σ_o X^o (T'(z_o) + T'(1))` with the sampling table
`T = (0, h, 0, h)`, `2 h = q₀ + 1`: the readout `readoutExp z` has residue `1` or `3` modulo `4`,
so `T'(z) + T'(1) = ((-1)^⌊z/4⌋ + 1) h`, which is `1` modulo `q₀` when `⌊z/4⌋` is even and `0`
otherwise, that is, `σ = 1 - ⌊(z mod 8) / 4⌋`.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Part_0_0

/-- The sampling table of the mask at chain `0`. -/
noncomputable def maskTable0 (k : Fin 4) : ErrorPoly N :=
  ![((0 : Int) : ErrorPoly N), ((536870721 : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N),
    ((536870721 : Int) : ErrorPoly N)] k

theorem readoutExp_split (z : Nat) : readoutExp z / 4 = z / 4 ∧
    (readoutExp z % 4 = 1 ∨ readoutExp z % 4 = 3) := by
  have hv : ∀ f : Fin 4, (![1, 1, 3, 3] f : Nat) = 1 ∨ (![1, 1, 3, 3] f : Nat) = 3 := by decide
  unfold readoutExp
  rcases hv ⟨z % 4, Nat.mod_lt _ (by norm_num)⟩ with h | h <;> rw [h] <;> omega

theorem sampled_mask0 (z : Nat) :
    sampled maskTable0 (readoutExp z) = (((-1 : Int) ^ (z / 4) * 536870721 : Int) : ErrorPoly N) := by
  obtain ⟨hdiv, hmod⟩ := readoutExp_split z
  unfold sampled
  rw [hdiv]
  have ht : maskTable0 ⟨readoutExp z % 4, Nat.mod_lt _ (by norm_num)⟩ =
      ((536870721 : Int) : ErrorPoly N) := by
    rcases hmod with h | h
    · rw [show (⟨readoutExp z % 4, Nat.mod_lt _ (by norm_num)⟩ : Fin 4) = 1 from Fin.ext h]; rfl
    · rw [show (⟨readoutExp z % 4, Nat.mod_lt _ (by norm_num)⟩ : Fin 4) = 3 from Fin.ext h]; rfl
  rw [ht]
  push_cast
  ring

theorem sampled_one_mask0 : sampled maskTable0 1 = ((536870721 : Int) : ErrorPoly N) := by
  unfold sampled
  rw [show (⟨1 % 4, Nat.mod_lt _ (by norm_num)⟩ : Fin 4) = 1 from rfl]
  simp [maskTable0]

/-- A member's value is its `σ` modulo `q₀`. -/
theorem member_sigma (z : Nat) :
    reducePoly q0 N (sampled maskTable0 (readoutExp z) + sampled maskTable0 1) =
      reducePoly q0 N (((1 - ((z : Int) % 8) / 4 : Int)) : ErrorPoly N) := by
  rw [sampled_mask0, sampled_one_mask0, ← Int.cast_add, reducePoly_intCast, reducePoly_intCast]
  have h8 : ((z : Int) % 8) / 4 = ((z / 4 : Nat) : Int) % 2 := by omega
  rw [h8]
  rcases Nat.even_or_odd (z / 4) with ⟨m, hm⟩ | ⟨m, hm⟩
  · rw [hm, show ((m + m : Nat) : Int) % 2 = 0 by omega, Even.neg_one_pow ⟨m, rfl⟩]
    rw [intCast_eq_algebraMap, intCast_eq_algebraMap]
    congr 1
  · rw [hm, show ((2 * m + 1 : Nat) : Int) % 2 = 1 by omega, Odd.neg_one_pow ⟨m, rfl⟩]
    rw [intCast_eq_algebraMap, intCast_eq_algebraMap]
    congr 1

/-- The mask's value is `σ` modulo `q₀`. -/
theorem beta_sigma (z : Fin 8 → Nat) :
    reducePoly q0 N ((1 : ErrorPoly N) * (∑ o : Fin 8, Xe (n := N) o *
      (sampled maskTable0 (readoutExp (z o)) + sampled maskTable0 1)) * 1) =
      reducePoly q0 N (∑ o : Fin 8, (((1 - ((z o : Int) % 8) / 4 : Int)) : ErrorPoly N) *
        Xe (n := N) o) := by
  simp only [one_mul, mul_one, map_sum, map_mul, member_sigma]
  exact Finset.sum_congr rfl fun o _ ↦ mul_comm _ _

/-- `σ` as the decode stage builds it. -/
theorem sigmaPoly_eq (s : Fin 8 → Int) :
    sigmaPoly s = fun _ _ ↦ reducePoly Q0 N (∑ o : Fin 8, ((s o : Int) : ErrorPoly N) * Xe (n := N) o) := by
  funext r c
  simp only [sigmaPoly, matrixAdd, multiplyMonomial_liftInteger, Matrix.add_apply, Fin.sum_univ_eight,
    map_add, map_mul, message]
  simp [Xe]

end RefreshTopProof
