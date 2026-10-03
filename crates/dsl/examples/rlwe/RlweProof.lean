import Claim
import RuntimeLemmas

/-!
The generated correctness claim of the RLWE example: every execution decrypts its message bit.

Decryption reads the constant coefficient of `b - a s = e + floor(Q/2) m`. The Gaussian error
`e` is truncated at the cutoff `26`, so that coefficient lies within `26` of `0` or of
`floor(Q/2)`, and rounding it to the nearest multiple of `Q/2` returns `m`.
-/

namespace MxxRlweExample

open Mxx.Primitives MxxRuntime GeneratedClaim

abbrev Q : Nat := 1532495540865518635130821056977027158796330141975560193
abbrev N : Nat := 4096
/-- The message scale `floor(Q/2)`, as the generated relation writes it. -/
abbrev Δ : Int := Int.fdiv 1532495540865518635130821056977027158796330141975560193 2

/-- The constant coefficient of an integer constant. -/
theorem coeff_zero_intCast (c : Int) :
    ((c : ExactPoly Q N)).coeff ⟨0, by decide⟩ = (c : ZMod Q) := by
  letI : Fact (1 < Q) := ⟨by decide⟩
  have hc := Negacyclic.coeff_root_pow (R := ZMod Q) (by decide : 0 < N) ⟨0, by decide⟩
    ⟨0, by decide⟩
  simp only [pow_zero, if_true] at hc
  rw [intCast_eq_algebraMap, ← mul_one (algebraMap (ZMod Q) (ExactPoly Q N) (c : ZMod Q)),
    Negacyclic.coeff_smul, hc, mul_one]

theorem correctness : CorrectnessClaim := by
  intro hashModel external execution hruns
  obtain ⟨⟨_, hm0, hm1⟩, ⟨w, _, _, he, _, _, hdec, hout⟩, ⟨_, hideal⟩⟩ := hruns
  simp only at hdec hout hideal
  obtain ⟨E, hE, hEb⟩ := gaussianSample_bounded he
  have he0 := natAbs_coeff_le_of_polyNorm (hEb 0 0) ⟨0, by decide⟩
  simp only [stage_0_params] at he0
  -- `b - a s` is the error plus the scaled message.
  have hdiff : matrixSub (matrixAdd (matrixAdd (matrixMulScalarLeft w.w_1_0 w.w_2_0) w.w_4_0)
      (matrixMulScalarLeft (matrixPolynomial [Δ])
        (liftInteger external.input_1)))
      (matrixMulScalarLeft w.w_1_0 w.w_2_0) 0 0 =
      reducePoly Q N (E 0 0) + ((Δ * external.input_1 : Int) : ExactPoly Q N) := by
    simp only [matrixSub, matrixAdd, matrixMulScalarLeft, liftInteger, Matrix.sub_apply,
      Matrix.add_apply, matrixPolynomial_single, hE, reduceMatrix_apply]
    push_cast
    ring
  obtain ⟨_, _, _, _, index, hindex, _, hdecoded⟩ := hdec
  have hi : index = ⟨0, by decide⟩ := by
    have h : (index : Nat) = 0 := by exact_mod_cast hindex
    exact Fin.ext h
  subst hi
  rw [hdiff, Negacyclic.coeff_add, reducePoly_coeff (by decide) (by decide), coeff_zero_intCast,
    ← Int.cast_add, val_intCast_emod (by decide)] at hdecoded
  have hΔ : Δ = 766247770432759317565410528488513579398165070987780096 := by decide
  rw [hΔ] at hdecoded
  have he0' : ((E 0 0).coeff ⟨0, by decide⟩).natAbs ≤ 26 := by simpa using he0
  -- The error coefficient is within `26` of zero, so the constant coefficient rounds to the
  -- message.
  generalize (E 0 0).coeff ⟨0, by decide⟩ = e at hdecoded he0'
  have hlow : -26 ≤ e := by omega
  have hhigh : e ≤ 26 := by omega
  have hbit : w.w_13_0_decoded = external.input_1 := by
    rw [hdecoded]
    clear hdecoded he0' hdiff hout
    rcases (show external.input_1 = 0 ∨ external.input_1 = 1 by omega) with h | h <;>
      rw [h] <;> interval_cases e <;> norm_num
  rw [hout, hideal, hbit]
  rcases (show external.input_1 = 0 ∨ external.input_1 = 1 by omega) with h | h <;> simp [h]

end MxxRlweExample
