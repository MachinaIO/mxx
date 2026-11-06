import PeDecode

/-!
# The phase

With the anchor encoding, the targets of the key, the helper, and the online output substituted,
every secret-dependent term of the phase cancels:
`phase = e_m + ⌊Q/2⌋ m - (e_b k̃₀ + σ_K(e_b k̃₁) + e_h v₀ + σ_K(e_h) v₁ + e_F v_F)`.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- `σ_K` on matrices over `Z_Q[X]/(X^N + 1)`. -/
noncomputable abbrev σK {r c : Nat} (x : ExactMatrix Q N r c) : ExactMatrix Q N r c := substituteMatrix K hK x

/-- `σ_K` on polynomials. -/
noncomputable abbrev σp (x : ExactPoly Q N) : ExactPoly Q N := substitution K hK x

theorem σK_add {r c : Nat} (x y : ExactMatrix Q N r c) : σK (x + y) = σK x + σK y := by
  funext i j
  simp [substituteMatrix]

theorem σK_sub {r c : Nat} (x y : ExactMatrix Q N r c) : σK (x - y) = σK x - σK y := by
  funext i j
  simp [substituteMatrix]

theorem σK_neg {r c : Nat} (x : ExactMatrix Q N r c) : σK (-x) = -σK x := by
  funext i j
  simp [substituteMatrix]

theorem σK_smul {r c : Nat} (a : ExactPoly Q N) (x : ExactMatrix Q N r c) : σK (a • x) = σp a • σK x := by
  funext i j
  simp [substituteMatrix]

theorem σK_mul {r k c : Nat} (x : ExactMatrix Q N r k) (y : ExactMatrix Q N k c) :
    σK (x * y) = σK x * σK y := by
  funext i j
  simp [substituteMatrix, Matrix.mul_apply, map_sum]

theorem σK_σK {r c : Nat} (x : ExactMatrix Q N r c) : σK (σK x) = x := by
  funext i j
  exact substitution_inverse hK hK hKinv _

theorem σp_σp (a : ExactPoly Q N) : σp (σp a) = a := substitution_inverse hK hK hKinv _

/-- A gadget row has constant entries, which `σ_K` fixes. -/
theorem σK_gadget {g : ExactMatrix Q N 1 {{D}}} (hg : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g) :
    σK g = g := by
  funext i j
  obtain ⟨x, hx⟩ := gadgetMatrixRuns_constant hg i j
  simp only [σK, substituteMatrix, hx, substitution_algebraMap]

theorem σK_joinRow (r : Fin 2 → ExactMatrix Q N 1 {{D}}) : σK (joinRow r) = joinRow fun j ↦ σK (r j) := by
  funext i p
  simp only [σK, substituteMatrix, joinRow, joinedEntry_split]
  split <;> rfl

theorem σK_zero {r c : Nat} : σK (0 : ExactMatrix Q N r c) = 0 := by
  funext i j
  simp [substituteMatrix]

theorem σK_gZero {g : ExactMatrix Q N 1 {{D}}} (hg : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g) :
    σK (gZero g) = gZero g := by
  rw [gZero, σK_joinRow]
  congr 1
  funext j
  fin_cases j
  · exact σK_gadget hg
  · exact σK_zero

theorem σK_zeroG {g : ExactMatrix Q N 1 {{D}}} (hg : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g) :
    σK (zeroG g) = zeroG g := by
  rw [zeroG, σK_joinRow]
  congr 1
  funext j
  fin_cases j
  · exact σK_zero
  · exact σK_gadget hg

/-- The phase identity. -/
theorem phase_eq {s : ExactPoly Q N} {g : ExactMatrix Q N 1 {{D}}}
    (hg : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g)
    {cm d0 d1 Em P : ExactMatrix Q N 1 1} {W B Eb : ExactMatrix Q N 1 {{MB}}} {k0 k1 : ExactMatrix Q N {{MB}} 1}
    {D J Eh CF AF0 EF : ExactMatrix Q N 1 {{M}}} {v0 v1 vF : ExactMatrix Q N {{M}} 1}
    (hW : W = s • B + Eb) (hk0 : B * k0 = d0 - AF0 * vF - J * v0 + gZero g * v1)
    (hk1 : B * k1 = σK (-(σK J * v1) + gZero g * v0))
    (hD : D = s • J - σp s • gZero g - zeroG g + Eh) (hCF : CF = s • AF0 + EF)
    (hcm : cm - d1 = s • d0 + Em + P) :
    cm - d1 - (W * k0 + σK (W * k1) + ((D + zeroG g) * v0 + (σK D + zeroG g) * v1) + CF * vF) =
      Em + P - (Eb * k0 + σK (Eb * k1) + Eh * v0 + σK Eh * v1 + EF * vF) := by
  have hσD : σK D + zeroG g = σp s • σK J - s • gZero g + σK Eh := by
    rw [hD, σK_add, σK_sub, σK_sub, σK_smul, σK_smul, σp_σp, σK_gZero hg, σK_zeroG hg]
    abel
  rw [hσD, hcm, hD, hCF, hW]
  simp only [Matrix.add_mul, Matrix.sub_mul, Matrix.smul_mul, σK_add, σK_smul, σK_mul, hk0, hk1,
    σK_σK, smul_add, smul_sub, smul_neg]
  abel

end PeProof
