import PeIndep
import ExponentMaskForm

/-!
# The phase over the integers

The phase is the reduction of an integer polynomial: `⌊Q/2⌋ m + e_m` minus the bias
`e_b k̃₀ + σ_K(e_b k̃₁)` and the linear form `e_h v₀ + σ_K(e_h) v₁ + e_F v_F` in the key extension.
Each coefficient of the linear form is a sum over the tape coordinates of the extension.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The joined integer row `[e₀ | e₁]`. -/
def jZ (e : Fin 2 → ErrorMatrix N 1 8) : ErrorMatrix N 1 16 := fun _ p ↦ joinedEntry e p

theorem joinedEntry_split' {α : Type} (r : Fin 2 → Matrix (Fin 1) (Fin 8) α) (p : Fin 16) :
    joinedEntry r p = if h : p.val < 8 then r 0 0 ⟨p.val, h⟩ else r 1 0 ⟨p.val - 8, by omega⟩ := by
  unfold joinedEntry Fin.addCases
  split <;> simp_all <;> rfl

theorem joinRow_reduce (e : Fin 2 → ErrorMatrix N 1 8) :
    joinRow (fun j ↦ reduceMatrix Q N 1 8 (e j)) = reduceMatrix Q N 1 16 (jZ e) := by
  funext r p
  simp only [joinRow, jZ, reduceMatrix, joinedEntry_split']
  split <;> rfl

theorem σK_reduce {r c : Nat} (x : ErrorMatrix N r c) :
    σK (reduceMatrix Q N r c x) = reduceMatrix Q N r c (substituteMatrix K hK x) := by
  funext i j
  simp only [σK, substituteMatrix, reduceMatrix, reducePoly_substitution]

/-- The phase entry as the reduction of an integer polynomial. -/
theorem phase_entry {em : ErrorMatrix N 1 1} {m : Fin N → Int} {eb : ErrorMatrix N 1 10}
    {k0 k1 : ErrorMatrix N 10 1} {eh eF : Fin 2 → ErrorMatrix N 1 8} {v0 v1 vF : ErrorMatrix N 16 1} :
    (reduceMatrix Q N 1 1 em + ((halfQ : ExactPoly Q N) • (fun _ _ ↦ polynomialOfCoefficients m) :
        ExactMatrix Q N 1 1) -
      (reduceMatrix Q N 1 10 eb * reduceMatrix Q N 10 1 k0 + σK (reduceMatrix Q N 1 10 eb * reduceMatrix Q N 10 1 k1) +
        joinRow (fun j ↦ reduceMatrix Q N 1 8 (eh j)) * reduceMatrix Q N 16 1 v0 +
        σK (joinRow (fun j ↦ reduceMatrix Q N 1 8 (eh j))) * reduceMatrix Q N 16 1 v1 +
        joinRow (fun j ↦ reduceMatrix Q N 1 8 (eF j)) * reduceMatrix Q N 16 1 vF)) 0 0 =
    reducePoly Q N (em 0 0 + (halfQ : ErrorPoly N) * intPoly m -
      ((eb * k0) 0 0 + substitution K hK ((eb * k1) 0 0) + (jZ eh * v0) 0 0 +
        (substituteMatrix K hK (jZ eh) * v1) 0 0 + (jZ eF * vF) 0 0)) := by
  have hP : (((halfQ : ExactPoly Q N) • (fun _ _ ↦ polynomialOfCoefficients m) : ExactMatrix Q N 1 1)) 0 0 =
      (halfQ : ExactPoly Q N) * polynomialOfCoefficients m := rfl
  rw [joinRow_reduce, joinRow_reduce, σK_reduce, ← reduceMatrix_mul, ← reduceMatrix_mul,
    ← reduceMatrix_mul, ← reduceMatrix_mul, ← reduceMatrix_mul, σK_reduce]
  simp only [Matrix.add_apply, Matrix.sub_apply, hP]
  simp only [reduceMatrix, map_add, map_sub, map_mul, polynomialOfCoefficients_eq_reduce, map_intCast,
    substituteMatrix, reducePoly_substitution]

/-- A coefficient of `A v` for an integer row `A` and a tape column `v` is a linear form in the tape
coordinates of `v`. -/
theorem lin_coeff (A : ErrorMatrix N 1 16) (ω : SampleTape) (site : List Nat) (law : SamplerLaw)
    (i : Fin N) :
    ((A * (tapeWitness ω site law : ErrorMatrix N 16 1)) 0 0).coeff i =
      ∑ r : Fin 16, ∑ k : Fin N, maskCoefficient (A 0 r) i k * ω ⟨site, r.val, 0, k.val, law⟩ := by
  rw [Matrix.mul_apply, Negacyclic.coeff_sum]
  refine Finset.sum_congr rfl fun r _ ↦ ?_
  rw [mul_comm, coeff_mul_eq hN]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [tapeWitness, intPoly_coeff hN, maskCoefficient, Finset.sum_mul]
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  rw [show ((0 : Fin 1) : Nat) = 0 from rfl]
  ring

theorem halfQ_coeff (m : Fin N → Int) (i : Fin N) :
    ((halfQ : ErrorPoly N) * intPoly m).coeff i = halfQ * m i := by
  rw [intCast_eq_algebraMap, Negacyclic.coeff_smul, intPoly_coeff hN]
  rfl

end PeProof
