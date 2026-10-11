import TopLemmas

/-!
# Monomial placement

`mul_small_constant` by a monomial `X^k` multiplies every block of an encoding by `X^k`: the
target is multiplied by `X^k` and the error keeps its bound, since a monomial permutes the
coefficients up to sign.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem multiplyMonomial_eq {q rows columns : Nat} (x : ExactMatrix q N rows columns) {k : Nat}
    (hk : k < 2 * N) :
    multiplyMonomial x (k : Int) = reducePoly q N (Xe (n := N) k) • x := by
  funext r c
  simp only [multiplyMonomial, Matrix.smul_apply, smul_eq_mul]
  have hmod : ((k : Int) % (2 * (N : Int))).toNat = k := by
    have hk' : (k : Int) < 2 * (N : Int) := by simp only [N] at hk ⊢; omega
    rw [Int.emod_eq_of_lt (by omega) hk']
    simp
  rw [hmod, reducePoly_Xe, mul_comm]

/-- The monomial placement of an encoding. -/
theorem rel_monomial {q m : Nat} {sm : ErrorPoly N} {A : Fin 2 → Fin 2 → ExactMatrix q N 1 m}
    {t c : Fin 2 → ExactMatrix q N 1 m} {B : Nat} (h : Rel sm A t c B) {k : Nat} (hk : k < 2 * N) :
    Rel sm (fun a j ↦ multiplyMonomial (A a j) (k : Int))
      (fun j ↦ reducePoly q N (Xe (n := N) k) • t j) (fun j ↦ multiplyMonomial (c j) (k : Int)) B := by
  obtain ⟨e, he, hc⟩ := h
  refine ⟨fun j ↦ Xe (n := N) k • e j, fun j ↦ ?_, fun j ↦ ?_⟩
  · simpa using smulGain_root_pow hN1 k (he j)
  · simp only [multiplyMonomial_eq _ hk, hc j]
    funext r col
    simp only [Matrix.smul_apply, Matrix.add_apply, Matrix.sub_apply, smul_eq_mul, reduceMatrix,
      map_mul]
    ring

end RefreshTopProof
