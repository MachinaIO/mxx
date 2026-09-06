import ExponentRefreshRoutes

namespace MxxExponentLut
open Mxx.Primitives MxxRuntime

/-- The exact scalar and monomial expression emitted by the route constructor. -/
theorem route_target_expression {q n columns : Nat} (value : Int)
    (position : Fin n) (component : Fin columns) :
    matrixMul
      (matrixMulScalarLeft
        (matrixNeg (matrixMulScalarLeft (liftInteger value)
          (rotationPolynomial position.val))) (unitColumn 1))
      (unitRow component.val) =
      routeTarget (q := q) (-value) position component := by
  ext row column
  simp only [matrixMul, Matrix.mul_apply, Fin.sum_univ_one, matrixMulScalarLeft,
    matrixNeg, Matrix.neg_apply, liftInteger, rotationPolynomial, Int.toNat_natCast,
    unitColumn, unitRow, routeTarget, Int.cast_neg]
  have hr : (row.val : Int) = 1 ↔ row = (1 : Fin 2) := by
    constructor <;> intro h
    · apply Fin.ext; simpa using h
    · subst row; rfl
  have hc : (column.val : Int) = component.val ↔ column = component := by
    simp [Fin.ext_iff]
  simp only [hr, hc]
  split_ifs <;> simp_all

/-- Scaling preserves the exact route value, including nonunit CRT lifts. -/
theorem route_target_scale {q n columns : Nat} (scale value : Int)
    (position : Fin n) (component : Fin columns) :
    matrixMulScalarLeft (liftInteger scale)
      (routeTarget (q := q) value position component) =
      routeTarget (scale * value) position component := by
  ext row column
  simp only [matrixMulScalarLeft, liftInteger, routeTarget, Int.cast_mul]
  split_ifs <;> simp [mul_assoc]

theorem route_pack_at {α : Type} {size : Nat} {count : Int} {values : List α}
    {output : Fin size → α} (h : familyPack count values output) (i : Fin size) :
    output i = values[i.val]'(by rw [h.2.1]; exact i.isLt) := by
  obtain ⟨position, hp, he⟩ := h.2.2 i
  simpa only [List.get_eq_getElem, hp] using he

/-- A registered decomposition is the canonical integer action and reconstructs
    its target under any gadget with the same registered execution parameters. -/
theorem route_cache_canonical {q n columns : Nat} {backend : BackendContext}
    (layout : RegularLayout q) (lookup : backend.regularLayout q n = some layout)
    (hq : 1 < q) (hn : 0 < n)
    (width : columns = 2 * (layout.crtModuli.length * layout.digitsPerTower))
    {base digits : Int} {G : ExactMatrix q n 2 columns}
    {D : ExactMatrix q n columns columns} {target : ExactMatrix q n 2 columns}
    (hg : gadgetMatrixRuns backend base digits G)
    (hd : gadgetDecomposeRuns backend base digits target D) :
    D = reduceMatrix q n columns columns
      (fun row column => regularIntegerDigits layout target
        ⟨row.val, by rw [← width]; exact row.isLt⟩ column) ∧
    G * D = target := by
  obtain ⟨dl, hdl, _, _, hw, hd⟩ := hd
  have he : dl = layout := Option.some.inj (hdl.symm.trans lookup)
  subst dl
  obtain ⟨gl, hgl, _, _, hwg, hg⟩ := hg
  have he : gl = layout := Option.some.inj (hgl.symm.trans lookup)
  subst gl
  subst columns
  simp only [castMatrixRows] at hd
  simp only [castMatrixColumns] at hg
  subst D
  subst G
  exact ⟨regularIntegerDigits_reduce layout hq hn _,
    regularGadgetMatrix_reconstruct layout _ hq hn⟩

end MxxExponentLut
