import ExponentEncoding
import ExponentPublic

/-!
# Encoding relations under the operations of a Diamond iO 2 refresh

`Gamma` (`RnsBggParameters::retarget_encoding`) acts on every component row of an encoding and on
every key block by the same small matrix `D` (the digits of the target form); without a modulus
drop it is exactly this action.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {n q m : Nat} {sm : ErrorPoly n}

/-- A right action by a small matrix `d`, the same on every component, keeps the relation: the
targets and keys are acted on alike and the error bound gains `m n β`. -/
theorem Rel.mapEach (hn : 0 < n) {m' : Nat} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {t c : Fin 2 → ExactMatrix q n 1 m} {B β : Nat} (h : Rel sm A t c B)
    (d : ErrorMatrix n m m') (hd : CoeffBound d β) :
    Rel sm (fun k j ↦ A k j * reduceMatrix q n m m' d) (fun j ↦ t j * reduceMatrix q n m m' d)
      (fun j ↦ c j * reduceMatrix q n m m' d) (m * n * B * β) := by
  obtain ⟨e, he, hc⟩ := h
  refine ⟨fun j ↦ e j * d, fun j ↦ coeffBound_mul hn (he j) hd, fun j ↦ ?_⟩
  beta_reduce
  rw [hc j, reduceMatrix_mul, Matrix.add_mul, Matrix.sub_mul, Matrix.add_mul, Matrix.smul_mul]

end Dio2Proof

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {n q m : Nat} {sm : ErrorPoly n}

/-- `Gamma` without a modulus drop: an encoding of the targets `T_j G` in the form of the gadget
`G` acted on by the digits `D = G^-1(X)` of a row `X` is an encoding of `T_j X`, its keys acted on
alike, with the error bound multiplied by `m n (base / 2)`. -/
theorem Rel.gamma (hn : 0 < n) (hq : 1 < q) {m' : Nat} {backend : BackendContext}
    {base digits : Int} (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0) {G : ExactMatrix q n 1 m}
    (hG : gadgetMatrixRuns backend base digits G) {X : ExactMatrix q n 1 m'}
    {D : ExactMatrix q n m m'} (hD : gadgetDecomposeRuns backend base digits X D)
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {T : Fin 2 → ExactPoly q n}
    {c : Fin 2 → ExactMatrix q n 1 m} {B : Nat} (h : Rel sm A (fun j ↦ T j • G) c B) :
    Rel sm (fun k j ↦ A k j * D) (fun j ↦ T j • X) (fun j ↦ c j * D)
      (m * n * B * (base.toNat / 2)) := by
  obtain ⟨d, hd, hdb⟩ := gadgetDecomposeRuns_small hq hn hD
  have hGD := gadget_mul_decompose hq hn layout hlayout hexact hG hD
  have hmap := Rel.mapEach hn h d hdb
  rw [← hd] at hmap
  refine Rel.target_eq hmap (funext fun j ↦ ?_)
  beta_reduce
  rw [Matrix.smul_mul, hGD]

set_option maxHeartbeats 8000000 in
/-- **The raw-row public product** (`RnsBggParameters::mul_public_rows`), with each key block
decomposed on its own: the public encoding `x` of `α` (gadget `G`) times the general row `(As, cs)`
of `ts` is `x₀ G⁻¹(As_{0j}) + x₁ G⁻¹(As_{1j}) + α cs_j`, an encoding of `α ts_j` with keys
`Aa_{k0} G⁻¹(As_{0j}) + Aa_{k1} G⁻¹(As_{1j})`. -/
theorem mul_public_split (hn : 0 < n) {q m m' : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm α : ErrorPoly n} (hα : polyNorm α ≤ 1)
    {Aa : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {x : Fin 2 → ExactMatrix q n 1 m} {Ba : Nat}
    (ha : Rel sm Aa (fun j ↦ reducePoly q n (α * unitPair sm j) • G) x Ba)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {ts cs : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hs : Rel sm As ts cs Bs)
    {D0 D1 : Fin 2 → ExactMatrix q n m m'}
    (hD0 : ∀ j, gadgetDecomposeRuns backend base digits (As 0 j) (D0 j))
    (hD1 : ∀ j, gadgetDecomposeRuns backend base digits (As 1 j) (D1 j)) :
    Rel sm (fun k j ↦ Aa k 0 * D0 j + Aa k 1 * D1 j) (fun j ↦ reducePoly q n α • ts j)
      (fun j ↦ x 0 * D0 j + x 1 * D1 j + reducePoly q n α • cs j)
      (m * n * Ba * (base.toNat / 2) + m * n * Ba * (base.toNat / 2) + n * 1 * Bs) := by
  obtain ⟨ea, hea, hx⟩ := ha
  obtain ⟨es, hes, hc⟩ := hs
  choose d0 hd0 hd0b using fun j ↦ gadgetDecomposeRuns_small hq1 hn (hD0 j)
  choose d1 hd1 hd1b using fun j ↦ gadgetDecomposeRuns_small hq1 hn (hD1 j)
  have hG0 : ∀ j, G * D0 j = As 0 j := fun j ↦ gadget_mul_decompose hq1 hn layout hlayout hexact hG (hD0 j)
  have hG1 : ∀ j, G * D1 j = As 1 j := fun j ↦ gadget_mul_decompose hq1 hn layout hlayout hexact hG (hD1 j)
  refine ⟨fun j ↦ ea 0 * d0 j + ea 1 * d1 j + α • es j, fun j ↦ coeffBound_add (coeffBound_add
    (coeffBound_mul hn (hea 0) (hd0b j)) (coeffBound_mul hn (hea 1) (hd1b j)))
    (coeffBound_smul hn hα (hes j)), fun j ↦ ?_⟩
  beta_reduce
  rw [hx 0, hx 1, hc j, reduceMatrix_add, reduceMatrix_add, reduceMatrix_mul, reduceMatrix_mul,
    ← hd0 j, ← hd1 j, reduceMatrix_smul]
  simp only [Matrix.add_mul, Matrix.sub_mul, Matrix.smul_mul, hG0, hG1, unitPair,
    Matrix.cons_val_zero, Matrix.cons_val_one, map_mul, mul_one, smul_add, smul_sub,
    smul_smul]
  module

end Dio2Proof
