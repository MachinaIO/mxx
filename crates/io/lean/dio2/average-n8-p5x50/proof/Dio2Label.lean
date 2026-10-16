import Dio2Rel
import Dio2Lut
import ExponentRuntime

/-!
# A label target of a refresh

A refresh part of a label pair (DIO2-SPEC §6.7) re-targets the label product to the ONEPUB target
`E^σ`: the public label member `1` (`α₁`) times the general member `3` (`α₃`) (`mul_public`,
whose keys the graph does not compute: they are rows of the product), acted on by the digits of
the gadget (`Rel.gamma`), then publicly times the rows of `E^σ` (`mul_public_split`).
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {n : Nat}

set_option maxHeartbeats 8000000 in
/-- **A product of labels.** The public encoding of `α₁` (`ha`) times the encoding of `α₃` (`hb`),
both against the gadget `G`, is an encoding of `α₁ α₃` against `G` (`mul_public`). -/
theorem label_mul (hn : 0 < n) {q m bound : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm α₁ α₃ : ErrorPoly n} (hα₁ : polyNorm α₁ ≤ 1)
    {Aa : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {ca : Fin 2 → ExactMatrix q n 1 m} {Ba : Nat}
    (ha : Rel sm Aa (fun j ↦ reducePoly q n (α₁ * unitPair sm j) • G) ca Ba)
    {Ab : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {cb : Fin 2 → ExactMatrix q n 1 m} {Bb : Nat}
    (hb : Rel sm Ab (fun j ↦ reducePoly q n (α₃ * unitPair sm j) • G) cb Bb)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows (ca 0) (Aa 0 0) l01) (hl : concatRows l01 (Aa 1 0) l)
    (hr01 : concatRows (ca 1) (Aa 0 1) r01) (hr : concatRows r01 (Aa 1 1) r)
    (hboth : concatColumns l r both)
    {Xs : Fin 2 → ExactMatrix q n (1 + 1) m} (hXs : ∀ j, concatRows (Ab 0 j) (Ab 1 j) (Xs j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xs j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {payload : ExactMatrix q n 1 1} (hpayload : payload = fun _ _ ↦ reducePoly q n α₁)
    {Y : Fin 2 → ExactMatrix q n 1 m} {A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    (hY : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m (Y j))
    (hA0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m (A' 0 j))
    (hA1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m (A' 1 j)) :
    Rel sm A' (fun j ↦ reducePoly q n (α₁ * α₃ * unitPair sm j) • G)
      (fun j ↦ Y j + matrixMulScalarRight (cb j) payload) ((m + m) * n * Ba * bound + n * 1 * Bb) := by
  have hpub := mul_public (bound := bound) hn hq1 layout hlayout hexact hG hα₁ ha hb hl01 hl hr01 hr
    hboth hXs hD' hD hpayload hY hA0 hA1
    (c' := fun j ↦ Y j + matrixMulScalarRight (cb j) payload) (fun _ ↦ rfl)
  exact hpub.target_eq (by funext j; rw [smul_smul, ← map_mul, mul_assoc])

set_option maxHeartbeats 8000000 in
/-- **The last product of labels**, whose keys the graph does not compute: they are rows `1, 2`
of the product. -/
theorem label_last (hn : 0 < n) {q m bound : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm α₁ α₃ : ErrorPoly n} (hα₁ : polyNorm α₁ ≤ 1)
    {Aa : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {ca : Fin 2 → ExactMatrix q n 1 m} {Ba : Nat}
    (ha : Rel sm Aa (fun j ↦ reducePoly q n (α₁ * unitPair sm j) • G) ca Ba)
    {Ab : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {cb : Fin 2 → ExactMatrix q n 1 m} {Bb : Nat}
    (hb : Rel sm Ab (fun j ↦ reducePoly q n (α₃ * unitPair sm j) • G) cb Bb)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows (ca 0) (Aa 0 0) l01) (hl : concatRows l01 (Aa 1 0) l)
    (hr01 : concatRows (ca 1) (Aa 0 1) r01) (hr : concatRows r01 (Aa 1 1) r)
    (hboth : concatColumns l r both)
    {Xs : Fin 2 → ExactMatrix q n (1 + 1) m} (hXs : ∀ j, concatRows (Ab 0 j) (Ab 1 j) (Xs j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xs j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {payload : ExactMatrix q n 1 1} (hpayload : payload = fun _ _ ↦ reducePoly q n α₁)
    {Y : Fin 2 → ExactMatrix q n 1 m}
    (hY : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m (Y j)) :
    Rel sm (fun k j ↦ rowOf (matrixMul both (D j)) k.succ)
      (fun j ↦ reducePoly q n (α₁ * α₃ * unitPair sm j) • G)
      (fun j ↦ Y j + matrixMulScalarRight (cb j) payload) ((m + m) * n * Ba * bound + n * 1 * Bb) :=
  label_mul hn hq1 layout hlayout hexact hG hα₁ ha hb hl01 hl hr01 hr hboth hXs hD' hD hpayload hY
    (fun j ↦ by simpa using sliceMatrix_rowOf (matrixMul both (D j)) (1 : Fin 3))
    (fun j ↦ by simpa using sliceMatrix_rowOf (matrixMul both (D j)) (2 : Fin 3))

set_option maxHeartbeats 8000000 in
/-- **A label target from its product.** The encoding `x` of the label product `α` (keys `K`)
acted on by the digits `Dg` of the gadget and publicly multiplied into `E^σ` (`hs`, targets `ts`)
encodes `α ts_j`. -/
theorem label_finish (hn : 0 < n) {q m : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm α : ErrorPoly n} (hα : polyNorm α ≤ 1)
    {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {x : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (hp : Rel sm K (fun j ↦ reducePoly q n (α * unitPair sm j) • G) x B)
    {Dg : ExactMatrix q n m m} (hDg : gadgetDecomposeRuns backend base digits G Dg)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {ts cs : Fin 2 → ExactMatrix q n 1 m} {Bs : Nat}
    (hs : Rel sm As ts cs Bs)
    {D0 D1 : Fin 2 → ExactMatrix q n m m}
    (hD0 : ∀ j, gadgetDecomposeRuns backend base digits (As 0 j) (D0 j))
    (hD1 : ∀ j, gadgetDecomposeRuns backend base digits (As 1 j) (D1 j)) :
    Rel sm (fun k j ↦ K k 0 * Dg * D0 j + K k 1 * Dg * D1 j) (fun j ↦ reducePoly q n α • ts j)
      (fun j ↦ x 0 * Dg * D0 j + x 1 * Dg * D1 j + reducePoly q n α • cs j)
      (m * n * (m * n * B * (base.toNat / 2)) * (base.toNat / 2) +
        m * n * (m * n * B * (base.toNat / 2)) * (base.toNat / 2) + n * 1 * Bs) :=
  mul_public_split hn hq1 layout hlayout hexact hG hα
    (Rel.gamma hn hq1 layout hlayout hexact hG hDg hp) hs hD0 hD1

/-- Equal rows keep the relation. -/
theorem Rel.c_eq {q m : Nat} {sm : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {t c c' : Fin 2 → ExactMatrix q n 1 m} {B : Nat} (h : Rel sm A t c B) (hc : c = c') :
    Rel sm A t c' B :=
  hc ▸ h

/-- A product of two constant `1 × 1` payloads is the constant of the product. -/
theorem payload_mul {q : Nat} {a b : ErrorPoly n} {x y : ExactMatrix q n 1 1}
    (hx : x = fun _ _ ↦ reducePoly q n a) (hy : y = fun _ _ ↦ reducePoly q n b) :
    matrixMulScalarLeft x y = fun _ _ ↦ reducePoly q n (a * b) := by
  subst hx hy
  funext r c
  rw [matrixMulScalarLeft_eq, map_mul]
  rfl

end Dio2Proof
