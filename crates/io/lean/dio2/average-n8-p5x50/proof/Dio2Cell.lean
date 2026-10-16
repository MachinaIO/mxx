import ExponentLut
import ExponentPublic

/-!
# A PRF cell without a modulus drop

A dio2 chain keeps one modulus `Q_0` at every level, so the cell of a PRF (a label member `α`
times the member's state encoding of `μ`, then times the selector right-hand side) is the public
multiplication `mul_public` followed by the private multiplication `mul_private` at the same
modulus, where refresh_top's cell (`Gen.cell_mul_gen`) moves down after the private one.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {n : Nat}

set_option maxHeartbeats 4000000 in
/-- **One PRF cell at one modulus.** The label encoding `x` of `α` (gadget `G`, payload
`xpay = α`) is multiplied publicly into the state encoding `(As, cs)` of `μ` (gadget `g`) and the
product privately by the right-hand side `(top, bot)` of `T` (`hrhs`): an encoding of
`(α μ) T`. -/
theorem cell_flat (hn : 0 < n) {q m m' m'' bound : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {base digits base' digits' : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    {sm sp μ α : ErrorPoly n}
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {x0 x1 xA00 xA10 xA01 xA11 : ExactMatrix q n 1 m} {xpay : ExactMatrix q n 1 1} {Ba : Nat}
    (hα : polyNorm α ≤ 1) (hαμ : polyNorm (α * μ) ≤ 1)
    (ha : Rel sm ![![xA00, xA01], ![xA10, xA11]]
      (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ![x0, x1] Ba)
    (hpay : xpay = fun _ _ ↦ reducePoly q n α)
    {g' : ExactMatrix q n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {cs : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hs : Rel sm As (fun j ↦ reducePoly q n (μ * unitPair sp j) • g) cs Bs)
    -- the public multiplication
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows x0 xA00 l01) (hl : concatRows l01 xA10 l)
    (hr01 : concatRows x1 xA01 r01) (hr : concatRows r01 xA11 r) (hboth : concatColumns l r both)
    {Xs : Fin 2 → ExactMatrix q n (1 + 1) m'} (hXs : ∀ j, concatRows (As 0 j) (As 1 j) (Xs j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xs j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {Y : Fin 2 → ExactMatrix q n 1 m'} {A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (Y j))
    (hA0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (A' 0 j))
    (hA1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (A' 1 j))
    {c' : Fin 2 → ExactMatrix q n 1 m'}
    (hc' : ∀ j, c' j = Y j + matrixMulScalarRight (cs j) xpay)
    -- the private multiplication
    {l01' r01' : ExactMatrix q n (1 + 1) m'} {l' r' : ExactMatrix q n (1 + 1 + 1) m'}
    {both' : ExactMatrix q n (1 + 1 + 1) (m' + m')}
    (hl01' : concatRows (c' 0) (A' 0 0) l01') (hl' : concatRows l01' (A' 1 0) l')
    (hr01' : concatRows (c' 1) (A' 0 1) r01') (hr' : concatRows r01' (A' 1 1) r')
    (hboth' : concatColumns l' r' both')
    {top bot : Fin 2 → ExactMatrix q n 1 m''} {Xb : Fin 2 → ExactMatrix q n (1 + 1) m''}
    (hXb : ∀ j, concatRows (top j) (bot j) (Xb j))
    {E' : Fin 2 → ExactMatrix q n (m' + m') m''}
    (hE' : ∀ j, gadgetDecomposeRuns backend base' digits' (Xb j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m' + m') m''} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {T : Fin 2 → ExactMatrix q n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q n sp • top j + bot j = T j + reduceMatrix q n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    {X : Fin 2 → ExactMatrix q n 1 m''} {Mk : Fin 2 → Fin 2 → ExactMatrix q n 1 m''}
    (hX : ∀ j, sliceMatrix (matrixMul both' (E j)) 0 1 0 m'' (X j))
    (hM0 : ∀ j, sliceMatrix (matrixMul both' (E j)) 1 2 0 m'' (Mk 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both' (E j)) 2 3 0 m'' (Mk 1 j)) :
    Rel sm Mk (fun j ↦ reducePoly q n (α * μ) • T j) X
      ((m' + m') * n * ((m + m) * n * Ba * bound + n * 1 * Bs) * bound + n * 1 * BB) := by
  have hpub := mul_public (m := m) (m' := m') (bound := bound) hn hq1 layout hlayout hexact hG
    hα ha hs hl01 hl hr01 hr hboth hXs hD' hD hpay hY hA0 hA1 (c' := c') (fun j ↦ hc' j)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly q n (((1 : Nat) : Int) * (α * μ) *
      unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    push_cast
    ring)
  exact mul_private hn hq1 layout hlayout hexact hg' hg hpub' hl01' hl' hr01' hr' hboth' hXb hE'
    hE hrhs heB hαμ hX hM0 hM1

end Dio2Proof
