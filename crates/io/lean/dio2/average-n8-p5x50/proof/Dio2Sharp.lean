import Dio2Cell
import Dio2Lut
import Dio2Rel

/-!
# Gain-aware same-modulus noise bounds

The conservative recurrence in a3's `recompute.py` changes only the public plaintext action
inside a PRF cell from `n * Bs` to `Bs`. `cell_flat_gain` and `cell_flat_mono` expose that
parameter-generic improvement without changing any existing lemma. The action hypothesis is
`SmulGain`, not a coefficient bound: a dense sign polynomial need not have gain one.

Signed-monomial and product closure lemmas discharge this hypothesis from algebraic plaintext
identities. The label-product analogues also retain gain one on the right error. The passing
recurrence does not require sharper private fresh terms, LUT terms, constant retarget actions,
or prime-combination target terms, so those bounds are left unchanged here.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {n : Nat}

/-- Negating a plaintext preserves its coefficient-bound action gain. -/
theorem smulGain_neg {α : ErrorPoly n} {K : Nat} (hα : SmulGain α K) :
    SmulGain (-α) K := by
  intro rows columns error bound herror
  simpa only [neg_smul] using coeffBound_neg (hα herror)

/-- A signed power of the negacyclic root has action gain one at every matrix size. -/
theorem smulGain_signed_root_pow (hn : 1 < n) {α : ErrorPoly n} (exponent : Nat)
    (hα : α = AdjoinRoot.root (negacyclicModulus n Int) ^ exponent ∨
      α = -(AdjoinRoot.root (negacyclicModulus n Int) ^ exponent)) : SmulGain α 1 := by
  rcases hα with rfl | rfl
  · exact smulGain_root_pow hn exponent
  · exact smulGain_neg (smulGain_root_pow hn exponent)

/-- Products compose plaintext action gains, so signed-monomial label products keep gain one. -/
theorem smulGain_mul {α β : ErrorPoly n} {K L : Nat}
    (hα : SmulGain α K) (hβ : SmulGain β L) : SmulGain (α * β) (K * L) := by
  intro rows columns error bound herror
  simpa only [smul_smul, Nat.mul_assoc] using hα (hβ herror)

set_option maxHeartbeats 4000000 in
/-- A same-modulus PRF cell with public plaintext action gain `K`. Only the public
state-error term changes; both dense digit actions and the private fresh term are retained. -/
theorem cell_flat_gain (hn : 0 < n) {q m m' m'' bound K : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {base digits base' digits' : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    {sm sp μ α : ErrorPoly n}
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {x0 x1 xA00 xA10 xA01 xA11 : ExactMatrix q n 1 m} {xpay : ExactMatrix q n 1 1} {Ba : Nat}
    (hα : SmulGain α K) (hαμ : polyNorm (α * μ) ≤ 1)
    (ha : Rel sm ![![xA00, xA01], ![xA10, xA11]]
      (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ![x0, x1] Ba)
    (hpay : xpay = fun _ _ ↦ reducePoly q n α)
    {g' : ExactMatrix q n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {cs : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hs : Rel sm As (fun j ↦ reducePoly q n (μ * unitPair sp j) • g) cs Bs)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows x0 xA00 l01) (hl : concatRows l01 xA10 l)
    (hr01 : concatRows x1 xA01 r01) (hr : concatRows r01 xA11 r) (hboth : concatColumns l r both)
    {Xs : Fin 2 → ExactMatrix q n (1 + 1) m'} (hXs : ∀ j, concatRows (As 0 j) (As 1 j) (Xs j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xs j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {Y : Fin 2 → ExactMatrix q n 1 m'} {A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (Y j))
    (hA0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (A' 0 j))
    (hA1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (A' 1 j))
    {c' : Fin 2 → ExactMatrix q n 1 m'}
    (hc' : ∀ j, c' j = Y j + matrixMulScalarRight (cs j) xpay)
    {l01' r01' : ExactMatrix q n (1 + 1) m'} {l' r' : ExactMatrix q n (1 + 1 + 1) m'}
    {both' : ExactMatrix q n (1 + 1 + 1) (m' + m')}
    (hl01' : concatRows (c' 0) (A' 0 0) l01') (hl' : concatRows l01' (A' 1 0) l')
    (hr01' : concatRows (c' 1) (A' 0 1) r01') (hr' : concatRows r01' (A' 1 1) r')
    (hboth' : concatColumns l' r' both')
    {top bot : Fin 2 → ExactMatrix q n 1 m''} {Xb : Fin 2 → ExactMatrix q n (1 + 1) m''}
    (hXb : ∀ j, concatRows (top j) (bot j) (Xb j))
    {E' : Fin 2 → ExactMatrix q n (m' + m') m''}
    (hE' : ∀ j, gadgetDecomposeRuns backend base' digits' (Xb j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m' + m') m''}
    (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {T : Fin 2 → ExactMatrix q n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q n sp • top j + bot j = T j + reduceMatrix q n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    {X : Fin 2 → ExactMatrix q n 1 m''} {Mk : Fin 2 → Fin 2 → ExactMatrix q n 1 m''}
    (hX : ∀ j, sliceMatrix (matrixMul both' (E j)) 0 1 0 m'' (X j))
    (hM0 : ∀ j, sliceMatrix (matrixMul both' (E j)) 1 2 0 m'' (Mk 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both' (E j)) 2 3 0 m'' (Mk 1 j)) :
    Rel sm Mk (fun j ↦ reducePoly q n (α * μ) • T j) X
      ((m' + m') * n * ((m + m) * n * Ba * bound + K * Bs) * bound + n * 1 * BB) := by
  have hpub := mul_public_gain (m := m) (m' := m') (bound := bound) hn hq1 layout hlayout hexact hG
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

/-- A signed-monomial label has public action gain one, giving
`A' * (A * Ba + Bs) + n * BB`. The hypothesis is the exact coefficient-bound action property;
`smulGain_signed_root_pow` supplies it for signed monomials. No private fresh-term saving or
inactive-selector error cancellation is assumed. -/
theorem cell_flat_mono (hn : 0 < n) {q m m' m'' bound : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {base digits base' digits' : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    {sm sp μ α : ErrorPoly n}
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {x0 x1 xA00 xA10 xA01 xA11 : ExactMatrix q n 1 m} {xpay : ExactMatrix q n 1 1} {Ba : Nat}
    (hα : SmulGain α 1) (hαμ : polyNorm (α * μ) ≤ 1)
    (ha : Rel sm ![![xA00, xA01], ![xA10, xA11]]
      (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ![x0, x1] Ba)
    (hpay : xpay = fun _ _ ↦ reducePoly q n α)
    {g' : ExactMatrix q n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {cs : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hs : Rel sm As (fun j ↦ reducePoly q n (μ * unitPair sp j) • g) cs Bs)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows x0 xA00 l01) (hl : concatRows l01 xA10 l)
    (hr01 : concatRows x1 xA01 r01) (hr : concatRows r01 xA11 r) (hboth : concatColumns l r both)
    {Xs : Fin 2 → ExactMatrix q n (1 + 1) m'} (hXs : ∀ j, concatRows (As 0 j) (As 1 j) (Xs j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xs j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {Y : Fin 2 → ExactMatrix q n 1 m'} {A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (Y j))
    (hA0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (A' 0 j))
    (hA1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (A' 1 j))
    {c' : Fin 2 → ExactMatrix q n 1 m'}
    (hc' : ∀ j, c' j = Y j + matrixMulScalarRight (cs j) xpay)
    {l01' r01' : ExactMatrix q n (1 + 1) m'} {l' r' : ExactMatrix q n (1 + 1 + 1) m'}
    {both' : ExactMatrix q n (1 + 1 + 1) (m' + m')}
    (hl01' : concatRows (c' 0) (A' 0 0) l01') (hl' : concatRows l01' (A' 1 0) l')
    (hr01' : concatRows (c' 1) (A' 0 1) r01') (hr' : concatRows r01' (A' 1 1) r')
    (hboth' : concatColumns l' r' both')
    {top bot : Fin 2 → ExactMatrix q n 1 m''} {Xb : Fin 2 → ExactMatrix q n (1 + 1) m''}
    (hXb : ∀ j, concatRows (top j) (bot j) (Xb j))
    {E' : Fin 2 → ExactMatrix q n (m' + m') m''}
    (hE' : ∀ j, gadgetDecomposeRuns backend base' digits' (Xb j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m' + m') m''}
    (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {T : Fin 2 → ExactMatrix q n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q n sp • top j + bot j = T j + reduceMatrix q n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    {X : Fin 2 → ExactMatrix q n 1 m''} {Mk : Fin 2 → Fin 2 → ExactMatrix q n 1 m''}
    (hX : ∀ j, sliceMatrix (matrixMul both' (E j)) 0 1 0 m'' (X j))
    (hM0 : ∀ j, sliceMatrix (matrixMul both' (E j)) 1 2 0 m'' (Mk 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both' (E j)) 2 3 0 m'' (Mk 1 j)) :
    Rel sm Mk (fun j ↦ reducePoly q n (α * μ) • T j) X
      ((m' + m') * n * ((m + m) * n * Ba * bound + Bs) * bound + n * 1 * BB) := by
  simpa only [one_mul] using
    cell_flat_gain hn hq1 layout hlayout hexact hG hα hαμ ha hpay hg' hg hs
      hl01 hl hr01 hr hboth hXs hD' hD hY hA0 hA1 hc'
      hl01' hl' hr01' hr' hboth' hXb hE' hE hrhs heB hX hM0 hM1

set_option maxHeartbeats 8000000 in
/-- The raw-row public product with a gain-one plaintext: both digit actions retain their
usual dense bounds, while the general row's error contributes only `Bs`. -/
theorem mul_public_split_mono (hn : 0 < n) {q m m' : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm α : ErrorPoly n} (hα : SmulGain α 1)
    {Aa : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {x : Fin 2 → ExactMatrix q n 1 m} {Ba : Nat}
    (ha : Rel sm Aa (fun j ↦ reducePoly q n (α * unitPair sm j) • G) x Ba)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {ts cs : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hs : Rel sm As ts cs Bs)
    {D0 D1 : Fin 2 → ExactMatrix q n m m'}
    (hD0 : ∀ j, gadgetDecomposeRuns backend base digits (As 0 j) (D0 j))
    (hD1 : ∀ j, gadgetDecomposeRuns backend base digits (As 1 j) (D1 j)) :
    Rel sm (fun k j ↦ Aa k 0 * D0 j + Aa k 1 * D1 j) (fun j ↦ reducePoly q n α • ts j)
      (fun j ↦ x 0 * D0 j + x 1 * D1 j + reducePoly q n α • cs j)
      (m * n * Ba * (base.toNat / 2) + m * n * Ba * (base.toNat / 2) + Bs) := by
  obtain ⟨ea, hea, hx⟩ := ha
  obtain ⟨es, hes, hc⟩ := hs
  choose d0 hd0 hd0b using fun j ↦ gadgetDecomposeRuns_small hq1 hn (hD0 j)
  choose d1 hd1 hd1b using fun j ↦ gadgetDecomposeRuns_small hq1 hn (hD1 j)
  have hG0 : ∀ j, G * D0 j = As 0 j :=
    fun j ↦ gadget_mul_decompose hq1 hn layout hlayout hexact hG (hD0 j)
  have hG1 : ∀ j, G * D1 j = As 1 j :=
    fun j ↦ gadget_mul_decompose hq1 hn layout hlayout hexact hG (hD1 j)
  refine ⟨fun j ↦ ea 0 * d0 j + ea 1 * d1 j + α • es j, fun j ↦ coeffBound_add (coeffBound_add
    (coeffBound_mul hn (hea 0) (hd0b j)) (coeffBound_mul hn (hea 1) (hd1b j)))
    (by simpa only [one_mul] using hα (hes j)), fun j ↦ ?_⟩
  beta_reduce
  rw [hx 0, hx 1, hc j, reduceMatrix_add, reduceMatrix_add, reduceMatrix_mul, reduceMatrix_mul,
    ← hd0 j, ← hd1 j, reduceMatrix_smul]
  simp only [Matrix.add_mul, Matrix.sub_mul, Matrix.smul_mul, hG0, hG1, unitPair,
    Matrix.cons_val_zero, Matrix.cons_val_one, map_mul, mul_one, smul_add, smul_sub,
    smul_smul]
  module

set_option maxHeartbeats 8000000 in
/-- **A product of labels.** The public encoding of `α₁` (`ha`) times the encoding of `α₃` (`hb`),
both against the gadget `G`, is an encoding of `α₁ α₃` against `G`, with right-error gain one. -/
theorem label_mul_mono (hn : 0 < n) {q m bound : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm α₁ α₃ : ErrorPoly n} (hα₁ : SmulGain α₁ 1)
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
      (fun j ↦ Y j + matrixMulScalarRight (cb j) payload) ((m + m) * n * Ba * bound + Bb) := by
  have hpub := mul_public_gain (bound := bound) hn hq1 layout hlayout hexact hG hα₁ ha hb
    hl01 hl hr01 hr
    hboth hXs hD' hD hpayload hY hA0 hA1
    (c' := fun j ↦ Y j + matrixMulScalarRight (cb j) payload) (fun _ ↦ rfl)
  simpa only [one_mul] using hpub.target_eq (by funext j; rw [smul_smul, ← map_mul, mul_assoc])

set_option maxHeartbeats 8000000 in
/-- **The last product of labels**, whose keys the graph does not compute: they are rows `1, 2`
of the product. The monomial action keeps the right-error gain one. -/
theorem label_last_mono (hn : 0 < n) {q m bound : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm α₁ α₃ : ErrorPoly n} (hα₁ : SmulGain α₁ 1)
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
      (fun j ↦ Y j + matrixMulScalarRight (cb j) payload) ((m + m) * n * Ba * bound + Bb) :=
  label_mul_mono hn hq1 layout hlayout hexact hG hα₁ ha hb
    hl01 hl hr01 hr hboth hXs hD' hD hpayload hY
    (fun j ↦ by simpa using sliceMatrix_rowOf (matrixMul both (D j)) (1 : Fin 3))
    (fun j ↦ by simpa using sliceMatrix_rowOf (matrixMul both (D j)) (2 : Fin 3))

set_option maxHeartbeats 8000000 in
/-- **A label target from its product.** The encoding `x` of the label product `α` (keys `K`)
acted on by the digits `Dg` of the gadget and publicly multiplied into `E^σ` (`hs`, targets `ts`)
encodes `α ts_j`, with mask-error gain one. The two retarget actions remain dense. -/
theorem label_finish_mono (hn : 0 < n) {q m : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm α : ErrorPoly n} (hα : SmulGain α 1)
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
        m * n * (m * n * B * (base.toNat / 2)) * (base.toNat / 2) + Bs) :=
  mul_public_split_mono hn hq1 layout hlayout hexact hG hα
    (Rel.gamma hn hq1 layout hlayout hexact hG hDg hp) hs hD0 hD1

end Dio2Proof
