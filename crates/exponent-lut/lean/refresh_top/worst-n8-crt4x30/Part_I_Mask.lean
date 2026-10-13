import TopInst
import Part_I_Cells
import Part_I_Sums
import Part_I_Readout
import TopPrf

/-!
# The mask PRF of part `I`: member readouts

Member `o` of the mask PRF sums its five cells of bucket `0` (each the label member of its slot
times its selector bit), multiplies the sum by its five cells of bucket `1`, and reads the result
`X^(2 z_o)` with the readout LUT; one-hot selectors make the sums the selected label members.
-/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

/-- The bound of a bucket-`0` cell. -/
abbrev maskCell0Bound (Ba Bs BB : Nat) : Nat :=
  (((12 + 12) * N * ((14 + 14) * N * Ba * 16384 + N * 1 * Bs) * 16384 + (N + 2) * (p6 / 2)) / p6) +
    N * 1 * BB

/-- The bound of a bucket-`1` cell. -/
abbrev maskCell1Bound (Ba Bs BB : Nat) : Nat :=
  (((10 + 10) * N * ((12 + 12) * N * ((14 * N * Ba * (32768 / 2) + (N + 2) * (p6 / 2)) / p6) *
    16384 + N * 1 * Bs) * 16384 + (N + 2) * (p5 / 2)) / p5) + N * 1 * BB

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Member `o` of bucket `0`: an encoding of `p₅ X^(2 a_(sel₀ o))` at `Q6`. -/
theorem maskMember0 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q7 N 1 14}
    {apay : Fin 5 → ExactMatrix Q7 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q7 N 1 12}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 40 → ExactMatrix Q6 N 1 10}
    (h115 : ∀ i : Fin 40, Stage_online_eval.parallel_parallel_generatedRoot_290_115 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 8 → ExactMatrix Q6 N 1 10}
    (h116 : ∀ o : Fin 8, Stage_online_eval.parallel_parallel_generatedRoot_290_116 Backend.backend model
      params i0 o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {a : Fin 5 → Nat}
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 32768 14 G)
    {g6' : ExactMatrix Q6 N 1 12} {g7 : ExactMatrix Q7 N 1 12}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 12 g6') (hg7 : centeredRebaseRuns g6' g7)
    {h6 : ExactMatrix Q6 N 1 10} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q7 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q7 N (Xe (n := N) (2 * a k)))
    {Bs : Nat} (hone : Rel sm ![![one00, one01], ![one10, one11]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * 1 * unitPair sp j) • g7) ![onec0, onec1] Bs)
    {s : Fin 40 → Int} {eB : Fin 40 → Fin 2 → ErrorMatrix N 1 10} {BB : Nat}
    (hrhs : ∀ (i : Fin 40) j, reducePoly Q6 N sp • rhsTop model key Q6 10 (maskRhs0 i0) i j + ![bot0 i, bot1 i] j =
      reducePoly Q6 N (((p5 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q6 N 1 10 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (o : Fin 8) {sel : Fin 5} (hsel : ∀ k : Fin 5, s (cellIndex o k) = if k = sel then 1 else 0) :
    Rel sm ![![m0 o, m1 o], ![m2 o, m3 o]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * Xe (n := N) (2 * a sel) * unitPair sp j) • h6)
      ![m4 o, m5 o] (5 * maskCell0Bound Ba Bs BB) := by
  have hm := members116 (h116 o)
  simp only [Prod.mk.injEq] at hm
  obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := hm
  have hcell : ∀ k : Fin 5, Rel sm
      ![![f0 (cellIndex o k), f1 (cellIndex o k)], ![f2 (cellIndex o k), f3 (cellIndex o k)]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * ((Xe (n := N) (2 * a k) * 1) *
        ((s (cellIndex o k) : Int) : ErrorPoly N)) * unitPair sp j) • h6)
      ![f4 (cellIndex o k), f5 (cellIndex o k)] (maskCell0Bound Ba Bs BB) := by
    intro k
    have hc := mask_cell0 (h115 (cellIndex o k)) hsm (μ := 1) (α := fun k ↦ Xe (n := N) (2 * a k))
      (fun k ↦ by simpa using polyNorm_Xe hN1 (2 * a k)) (fun k ↦ polyNorm_Xe hN1 _) hG hg6' hg7
      ha hpay hone (hrhs (cellIndex o k)) (heB (cellIndex o k))
    have hk : (⟨(cellIndex o k).val % 5, Nat.mod_lt _ (by inst_num)⟩ : Fin 5) = k :=
      Fin.ext (cellIndex_mod o k)
    rw [hk] at hc
    exact hc
  have hsum := Rel.sum Finset.univ (fun k _ ↦ hcell k)
  rw [blocks_sum, pair_sum, onehot_collapse h6 sp p5 (fun k ↦ Xe (n := N) (2 * a k) * 1) _ sel hsel]
    at hsum
  rw [e0, e1, e2, e3, e4, e5]
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  simp

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Member `o` of bucket `1`: an encoding of `p₄ X^(2 z_o)` at `Q5`, `z_o = a_(sel₁ o) + a_(sel₀ o)`. -/
theorem maskMember1 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q7 N 1 14}
    {apay : Fin 5 → ExactMatrix Q7 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q7 N 1 12}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 40 → ExactMatrix Q6 N 1 10}
    (h115 : ∀ i : Fin 40, Stage_online_eval.parallel_parallel_generatedRoot_290_115 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 8 → ExactMatrix Q6 N 1 10}
    (h116 : ∀ o : Fin 8, Stage_online_eval.parallel_parallel_generatedRoot_290_116 Backend.backend model
      params i0 o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {bot0' bot1' g0 g1 g2 g3 g4 g5 : Fin 40 → ExactMatrix Q5 N 1 8}
    (h119 : ∀ i : Fin 40, Stage_online_eval.parallel_parallel_generatedRoot_290_119 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, m0, m2, m4, apay, m1, m3, m5, key, bot0' i,
        bot1' i, ()) (g0 i, g1 i, g2 i, g3 i, g4 i, g5 i, ()))
    {n0 n1 n2 n3 n4 n5 : Fin 8 → ExactMatrix Q5 N 1 8}
    (h120 : ∀ o : Fin 8, Stage_online_eval.parallel_parallel_generatedRoot_290_120 Backend.backend model
      params i0 o (g0, g1, g2, g3, g4, g5, ()) (n0 o, n1 o, n2 o, n3 o, n4 o, n5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {a : Fin 5 → Nat}
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 32768 14 G)
    {g6' : ExactMatrix Q6 N 1 12} {g7 : ExactMatrix Q7 N 1 12}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 12 g6') (hg7 : centeredRebaseRuns g6' g7)
    {g5' : ExactMatrix Q5 N 1 10} {h6 : ExactMatrix Q6 N 1 10}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 10 g5') (hh6 : centeredRebaseRuns g5' h6)
    {h5 : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q7 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q7 N (Xe (n := N) (2 * a k)))
    {Bs : Nat} (hone : Rel sm ![![one00, one01], ![one10, one11]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * 1 * unitPair sp j) • g7) ![onec0, onec1] Bs)
    {s : Fin 40 → Int} {eB : Fin 40 → Fin 2 → ErrorMatrix N 1 10} {BB : Nat}
    (hrhs : ∀ (i : Fin 40) j, reducePoly Q6 N sp • rhsTop model key Q6 10 (maskRhs0 i0) i j +
        ![bot0 i, bot1 i] j =
      reducePoly Q6 N (((p5 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q6 N 1 10 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 40 → Int} {eB' : Fin 40 → Fin 2 → ErrorMatrix N 1 8} {BB' : Nat}
    (hrhs' : ∀ (i : Fin 40) j, reducePoly Q5 N sp • rhsTop model key Q5 8 (maskRhs1 i0) i j +
        ![bot0' i, bot1' i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q5 N 1 8 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 8 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0) (o : Fin 8) :
    Rel sm ![![n0 o, n1 o], ![n2 o, n3 o]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * (a (sel' o) + a (sel o))) *
        unitPair sp j) • h5)
      ![n4 o, n5 o] (5 * maskCell1Bound Ba (5 * maskCell0Bound Ba Bs BB) BB') := by
  have hm := members120 (h120 o)
  simp only [Prod.mk.injEq] at hm
  obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := hm
  have hmem : ∀ o' : Fin 8, Rel sm ![![m0 o', m1 o'], ![m2 o', m3 o']]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * Xe (n := N) (2 * a (sel o')) * unitPair sp j) • h6)
      ![m4 o', m5 o'] (5 * maskCell0Bound Ba Bs BB) :=
    fun o' ↦ maskMember0 h115 h116 hsm hG hg6' hg7 ha hpay hone hrhs heB o' (hsel o')
  have hcell : ∀ k : Fin 5, Rel sm
      ![![g0 (cellIndex o k), g1 (cellIndex o k)], ![g2 (cellIndex o k), g3 (cellIndex o k)]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * ((Xe (n := N) (2 * a k) *
        Xe (n := N) (2 * a (sel o))) * ((s' (cellIndex o k) : Int) : ErrorPoly N)) * unitPair sp j) • h5)
      ![g4 (cellIndex o k), g5 (cellIndex o k)] (maskCell1Bound Ba (5 * maskCell0Bound Ba Bs BB) BB') := by
    intro k
    have hc := mask_cell1 (h119 (cellIndex o k)) hsm (μ := fun o' ↦ Xe (n := N) (2 * a (sel o')))
      (α := fun k ↦ Xe (n := N) (2 * a k)) (fun k o' ↦ polyNorm_Xe_mul _ _)
      (fun k ↦ polyNorm_Xe hN1 _) hG hg5' hh6 ha hpay hmem (hrhs' (cellIndex o k))
      (heB' (cellIndex o k))
    have hk : (⟨(cellIndex o k).val % 5, Nat.mod_lt _ (by inst_num)⟩ : Fin 5) = k :=
      Fin.ext (cellIndex_mod o k)
    have ho : (⟨(cellIndex o k).val / 5, by have := (cellIndex o k).isLt; omega⟩ : Fin 8) = o :=
      Fin.ext (cellIndex_div o k)
    rw [hk, ho] at hc
    exact hc
  have hsum := Rel.sum Finset.univ (fun k _ ↦ hcell k)
  rw [blocks_sum, pair_sum, onehot_collapse h5 sp p4
    (fun k ↦ Xe (n := N) (2 * a k) * Xe (n := N) (2 * a (sel o))) _ (sel' o) (hsel' o)] at hsum
  rw [e0, e1, e2, e3, e4, e5]
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  simp only [Xe_mul_Xe]
  ring_nf

/-- The readout exponent: `X^(2 z)` reads `(-1)^⌊z/4⌋ X^(2 τ(z mod 4))`, `τ = (1, 1, 3, 3)`. -/
def readoutExp (z : Nat) : Nat := ![1, 1, 3, 3] ⟨z % 4, Nat.mod_lt _ (by norm_num)⟩ + 4 * (z / 4)

theorem readout_value (z : Nat) :
    ((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) * readoutTable ⟨z % 4, Nat.mod_lt _ (by norm_num)⟩) =
      Xe (n := N) (2 * readoutExp z) := by
  have h8 : AdjoinRoot.root (negacyclicModulus N Int) ^ 8 = -1 := Negacyclic.root_pow_n
  unfold readoutExp readoutTable
  rw [Nat.mul_add, show 2 * (4 * (z / 4)) = 8 * (z / 4) by ring]
  simp only [Xe]
  rw [pow_add, pow_mul _ 8 (z / 4), h8]
  push_cast
  ring

/-- The bound of a member's readout. -/
abbrev maskReadoutBound (Ba Bs BB BB' Bh BS : Nat) : Nat :=
  (3 * lutBranchBound121 (5 * maskCell1Bound Ba (5 * maskCell0Bound Ba Bs BB) BB') Bh +
    ((8 + 8) * N * Bh * 16384 + (8 + 8) * N * (5 * maskCell1Bound Ba (5 * maskCell0Bound Ba Bs BB) BB') * 16384 +
      (N * 1 * (p4 / 2) + p4 / 2)) + (N + 2) * (p4 / 2)) / p4 + 4 * (N * 1 * BS)

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The readout of member `o`: an encoding of `q₃ X^(2 τ'(z_o))` at `Q0`. -/
theorem maskReadout {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q7 N 1 14}
    {apay : Fin 5 → ExactMatrix Q7 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q7 N 1 12}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 40 → ExactMatrix Q6 N 1 10}
    (h115 : ∀ i : Fin 40, Stage_online_eval.parallel_parallel_generatedRoot_290_115 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 8 → ExactMatrix Q6 N 1 10}
    (h116 : ∀ o : Fin 8, Stage_online_eval.parallel_parallel_generatedRoot_290_116 Backend.backend model
      params i0 o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {bot0' bot1' g0 g1 g2 g3 g4 g5 : Fin 40 → ExactMatrix Q5 N 1 8}
    (h119 : ∀ i : Fin 40, Stage_online_eval.parallel_parallel_generatedRoot_290_119 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, m0, m2, m4, apay, m1, m3, m5, key, bot0' i,
        bot1' i, ()) (g0 i, g1 i, g2 i, g3 i, g4 i, g5 i, ()))
    {n0 n1 n2 n3 n4 n5 : Fin 8 → ExactMatrix Q5 N 1 8}
    (h120 : ∀ o : Fin 8, Stage_online_eval.parallel_parallel_generatedRoot_290_120 Backend.backend model
      params i0 o (g0, g1, g2, g3, g4, g5, ()) (n0 o, n1 o, n2 o, n3 o, n4 o, n5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {a : Fin 5 → Nat}
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 32768 14 G)
    {g6' : ExactMatrix Q6 N 1 12} {g7 : ExactMatrix Q7 N 1 12}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 12 g6') (hg7 : centeredRebaseRuns g6' g7)
    {g5' : ExactMatrix Q5 N 1 10} {h6 : ExactMatrix Q6 N 1 10}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 10 g5') (hh6 : centeredRebaseRuns g5' h6)
    {h5 : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q7 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q7 N (Xe (n := N) (2 * a k)))
    {Bs : Nat} (hone : Rel sm ![![one00, one01], ![one10, one11]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * 1 * unitPair sp j) • g7) ![onec0, onec1] Bs)
    {s : Fin 40 → Int} {eB : Fin 40 → Fin 2 → ErrorMatrix N 1 10} {BB : Nat}
    (hrhs : ∀ (i : Fin 40) j, reducePoly Q6 N sp • rhsTop model key Q6 10 (maskRhs0 i0) i j +
        ![bot0 i, bot1 i] j =
      reducePoly Q6 N (((p5 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q6 N 1 10 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 40 → Int} {eB' : Fin 40 → Fin 2 → ErrorMatrix N 1 8} {BB' : Nat}
    (hrhs' : ∀ (i : Fin 40) j, reducePoly Q5 N sp • rhsTop model key Q5 8 (maskRhs1 i0) i j +
        ![bot0' i, bot1' i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q5 N 1 8 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 8 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q0 N 1 6} {hl0 hl1 : Fin 4 → ExactMatrix Q5 N 1 8}
    {r0 r1 r2 r3 r4 r5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h121 : ∀ o : Fin 8, Stage_online_eval.parallel_parallel_generatedRoot_290_121 Backend.backend model
      params i0 o (fam00, fam01, fam10, fam11, hl0, key, hl1, n4 o, n0 o, n2 o, n5 o, n1 o, n3 o, ())
        (r0 o, r1 o, r2 o, r3 o, r4 o, r5 o, ()))
    {g0' : ExactMatrix Q0 N 1 8} (hg0' : gadgetMatrixRuns Backend.backend 32768 8 g0')
    (hh5 : centeredRebaseRuns g0' h5)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![hl0 i, hl1 i] Bh)
    {gq : ExactMatrix Q0 N 1 6} {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q0 N) * ((4 : Nat) : ExactPoly Q0 N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q0 inverse (fun k ↦ reducePoly Q0 N (readoutTable k)) (sigma i).toNat *
          reducePoly Q0 N (((((qd i0) : Nat) : Int) : ErrorPoly N) * unitPair sp j)) • gq +
            reduceMatrix Q0 N 1 6 (e j))
    (o : Fin 8) :
    Rel sm ![![r0 o, r1 o], ![r2 o, r3 o]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * readoutExp (a (sel' o) + a (sel o))) *
        unitPair sp j) • gq)
      ![r4 o, r5 o] (maskReadoutBound Ba Bs BB BB' Bh BS) := by
  have hm := maskMember1 h115 h116 h119 h120 hsm hG hg6' hg7 hg5' hh6 ha hpay hone hrhs heB hrhs'
    heB' hsel hsel' o
  have hr := lut121 (h121 o) hsm hg0' hh5 hm hhelp hinverse hswitch
  refine hr.target_eq ?_
  funext j
  rw [← readout_value]
  try push_cast
  try ring_nf

end RefreshTopProof.Part_I
