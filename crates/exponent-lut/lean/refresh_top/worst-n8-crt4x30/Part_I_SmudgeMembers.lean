import TopInst
import Part_I_Mask

/-! The member sums of every decoder's smudge PRF (written by `generator/gen_smudge_members.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

/-- The bound of a bucket-`0` cell. -/
abbrev smudge0Cell0Bound (Ba Bs BB : Nat) : Nat :=
  (((10 + 10) * N * ((12 + 12) * N * Ba * 16384 + N * 1 * Bs) * 16384 + (N + 2) * (p5 / 2)) / p5) +
    N * 1 * BB

/-- The bound of a bucket-`1` cell. -/
abbrev smudge0Cell1Bound (Ba Bs BB : Nat) : Nat :=
  (((8 + 8) * N * ((10 + 10) * N * ((12 * N * Ba * (32768 / 2) + (N + 2) * (p5 / 2)) / p5) *
    16384 + N * 1 * Bs) * 16384 + (N + 2) * (p4 / 2)) / p4) + N * 1 * BB

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Member `o` of bucket `0`: an encoding of `p₅ X^(2 a_(sel₀ o))` at `Q5`. -/
theorem smudge0Member0 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h554 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_556 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h555 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_557 Backend.backend model
      params i0 o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10} {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6') (hg7 : centeredRebaseRuns g6' g7)
    {h6 : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat} (hone : Rel sm ![![one00, one01], ![one10, one11]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![onec0, onec1] Bs)
    {s : Fin 800 → Int} {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 (smudge0Rhs0 i0) i j + ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (o : Fin 160) {sel : Fin 5} (hsel : ∀ k : Fin 5, s (cellIndex o k) = if k = sel then 1 else 0) :
    Rel sm ![![m0 o, m1 o], ![m2 o, m3 o]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * a sel) * unitPair sp j) • h6)
      ![m4 o, m5 o] (5 * smudge0Cell0Bound Ba Bs BB) := by
  have hm := members555 (h555 o)
  simp only [Prod.mk.injEq] at hm
  obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := hm
  have hcell : ∀ k : Fin 5, Rel sm
      ![![f0 (cellIndex o k), f1 (cellIndex o k)], ![f2 (cellIndex o k), f3 (cellIndex o k)]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * ((Xe (n := N) (2 * a k) * 1) *
        ((s (cellIndex o k) : Int) : ErrorPoly N)) * unitPair sp j) • h6)
      ![f4 (cellIndex o k), f5 (cellIndex o k)] (smudge0Cell0Bound Ba Bs BB) := by
    intro k
    have hc := smudge0_cell0 (h554 (cellIndex o k)) hsm (μ := 1) (α := fun k ↦ Xe (n := N) (2 * a k))
      (fun k ↦ by simpa using polyNorm_Xe hN1 (2 * a k)) (fun k ↦ polyNorm_Xe hN1 _) hG hg6' hg7
      ha hpay hone (hrhs (cellIndex o k)) (heB (cellIndex o k))
    have hk : (⟨(cellIndex o k).val % 5, Nat.mod_lt _ (by inst_num)⟩ : Fin 5) = k :=
      Fin.ext (cellIndex_mod o k)
    rw [hk] at hc
    exact hc
  have hsum := Rel.sum Finset.univ (fun k _ ↦ hcell k)
  rw [blocks_sum, pair_sum, onehot_collapse h6 sp p4 (fun k ↦ Xe (n := N) (2 * a k) * 1) _ sel hsel]
    at hsum
  rw [e0, e1, e2, e3, e4, e5]
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  simp

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Member `o` of bucket `1`: an encoding of `p₄ X^(2 z_o)` at `Q0`, `z_o = a_(sel₁ o) + a_(sel₀ o)`. -/
theorem smudge0Member1 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h554 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_556 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h555 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_557 Backend.backend model
      params i0 o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {bot0' bot1' g0 g1 g2 g3 g4 g5 : Fin 800 → ExactMatrix Q0 N 1 6}
    (h558 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_560 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, m0, m2, m4, apay, m1, m3, m5, key, bot0' i,
        bot1' i, ()) (g0 i, g1 i, g2 i, g3 i, g4 i, g5 i, ()))
    {n0 n1 n2 n3 n4 n5 : Fin 160 → ExactMatrix Q0 N 1 6}
    (h559 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_561 Backend.backend model
      params i0 o (g0, g1, g2, g3, g4, g5, ()) (n0 o, n1 o, n2 o, n3 o, n4 o, n5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10} {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6') (hg7 : centeredRebaseRuns g6' g7)
    {g5' : ExactMatrix Q0 N 1 8} {h6 : ExactMatrix Q5 N 1 8}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 8 g5') (hh6 : centeredRebaseRuns g5' h6)
    {h5 : ExactMatrix Q0 N 1 6} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat} (hone : Rel sm ![![one00, one01], ![one10, one11]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![onec0, onec1] Bs)
    {s : Fin 800 → Int} {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 (smudge0Rhs0 i0) i j +
        ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int} {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6} {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model key Q0 6 (smudge0Rhs1 i0) i j +
        ![bot0' i, bot1' i] j =
      reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0) (o : Fin 160) :
    Rel sm ![![n0 o, n1 o], ![n2 o, n3 o]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * (a (sel' o) + a (sel o))) *
        unitPair sp j) • h5)
      ![n4 o, n5 o] (5 * smudge0Cell1Bound Ba (5 * smudge0Cell0Bound Ba Bs BB) BB') := by
  have hm := members559 (h559 o)
  simp only [Prod.mk.injEq] at hm
  obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := hm
  have hmem : ∀ o' : Fin 160, Rel sm ![![m0 o', m1 o'], ![m2 o', m3 o']]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * a (sel o')) * unitPair sp j) • h6)
      ![m4 o', m5 o'] (5 * smudge0Cell0Bound Ba Bs BB) :=
    fun o' ↦ smudge0Member0 h554 h555 hsm hG hg6' hg7 ha hpay hone hrhs heB o' (hsel o')
  have hcell : ∀ k : Fin 5, Rel sm
      ![![g0 (cellIndex o k), g1 (cellIndex o k)], ![g2 (cellIndex o k), g3 (cellIndex o k)]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((Xe (n := N) (2 * a k) *
        Xe (n := N) (2 * a (sel o))) * ((s' (cellIndex o k) : Int) : ErrorPoly N)) * unitPair sp j) • h5)
      ![g4 (cellIndex o k), g5 (cellIndex o k)] (smudge0Cell1Bound Ba (5 * smudge0Cell0Bound Ba Bs BB) BB') := by
    intro k
    have hc := smudge0_cell1 (h558 (cellIndex o k)) hsm (μ := fun o' ↦ Xe (n := N) (2 * a (sel o')))
      (α := fun k ↦ Xe (n := N) (2 * a k)) (fun k o' ↦ polyNorm_Xe_mul _ _)
      (fun k ↦ polyNorm_Xe hN1 _) hG hg5' hh6 ha hpay hmem (hrhs' (cellIndex o k))
      (heB' (cellIndex o k))
    have hk : (⟨(cellIndex o k).val % 5, Nat.mod_lt _ (by inst_num)⟩ : Fin 5) = k :=
      Fin.ext (cellIndex_mod o k)
    have ho : (⟨(cellIndex o k).val / 5, by have := (cellIndex o k).isLt; omega⟩ : Fin 160) = o :=
      Fin.ext (cellIndex_div o k)
    rw [hk, ho] at hc
    exact hc
  have hsum := Rel.sum Finset.univ (fun k _ ↦ hcell k)
  rw [blocks_sum, pair_sum, onehot_collapse h5 sp (qd i0)
    (fun k ↦ Xe (n := N) (2 * a k) * Xe (n := N) (2 * a (sel o))) _ (sel' o) (hsel' o)] at hsum
  rw [e0, e1, e2, e3, e4, e5]
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  simp only [Xe_mul_Xe]
  ring_nf


/-- The bound of a bucket-`0` cell. -/
abbrev smudge1Cell0Bound (Ba Bs BB : Nat) : Nat :=
  (((10 + 10) * N * ((12 + 12) * N * Ba * 16384 + N * 1 * Bs) * 16384 + (N + 2) * (p5 / 2)) / p5) +
    N * 1 * BB

/-- The bound of a bucket-`1` cell. -/
abbrev smudge1Cell1Bound (Ba Bs BB : Nat) : Nat :=
  (((8 + 8) * N * ((10 + 10) * N * ((12 * N * Ba * (32768 / 2) + (N + 2) * (p5 / 2)) / p5) *
    16384 + N * 1 * Bs) * 16384 + (N + 2) * (p4 / 2)) / p4) + N * 1 * BB

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Member `o` of bucket `0`: an encoding of `p₅ X^(2 a_(sel₀ o))` at `Q5`. -/
theorem smudge1Member0 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h609 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_611 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h610 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_612 Backend.backend model
      params i0 o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10} {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6') (hg7 : centeredRebaseRuns g6' g7)
    {h6 : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat} (hone : Rel sm ![![one00, one01], ![one10, one11]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![onec0, onec1] Bs)
    {s : Fin 800 → Int} {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 (smudge1Rhs0 i0) i j + ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (o : Fin 160) {sel : Fin 5} (hsel : ∀ k : Fin 5, s (cellIndex o k) = if k = sel then 1 else 0) :
    Rel sm ![![m0 o, m1 o], ![m2 o, m3 o]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * a sel) * unitPair sp j) • h6)
      ![m4 o, m5 o] (5 * smudge1Cell0Bound Ba Bs BB) := by
  have hm := members610 (h610 o)
  simp only [Prod.mk.injEq] at hm
  obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := hm
  have hcell : ∀ k : Fin 5, Rel sm
      ![![f0 (cellIndex o k), f1 (cellIndex o k)], ![f2 (cellIndex o k), f3 (cellIndex o k)]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * ((Xe (n := N) (2 * a k) * 1) *
        ((s (cellIndex o k) : Int) : ErrorPoly N)) * unitPair sp j) • h6)
      ![f4 (cellIndex o k), f5 (cellIndex o k)] (smudge1Cell0Bound Ba Bs BB) := by
    intro k
    have hc := smudge1_cell0 (h609 (cellIndex o k)) hsm (μ := 1) (α := fun k ↦ Xe (n := N) (2 * a k))
      (fun k ↦ by simpa using polyNorm_Xe hN1 (2 * a k)) (fun k ↦ polyNorm_Xe hN1 _) hG hg6' hg7
      ha hpay hone (hrhs (cellIndex o k)) (heB (cellIndex o k))
    have hk : (⟨(cellIndex o k).val % 5, Nat.mod_lt _ (by inst_num)⟩ : Fin 5) = k :=
      Fin.ext (cellIndex_mod o k)
    rw [hk] at hc
    exact hc
  have hsum := Rel.sum Finset.univ (fun k _ ↦ hcell k)
  rw [blocks_sum, pair_sum, onehot_collapse h6 sp p4 (fun k ↦ Xe (n := N) (2 * a k) * 1) _ sel hsel]
    at hsum
  rw [e0, e1, e2, e3, e4, e5]
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  simp

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Member `o` of bucket `1`: an encoding of `p₄ X^(2 z_o)` at `Q0`, `z_o = a_(sel₁ o) + a_(sel₀ o)`. -/
theorem smudge1Member1 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h609 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_611 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h610 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_612 Backend.backend model
      params i0 o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {bot0' bot1' g0 g1 g2 g3 g4 g5 : Fin 800 → ExactMatrix Q0 N 1 6}
    (h613 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_615 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, m0, m2, m4, apay, m1, m3, m5, key, bot0' i,
        bot1' i, ()) (g0 i, g1 i, g2 i, g3 i, g4 i, g5 i, ()))
    {n0 n1 n2 n3 n4 n5 : Fin 160 → ExactMatrix Q0 N 1 6}
    (h614 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_616 Backend.backend model
      params i0 o (g0, g1, g2, g3, g4, g5, ()) (n0 o, n1 o, n2 o, n3 o, n4 o, n5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10} {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6') (hg7 : centeredRebaseRuns g6' g7)
    {g5' : ExactMatrix Q0 N 1 8} {h6 : ExactMatrix Q5 N 1 8}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 8 g5') (hh6 : centeredRebaseRuns g5' h6)
    {h5 : ExactMatrix Q0 N 1 6} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat} (hone : Rel sm ![![one00, one01], ![one10, one11]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![onec0, onec1] Bs)
    {s : Fin 800 → Int} {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 (smudge1Rhs0 i0) i j +
        ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int} {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6} {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model key Q0 6 (smudge1Rhs1 i0) i j +
        ![bot0' i, bot1' i] j =
      reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0) (o : Fin 160) :
    Rel sm ![![n0 o, n1 o], ![n2 o, n3 o]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * (a (sel' o) + a (sel o))) *
        unitPair sp j) • h5)
      ![n4 o, n5 o] (5 * smudge1Cell1Bound Ba (5 * smudge1Cell0Bound Ba Bs BB) BB') := by
  have hm := members614 (h614 o)
  simp only [Prod.mk.injEq] at hm
  obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := hm
  have hmem : ∀ o' : Fin 160, Rel sm ![![m0 o', m1 o'], ![m2 o', m3 o']]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * a (sel o')) * unitPair sp j) • h6)
      ![m4 o', m5 o'] (5 * smudge1Cell0Bound Ba Bs BB) :=
    fun o' ↦ smudge1Member0 h609 h610 hsm hG hg6' hg7 ha hpay hone hrhs heB o' (hsel o')
  have hcell : ∀ k : Fin 5, Rel sm
      ![![g0 (cellIndex o k), g1 (cellIndex o k)], ![g2 (cellIndex o k), g3 (cellIndex o k)]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((Xe (n := N) (2 * a k) *
        Xe (n := N) (2 * a (sel o))) * ((s' (cellIndex o k) : Int) : ErrorPoly N)) * unitPair sp j) • h5)
      ![g4 (cellIndex o k), g5 (cellIndex o k)] (smudge1Cell1Bound Ba (5 * smudge1Cell0Bound Ba Bs BB) BB') := by
    intro k
    have hc := smudge1_cell1 (h613 (cellIndex o k)) hsm (μ := fun o' ↦ Xe (n := N) (2 * a (sel o')))
      (α := fun k ↦ Xe (n := N) (2 * a k)) (fun k o' ↦ polyNorm_Xe_mul _ _)
      (fun k ↦ polyNorm_Xe hN1 _) hG hg5' hh6 ha hpay hmem (hrhs' (cellIndex o k))
      (heB' (cellIndex o k))
    have hk : (⟨(cellIndex o k).val % 5, Nat.mod_lt _ (by inst_num)⟩ : Fin 5) = k :=
      Fin.ext (cellIndex_mod o k)
    have ho : (⟨(cellIndex o k).val / 5, by have := (cellIndex o k).isLt; omega⟩ : Fin 160) = o :=
      Fin.ext (cellIndex_div o k)
    rw [hk, ho] at hc
    exact hc
  have hsum := Rel.sum Finset.univ (fun k _ ↦ hcell k)
  rw [blocks_sum, pair_sum, onehot_collapse h5 sp (qd i0)
    (fun k ↦ Xe (n := N) (2 * a k) * Xe (n := N) (2 * a (sel o))) _ (sel' o) (hsel' o)] at hsum
  rw [e0, e1, e2, e3, e4, e5]
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  simp only [Xe_mul_Xe]
  ring_nf


/-- The bound of a bucket-`0` cell. -/
abbrev smudge2Cell0Bound (Ba Bs BB : Nat) : Nat :=
  (((10 + 10) * N * ((12 + 12) * N * Ba * 16384 + N * 1 * Bs) * 16384 + (N + 2) * (p5 / 2)) / p5) +
    N * 1 * BB

/-- The bound of a bucket-`1` cell. -/
abbrev smudge2Cell1Bound (Ba Bs BB : Nat) : Nat :=
  (((8 + 8) * N * ((10 + 10) * N * ((12 * N * Ba * (32768 / 2) + (N + 2) * (p5 / 2)) / p5) *
    16384 + N * 1 * Bs) * 16384 + (N + 2) * (p4 / 2)) / p4) + N * 1 * BB

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Member `o` of bucket `0`: an encoding of `p₅ X^(2 a_(sel₀ o))` at `Q5`. -/
theorem smudge2Member0 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h663 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_665 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h664 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_666 Backend.backend model
      params i0 o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10} {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6') (hg7 : centeredRebaseRuns g6' g7)
    {h6 : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat} (hone : Rel sm ![![one00, one01], ![one10, one11]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![onec0, onec1] Bs)
    {s : Fin 800 → Int} {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 (smudge2Rhs0 i0) i j + ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (o : Fin 160) {sel : Fin 5} (hsel : ∀ k : Fin 5, s (cellIndex o k) = if k = sel then 1 else 0) :
    Rel sm ![![m0 o, m1 o], ![m2 o, m3 o]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * a sel) * unitPair sp j) • h6)
      ![m4 o, m5 o] (5 * smudge2Cell0Bound Ba Bs BB) := by
  have hm := members664 (h664 o)
  simp only [Prod.mk.injEq] at hm
  obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := hm
  have hcell : ∀ k : Fin 5, Rel sm
      ![![f0 (cellIndex o k), f1 (cellIndex o k)], ![f2 (cellIndex o k), f3 (cellIndex o k)]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * ((Xe (n := N) (2 * a k) * 1) *
        ((s (cellIndex o k) : Int) : ErrorPoly N)) * unitPair sp j) • h6)
      ![f4 (cellIndex o k), f5 (cellIndex o k)] (smudge2Cell0Bound Ba Bs BB) := by
    intro k
    have hc := smudge2_cell0 (h663 (cellIndex o k)) hsm (μ := 1) (α := fun k ↦ Xe (n := N) (2 * a k))
      (fun k ↦ by simpa using polyNorm_Xe hN1 (2 * a k)) (fun k ↦ polyNorm_Xe hN1 _) hG hg6' hg7
      ha hpay hone (hrhs (cellIndex o k)) (heB (cellIndex o k))
    have hk : (⟨(cellIndex o k).val % 5, Nat.mod_lt _ (by inst_num)⟩ : Fin 5) = k :=
      Fin.ext (cellIndex_mod o k)
    rw [hk] at hc
    exact hc
  have hsum := Rel.sum Finset.univ (fun k _ ↦ hcell k)
  rw [blocks_sum, pair_sum, onehot_collapse h6 sp p4 (fun k ↦ Xe (n := N) (2 * a k) * 1) _ sel hsel]
    at hsum
  rw [e0, e1, e2, e3, e4, e5]
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  simp

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Member `o` of bucket `1`: an encoding of `p₄ X^(2 z_o)` at `Q0`, `z_o = a_(sel₁ o) + a_(sel₀ o)`. -/
theorem smudge2Member1 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h663 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_665 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h664 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_666 Backend.backend model
      params i0 o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {bot0' bot1' g0 g1 g2 g3 g4 g5 : Fin 800 → ExactMatrix Q0 N 1 6}
    (h667 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_669 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, m0, m2, m4, apay, m1, m3, m5, key, bot0' i,
        bot1' i, ()) (g0 i, g1 i, g2 i, g3 i, g4 i, g5 i, ()))
    {n0 n1 n2 n3 n4 n5 : Fin 160 → ExactMatrix Q0 N 1 6}
    (h668 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_670 Backend.backend model
      params i0 o (g0, g1, g2, g3, g4, g5, ()) (n0 o, n1 o, n2 o, n3 o, n4 o, n5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10} {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6') (hg7 : centeredRebaseRuns g6' g7)
    {g5' : ExactMatrix Q0 N 1 8} {h6 : ExactMatrix Q5 N 1 8}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 8 g5') (hh6 : centeredRebaseRuns g5' h6)
    {h5 : ExactMatrix Q0 N 1 6} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat} (hone : Rel sm ![![one00, one01], ![one10, one11]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![onec0, onec1] Bs)
    {s : Fin 800 → Int} {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 (smudge2Rhs0 i0) i j +
        ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int} {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6} {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model key Q0 6 (smudge2Rhs1 i0) i j +
        ![bot0' i, bot1' i] j =
      reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0) (o : Fin 160) :
    Rel sm ![![n0 o, n1 o], ![n2 o, n3 o]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * (a (sel' o) + a (sel o))) *
        unitPair sp j) • h5)
      ![n4 o, n5 o] (5 * smudge2Cell1Bound Ba (5 * smudge2Cell0Bound Ba Bs BB) BB') := by
  have hm := members668 (h668 o)
  simp only [Prod.mk.injEq] at hm
  obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := hm
  have hmem : ∀ o' : Fin 160, Rel sm ![![m0 o', m1 o'], ![m2 o', m3 o']]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * a (sel o')) * unitPair sp j) • h6)
      ![m4 o', m5 o'] (5 * smudge2Cell0Bound Ba Bs BB) :=
    fun o' ↦ smudge2Member0 h663 h664 hsm hG hg6' hg7 ha hpay hone hrhs heB o' (hsel o')
  have hcell : ∀ k : Fin 5, Rel sm
      ![![g0 (cellIndex o k), g1 (cellIndex o k)], ![g2 (cellIndex o k), g3 (cellIndex o k)]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((Xe (n := N) (2 * a k) *
        Xe (n := N) (2 * a (sel o))) * ((s' (cellIndex o k) : Int) : ErrorPoly N)) * unitPair sp j) • h5)
      ![g4 (cellIndex o k), g5 (cellIndex o k)] (smudge2Cell1Bound Ba (5 * smudge2Cell0Bound Ba Bs BB) BB') := by
    intro k
    have hc := smudge2_cell1 (h667 (cellIndex o k)) hsm (μ := fun o' ↦ Xe (n := N) (2 * a (sel o')))
      (α := fun k ↦ Xe (n := N) (2 * a k)) (fun k o' ↦ polyNorm_Xe_mul _ _)
      (fun k ↦ polyNorm_Xe hN1 _) hG hg5' hh6 ha hpay hmem (hrhs' (cellIndex o k))
      (heB' (cellIndex o k))
    have hk : (⟨(cellIndex o k).val % 5, Nat.mod_lt _ (by inst_num)⟩ : Fin 5) = k :=
      Fin.ext (cellIndex_mod o k)
    have ho : (⟨(cellIndex o k).val / 5, by have := (cellIndex o k).isLt; omega⟩ : Fin 160) = o :=
      Fin.ext (cellIndex_div o k)
    rw [hk, ho] at hc
    exact hc
  have hsum := Rel.sum Finset.univ (fun k _ ↦ hcell k)
  rw [blocks_sum, pair_sum, onehot_collapse h5 sp (qd i0)
    (fun k ↦ Xe (n := N) (2 * a k) * Xe (n := N) (2 * a (sel o))) _ (sel' o) (hsel' o)] at hsum
  rw [e0, e1, e2, e3, e4, e5]
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  simp only [Xe_mul_Xe]
  ring_nf


/-- The bound of a bucket-`0` cell. -/
abbrev smudge3Cell0Bound (Ba Bs BB : Nat) : Nat :=
  (((10 + 10) * N * ((12 + 12) * N * Ba * 16384 + N * 1 * Bs) * 16384 + (N + 2) * (p5 / 2)) / p5) +
    N * 1 * BB

/-- The bound of a bucket-`1` cell. -/
abbrev smudge3Cell1Bound (Ba Bs BB : Nat) : Nat :=
  (((8 + 8) * N * ((10 + 10) * N * ((12 * N * Ba * (32768 / 2) + (N + 2) * (p5 / 2)) / p5) *
    16384 + N * 1 * Bs) * 16384 + (N + 2) * (p4 / 2)) / p4) + N * 1 * BB

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Member `o` of bucket `0`: an encoding of `p₅ X^(2 a_(sel₀ o))` at `Q5`. -/
theorem smudge3Member0 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h717 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_719 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h718 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_720 Backend.backend model
      params i0 o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10} {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6') (hg7 : centeredRebaseRuns g6' g7)
    {h6 : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat} (hone : Rel sm ![![one00, one01], ![one10, one11]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![onec0, onec1] Bs)
    {s : Fin 800 → Int} {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 (smudge3Rhs0 i0) i j + ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (o : Fin 160) {sel : Fin 5} (hsel : ∀ k : Fin 5, s (cellIndex o k) = if k = sel then 1 else 0) :
    Rel sm ![![m0 o, m1 o], ![m2 o, m3 o]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * a sel) * unitPair sp j) • h6)
      ![m4 o, m5 o] (5 * smudge3Cell0Bound Ba Bs BB) := by
  have hm := members718 (h718 o)
  simp only [Prod.mk.injEq] at hm
  obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := hm
  have hcell : ∀ k : Fin 5, Rel sm
      ![![f0 (cellIndex o k), f1 (cellIndex o k)], ![f2 (cellIndex o k), f3 (cellIndex o k)]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * ((Xe (n := N) (2 * a k) * 1) *
        ((s (cellIndex o k) : Int) : ErrorPoly N)) * unitPair sp j) • h6)
      ![f4 (cellIndex o k), f5 (cellIndex o k)] (smudge3Cell0Bound Ba Bs BB) := by
    intro k
    have hc := smudge3_cell0 (h717 (cellIndex o k)) hsm (μ := 1) (α := fun k ↦ Xe (n := N) (2 * a k))
      (fun k ↦ by simpa using polyNorm_Xe hN1 (2 * a k)) (fun k ↦ polyNorm_Xe hN1 _) hG hg6' hg7
      ha hpay hone (hrhs (cellIndex o k)) (heB (cellIndex o k))
    have hk : (⟨(cellIndex o k).val % 5, Nat.mod_lt _ (by inst_num)⟩ : Fin 5) = k :=
      Fin.ext (cellIndex_mod o k)
    rw [hk] at hc
    exact hc
  have hsum := Rel.sum Finset.univ (fun k _ ↦ hcell k)
  rw [blocks_sum, pair_sum, onehot_collapse h6 sp p4 (fun k ↦ Xe (n := N) (2 * a k) * 1) _ sel hsel]
    at hsum
  rw [e0, e1, e2, e3, e4, e5]
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  simp

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Member `o` of bucket `1`: an encoding of `p₄ X^(2 z_o)` at `Q0`, `z_o = a_(sel₁ o) + a_(sel₀ o)`. -/
theorem smudge3Member1 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h717 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_719 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h718 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_720 Backend.backend model
      params i0 o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {bot0' bot1' g0 g1 g2 g3 g4 g5 : Fin 800 → ExactMatrix Q0 N 1 6}
    (h721 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_723 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, m0, m2, m4, apay, m1, m3, m5, key, bot0' i,
        bot1' i, ()) (g0 i, g1 i, g2 i, g3 i, g4 i, g5 i, ()))
    {n0 n1 n2 n3 n4 n5 : Fin 160 → ExactMatrix Q0 N 1 6}
    (h722 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_724 Backend.backend model
      params i0 o (g0, g1, g2, g3, g4, g5, ()) (n0 o, n1 o, n2 o, n3 o, n4 o, n5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10} {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6') (hg7 : centeredRebaseRuns g6' g7)
    {g5' : ExactMatrix Q0 N 1 8} {h6 : ExactMatrix Q5 N 1 8}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 8 g5') (hh6 : centeredRebaseRuns g5' h6)
    {h5 : ExactMatrix Q0 N 1 6} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat} (hone : Rel sm ![![one00, one01], ![one10, one11]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![onec0, onec1] Bs)
    {s : Fin 800 → Int} {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 (smudge3Rhs0 i0) i j +
        ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int} {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6} {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model key Q0 6 (smudge3Rhs1 i0) i j +
        ![bot0' i, bot1' i] j =
      reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0) (o : Fin 160) :
    Rel sm ![![n0 o, n1 o], ![n2 o, n3 o]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * (a (sel' o) + a (sel o))) *
        unitPair sp j) • h5)
      ![n4 o, n5 o] (5 * smudge3Cell1Bound Ba (5 * smudge3Cell0Bound Ba Bs BB) BB') := by
  have hm := members722 (h722 o)
  simp only [Prod.mk.injEq] at hm
  obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := hm
  have hmem : ∀ o' : Fin 160, Rel sm ![![m0 o', m1 o'], ![m2 o', m3 o']]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * a (sel o')) * unitPair sp j) • h6)
      ![m4 o', m5 o'] (5 * smudge3Cell0Bound Ba Bs BB) :=
    fun o' ↦ smudge3Member0 h717 h718 hsm hG hg6' hg7 ha hpay hone hrhs heB o' (hsel o')
  have hcell : ∀ k : Fin 5, Rel sm
      ![![g0 (cellIndex o k), g1 (cellIndex o k)], ![g2 (cellIndex o k), g3 (cellIndex o k)]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((Xe (n := N) (2 * a k) *
        Xe (n := N) (2 * a (sel o))) * ((s' (cellIndex o k) : Int) : ErrorPoly N)) * unitPair sp j) • h5)
      ![g4 (cellIndex o k), g5 (cellIndex o k)] (smudge3Cell1Bound Ba (5 * smudge3Cell0Bound Ba Bs BB) BB') := by
    intro k
    have hc := smudge3_cell1 (h721 (cellIndex o k)) hsm (μ := fun o' ↦ Xe (n := N) (2 * a (sel o')))
      (α := fun k ↦ Xe (n := N) (2 * a k)) (fun k o' ↦ polyNorm_Xe_mul _ _)
      (fun k ↦ polyNorm_Xe hN1 _) hG hg5' hh6 ha hpay hmem (hrhs' (cellIndex o k))
      (heB' (cellIndex o k))
    have hk : (⟨(cellIndex o k).val % 5, Nat.mod_lt _ (by inst_num)⟩ : Fin 5) = k :=
      Fin.ext (cellIndex_mod o k)
    have ho : (⟨(cellIndex o k).val / 5, by have := (cellIndex o k).isLt; omega⟩ : Fin 160) = o :=
      Fin.ext (cellIndex_div o k)
    rw [hk, ho] at hc
    exact hc
  have hsum := Rel.sum Finset.univ (fun k _ ↦ hcell k)
  rw [blocks_sum, pair_sum, onehot_collapse h5 sp (qd i0)
    (fun k ↦ Xe (n := N) (2 * a k) * Xe (n := N) (2 * a (sel o))) _ (sel' o) (hsel' o)] at hsum
  rw [e0, e1, e2, e3, e4, e5]
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  simp only [Xe_mul_Xe]
  ring_nf


/-- The bound of a bucket-`0` cell. -/
abbrev smudge4Cell0Bound (Ba Bs BB : Nat) : Nat :=
  (((10 + 10) * N * ((12 + 12) * N * Ba * 16384 + N * 1 * Bs) * 16384 + (N + 2) * (p5 / 2)) / p5) +
    N * 1 * BB

/-- The bound of a bucket-`1` cell. -/
abbrev smudge4Cell1Bound (Ba Bs BB : Nat) : Nat :=
  (((8 + 8) * N * ((10 + 10) * N * ((12 * N * Ba * (32768 / 2) + (N + 2) * (p5 / 2)) / p5) *
    16384 + N * 1 * Bs) * 16384 + (N + 2) * (p4 / 2)) / p4) + N * 1 * BB

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Member `o` of bucket `0`: an encoding of `p₅ X^(2 a_(sel₀ o))` at `Q5`. -/
theorem smudge4Member0 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h771 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_773 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h772 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_774 Backend.backend model
      params i0 o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10} {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6') (hg7 : centeredRebaseRuns g6' g7)
    {h6 : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat} (hone : Rel sm ![![one00, one01], ![one10, one11]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![onec0, onec1] Bs)
    {s : Fin 800 → Int} {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 (smudge4Rhs0 i0) i j + ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (o : Fin 160) {sel : Fin 5} (hsel : ∀ k : Fin 5, s (cellIndex o k) = if k = sel then 1 else 0) :
    Rel sm ![![m0 o, m1 o], ![m2 o, m3 o]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * a sel) * unitPair sp j) • h6)
      ![m4 o, m5 o] (5 * smudge4Cell0Bound Ba Bs BB) := by
  have hm := members772 (h772 o)
  simp only [Prod.mk.injEq] at hm
  obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := hm
  have hcell : ∀ k : Fin 5, Rel sm
      ![![f0 (cellIndex o k), f1 (cellIndex o k)], ![f2 (cellIndex o k), f3 (cellIndex o k)]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * ((Xe (n := N) (2 * a k) * 1) *
        ((s (cellIndex o k) : Int) : ErrorPoly N)) * unitPair sp j) • h6)
      ![f4 (cellIndex o k), f5 (cellIndex o k)] (smudge4Cell0Bound Ba Bs BB) := by
    intro k
    have hc := smudge4_cell0 (h771 (cellIndex o k)) hsm (μ := 1) (α := fun k ↦ Xe (n := N) (2 * a k))
      (fun k ↦ by simpa using polyNorm_Xe hN1 (2 * a k)) (fun k ↦ polyNorm_Xe hN1 _) hG hg6' hg7
      ha hpay hone (hrhs (cellIndex o k)) (heB (cellIndex o k))
    have hk : (⟨(cellIndex o k).val % 5, Nat.mod_lt _ (by inst_num)⟩ : Fin 5) = k :=
      Fin.ext (cellIndex_mod o k)
    rw [hk] at hc
    exact hc
  have hsum := Rel.sum Finset.univ (fun k _ ↦ hcell k)
  rw [blocks_sum, pair_sum, onehot_collapse h6 sp p4 (fun k ↦ Xe (n := N) (2 * a k) * 1) _ sel hsel]
    at hsum
  rw [e0, e1, e2, e3, e4, e5]
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  simp

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Member `o` of bucket `1`: an encoding of `p₄ X^(2 z_o)` at `Q0`, `z_o = a_(sel₁ o) + a_(sel₀ o)`. -/
theorem smudge4Member1 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h771 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_773 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h772 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_774 Backend.backend model
      params i0 o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {bot0' bot1' g0 g1 g2 g3 g4 g5 : Fin 800 → ExactMatrix Q0 N 1 6}
    (h775 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_777 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, m0, m2, m4, apay, m1, m3, m5, key, bot0' i,
        bot1' i, ()) (g0 i, g1 i, g2 i, g3 i, g4 i, g5 i, ()))
    {n0 n1 n2 n3 n4 n5 : Fin 160 → ExactMatrix Q0 N 1 6}
    (h776 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_778 Backend.backend model
      params i0 o (g0, g1, g2, g3, g4, g5, ()) (n0 o, n1 o, n2 o, n3 o, n4 o, n5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10} {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6') (hg7 : centeredRebaseRuns g6' g7)
    {g5' : ExactMatrix Q0 N 1 8} {h6 : ExactMatrix Q5 N 1 8}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 8 g5') (hh6 : centeredRebaseRuns g5' h6)
    {h5 : ExactMatrix Q0 N 1 6} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat} (hone : Rel sm ![![one00, one01], ![one10, one11]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![onec0, onec1] Bs)
    {s : Fin 800 → Int} {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 (smudge4Rhs0 i0) i j +
        ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int} {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6} {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model key Q0 6 (smudge4Rhs1 i0) i j +
        ![bot0' i, bot1' i] j =
      reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0) (o : Fin 160) :
    Rel sm ![![n0 o, n1 o], ![n2 o, n3 o]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * (a (sel' o) + a (sel o))) *
        unitPair sp j) • h5)
      ![n4 o, n5 o] (5 * smudge4Cell1Bound Ba (5 * smudge4Cell0Bound Ba Bs BB) BB') := by
  have hm := members776 (h776 o)
  simp only [Prod.mk.injEq] at hm
  obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := hm
  have hmem : ∀ o' : Fin 160, Rel sm ![![m0 o', m1 o'], ![m2 o', m3 o']]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * a (sel o')) * unitPair sp j) • h6)
      ![m4 o', m5 o'] (5 * smudge4Cell0Bound Ba Bs BB) :=
    fun o' ↦ smudge4Member0 h771 h772 hsm hG hg6' hg7 ha hpay hone hrhs heB o' (hsel o')
  have hcell : ∀ k : Fin 5, Rel sm
      ![![g0 (cellIndex o k), g1 (cellIndex o k)], ![g2 (cellIndex o k), g3 (cellIndex o k)]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((Xe (n := N) (2 * a k) *
        Xe (n := N) (2 * a (sel o))) * ((s' (cellIndex o k) : Int) : ErrorPoly N)) * unitPair sp j) • h5)
      ![g4 (cellIndex o k), g5 (cellIndex o k)] (smudge4Cell1Bound Ba (5 * smudge4Cell0Bound Ba Bs BB) BB') := by
    intro k
    have hc := smudge4_cell1 (h775 (cellIndex o k)) hsm (μ := fun o' ↦ Xe (n := N) (2 * a (sel o')))
      (α := fun k ↦ Xe (n := N) (2 * a k)) (fun k o' ↦ polyNorm_Xe_mul _ _)
      (fun k ↦ polyNorm_Xe hN1 _) hG hg5' hh6 ha hpay hmem (hrhs' (cellIndex o k))
      (heB' (cellIndex o k))
    have hk : (⟨(cellIndex o k).val % 5, Nat.mod_lt _ (by inst_num)⟩ : Fin 5) = k :=
      Fin.ext (cellIndex_mod o k)
    have ho : (⟨(cellIndex o k).val / 5, by have := (cellIndex o k).isLt; omega⟩ : Fin 160) = o :=
      Fin.ext (cellIndex_div o k)
    rw [hk, ho] at hc
    exact hc
  have hsum := Rel.sum Finset.univ (fun k _ ↦ hcell k)
  rw [blocks_sum, pair_sum, onehot_collapse h5 sp (qd i0)
    (fun k ↦ Xe (n := N) (2 * a k) * Xe (n := N) (2 * a (sel o))) _ (sel' o) (hsel' o)] at hsum
  rw [e0, e1, e2, e3, e4, e5]
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  simp only [Xe_mul_Xe]
  ring_nf


/-- The bound of a bucket-`0` cell. -/
abbrev smudge5Cell0Bound (Ba Bs BB : Nat) : Nat :=
  (((10 + 10) * N * ((12 + 12) * N * Ba * 16384 + N * 1 * Bs) * 16384 + (N + 2) * (p5 / 2)) / p5) +
    N * 1 * BB

/-- The bound of a bucket-`1` cell. -/
abbrev smudge5Cell1Bound (Ba Bs BB : Nat) : Nat :=
  (((8 + 8) * N * ((10 + 10) * N * ((12 * N * Ba * (32768 / 2) + (N + 2) * (p5 / 2)) / p5) *
    16384 + N * 1 * Bs) * 16384 + (N + 2) * (p4 / 2)) / p4) + N * 1 * BB

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Member `o` of bucket `0`: an encoding of `p₅ X^(2 a_(sel₀ o))` at `Q5`. -/
theorem smudge5Member0 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h825 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_827 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h826 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_828 Backend.backend model
      params i0 o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10} {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6') (hg7 : centeredRebaseRuns g6' g7)
    {h6 : ExactMatrix Q5 N 1 8} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat} (hone : Rel sm ![![one00, one01], ![one10, one11]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![onec0, onec1] Bs)
    {s : Fin 800 → Int} {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 (smudge5Rhs0 i0) i j + ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (o : Fin 160) {sel : Fin 5} (hsel : ∀ k : Fin 5, s (cellIndex o k) = if k = sel then 1 else 0) :
    Rel sm ![![m0 o, m1 o], ![m2 o, m3 o]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * a sel) * unitPair sp j) • h6)
      ![m4 o, m5 o] (5 * smudge5Cell0Bound Ba Bs BB) := by
  have hm := members826 (h826 o)
  simp only [Prod.mk.injEq] at hm
  obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := hm
  have hcell : ∀ k : Fin 5, Rel sm
      ![![f0 (cellIndex o k), f1 (cellIndex o k)], ![f2 (cellIndex o k), f3 (cellIndex o k)]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * ((Xe (n := N) (2 * a k) * 1) *
        ((s (cellIndex o k) : Int) : ErrorPoly N)) * unitPair sp j) • h6)
      ![f4 (cellIndex o k), f5 (cellIndex o k)] (smudge5Cell0Bound Ba Bs BB) := by
    intro k
    have hc := smudge5_cell0 (h825 (cellIndex o k)) hsm (μ := 1) (α := fun k ↦ Xe (n := N) (2 * a k))
      (fun k ↦ by simpa using polyNorm_Xe hN1 (2 * a k)) (fun k ↦ polyNorm_Xe hN1 _) hG hg6' hg7
      ha hpay hone (hrhs (cellIndex o k)) (heB (cellIndex o k))
    have hk : (⟨(cellIndex o k).val % 5, Nat.mod_lt _ (by inst_num)⟩ : Fin 5) = k :=
      Fin.ext (cellIndex_mod o k)
    rw [hk] at hc
    exact hc
  have hsum := Rel.sum Finset.univ (fun k _ ↦ hcell k)
  rw [blocks_sum, pair_sum, onehot_collapse h6 sp p4 (fun k ↦ Xe (n := N) (2 * a k) * 1) _ sel hsel]
    at hsum
  rw [e0, e1, e2, e3, e4, e5]
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  simp

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Member `o` of bucket `1`: an encoding of `p₄ X^(2 z_o)` at `Q0`, `z_o = a_(sel₁ o) + a_(sel₀ o)`. -/
theorem smudge5Member1 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h825 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_827 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h826 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_828 Backend.backend model
      params i0 o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {bot0' bot1' g0 g1 g2 g3 g4 g5 : Fin 800 → ExactMatrix Q0 N 1 6}
    (h829 : ∀ i : Fin 800, Stage_online_eval.parallel_parallel_generatedRoot_290_831 Backend.backend model
      params i0 i (ca0, aA00, aA10, ca1, aA01, aA11, m0, m2, m4, apay, m1, m3, m5, key, bot0' i,
        bot1' i, ()) (g0 i, g1 i, g2 i, g3 i, g4 i, g5 i, ()))
    {n0 n1 n2 n3 n4 n5 : Fin 160 → ExactMatrix Q0 N 1 6}
    (h830 : ∀ o : Fin 160, Stage_online_eval.parallel_parallel_generatedRoot_290_832 Backend.backend model
      params i0 o (g0, g1, g2, g3, g4, g5, ()) (n0 o, n1 o, n2 o, n3 o, n4 o, n5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10} {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6') (hg7 : centeredRebaseRuns g6' g7)
    {g5' : ExactMatrix Q0 N 1 8} {h6 : ExactMatrix Q5 N 1 8}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 8 g5') (hh6 : centeredRebaseRuns g5' h6)
    {h5 : ExactMatrix Q0 N 1 6} {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k : Fin 5, apay k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat} (hone : Rel sm ![![one00, one01], ![one10, one11]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![onec0, onec1] Bs)
    {s : Fin 800 → Int} {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8} {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 (smudge5Rhs0 i0) i j +
        ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int} {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6} {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model key Q0 6 (smudge5Rhs1 i0) i j +
        ![bot0' i, bot1' i] j =
      reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0) (o : Fin 160) :
    Rel sm ![![n0 o, n1 o], ![n2 o, n3 o]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * (a (sel' o) + a (sel o))) *
        unitPair sp j) • h5)
      ![n4 o, n5 o] (5 * smudge5Cell1Bound Ba (5 * smudge5Cell0Bound Ba Bs BB) BB') := by
  have hm := members830 (h830 o)
  simp only [Prod.mk.injEq] at hm
  obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := hm
  have hmem : ∀ o' : Fin 160, Rel sm ![![m0 o', m1 o'], ![m2 o', m3 o']]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * a (sel o')) * unitPair sp j) • h6)
      ![m4 o', m5 o'] (5 * smudge5Cell0Bound Ba Bs BB) :=
    fun o' ↦ smudge5Member0 h825 h826 hsm hG hg6' hg7 ha hpay hone hrhs heB o' (hsel o')
  have hcell : ∀ k : Fin 5, Rel sm
      ![![g0 (cellIndex o k), g1 (cellIndex o k)], ![g2 (cellIndex o k), g3 (cellIndex o k)]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((Xe (n := N) (2 * a k) *
        Xe (n := N) (2 * a (sel o))) * ((s' (cellIndex o k) : Int) : ErrorPoly N)) * unitPair sp j) • h5)
      ![g4 (cellIndex o k), g5 (cellIndex o k)] (smudge5Cell1Bound Ba (5 * smudge5Cell0Bound Ba Bs BB) BB') := by
    intro k
    have hc := smudge5_cell1 (h829 (cellIndex o k)) hsm (μ := fun o' ↦ Xe (n := N) (2 * a (sel o')))
      (α := fun k ↦ Xe (n := N) (2 * a k)) (fun k o' ↦ polyNorm_Xe_mul _ _)
      (fun k ↦ polyNorm_Xe hN1 _) hG hg5' hh6 ha hpay hmem (hrhs' (cellIndex o k))
      (heB' (cellIndex o k))
    have hk : (⟨(cellIndex o k).val % 5, Nat.mod_lt _ (by inst_num)⟩ : Fin 5) = k :=
      Fin.ext (cellIndex_mod o k)
    have ho : (⟨(cellIndex o k).val / 5, by have := (cellIndex o k).isLt; omega⟩ : Fin 160) = o :=
      Fin.ext (cellIndex_div o k)
    rw [hk, ho] at hc
    exact hc
  have hsum := Rel.sum Finset.univ (fun k _ ↦ hcell k)
  rw [blocks_sum, pair_sum, onehot_collapse h5 sp (qd i0)
    (fun k ↦ Xe (n := N) (2 * a k) * Xe (n := N) (2 * a (sel o))) _ (sel' o) (hsel' o)] at hsum
  rw [e0, e1, e2, e3, e4, e5]
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  simp only [Xe_mul_Xe]
  ring_nf


end RefreshTopProof.Part_I
