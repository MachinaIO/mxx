import Part_0_1_SmudgeMembers
import Part_0_1_Smudge

/-! Every decoder's smudge (written by `generator/gen_smudge_totals.py`). -/

namespace RefreshTopProof.Part_0_1

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

/-- The bound of one member's LUT output of decoder `0`. -/
abbrev smudge0LutBound (Ba Bs BB BB' Bh BS : Nat) : Nat :=
  (6 * ((3 * lutBranchBound560 (5 * smudge0Cell1Bound Ba (5 * smudge0Cell0Bound Ba Bs BB) BB') Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * (5 * smudge0Cell1Bound Ba (5 * smudge0Cell0Bound Ba Bs BB) BB') * 16384 +
    (N * 1 * (q3 / 2) + q3 / 2)) + (N + 2) * (q3 / 2)) / q3 + 4 * (N * 1 * BS)) * (32768 / 2))

/-- The smudge of decoder `0`: `Σ_o X^⌊o/20⌋ w_(o mod 20) T'(z_o)`. -/
noncomputable def smudgeValue0 (z : Fin 160 → Nat) : ErrorPoly N :=
  ∑ o : Fin 160, Xe (n := N) (o / 20) *
    (((smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
      ((1 : ErrorPoly N) * ((((-1 : Int) ^ (z o / 4) : Int) : ErrorPoly N) *
        smudgeTable ⟨z o % 4, Nat.mod_lt _ (by norm_num)⟩)))

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem smudgeTotal0 {model : HashModel} {params : Stage_online_eval_0_1.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h554 : ∀ i : Fin 800, Stage_online_eval_0_1.parallel_generatedRoot_554 Backend.backend model
      params i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h555 : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_555 Backend.backend model
      params o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {bot0' bot1' g0 g1 g2 g3 g4 g5 : Fin 800 → ExactMatrix Q0 N 1 6}
    (h558 : ∀ i : Fin 800, Stage_online_eval_0_1.parallel_generatedRoot_558 Backend.backend model
      params i (ca0, aA00, aA10, ca1, aA01, aA11, m0, m2, m4, apay, m1, m3, m5, key, bot0' i,
        bot1' i, ()) (g0 i, g1 i, g2 i, g3 i, g4 i, g5 i, ()))
    {n0 n1 n2 n3 n4 n5 : Fin 160 → ExactMatrix Q0 N 1 6}
    (h559 : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_559 Backend.backend model
      params o (g0, g1, g2, g3, g4, g5, ()) (n0 o, n1 o, n2 o, n3 o, n4 o, n5 o, ()))
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
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 smudge0Rhs0 i j +
        ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int} {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6} {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model key Q0 6 smudge0Rhs1 i j +
        ![bot0' i, bot1' i] j =
      reducePoly Q0 N (((q3 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {hl0 hl1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {u0 u1 u2 u3 u4 u5 : Fin 160 → ExactMatrix Q1 N 1 1}
    (hlut : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_560 Backend.backend model
      params o (fam00, fam01, fam10, fam11, hl0, key, hl1, n4 o, n0 o, n2 o, n5 o, n1 o, n3 o, ())
        (u0 o, u1 o, u2 o, u3 o, u4 o, u5 o, ()))
    {g1' : ExactMatrix Q1 N 1 6} (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1')
    (hh5 : centeredRebaseRuns g1' h5)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![hl0 i, hl1 i] Bh)
    {gq : ExactMatrix Q1 N 1 6} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (smudgeTable k)) (sigma i).toNat *
          reducePoly Q1 N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    Rel sm ![![∑ o, u0 o, ∑ o, u1 o], ![∑ o, u2 o, ∑ o, u3 o]]
      (fun j ↦ reducePoly Q1 N (smudgeValue0 (fun o ↦ a (sel' o) + a (sel o)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![∑ o, u4 o, ∑ o, u5 o] (160 * smudge0LutBound Ba Bs BB BB' Bh BS) := by
  have hmem : ∀ o ∈ (Finset.univ : Finset (Fin 160)), Rel sm ![![u0 o, u1 o], ![u2 o, u3 o]]
      (fun j ↦ reducePoly Q1 N (Xe (n := N) (o / 20)) • reducePoly Q1 N
        (((smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
          ((1 : ErrorPoly N) * ((((-1 : Int) ^ ((a (sel' o) + a (sel o)) / 4) : Int) : ErrorPoly N) *
            smudgeTable ⟨(a (sel' o) + a (sel o)) % 4, Nat.mod_lt _ (by norm_num)⟩)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![u4 o, u5 o] (smudge0LutBound Ba Bs BB BB' Bh BS) := by
    intro o _
    have hm := smudge0Member1 h554 h555 h558 h559 hsm hG hg6' hg7 hg5' hh6 ha hpay hone hrhs heB hrhs'
      heB' hsel hsel' o
    exact lut560 (hlut o) hsm hg1' hh5 hm hhelp o.isLt hgq hinverse hswitch
  have hsum := Rel.sum Finset.univ hmem
  rw [blocks_sum, pair_sum] at hsum
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  rw [Finset.sum_apply, smudgeValue0, Finset.sum_mul, map_sum, Finset.sum_smul]
  refine Finset.sum_congr rfl fun o _ ↦ ?_
  rw [smul_smul, ← map_mul]
  congr 2
  ring

/-- The bound of one member's LUT output of decoder `1`. -/
abbrev smudge1LutBound (Ba Bs BB BB' Bh BS : Nat) : Nat :=
  (6 * ((3 * lutBranchBound615 (5 * smudge1Cell1Bound Ba (5 * smudge1Cell0Bound Ba Bs BB) BB') Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * (5 * smudge1Cell1Bound Ba (5 * smudge1Cell0Bound Ba Bs BB) BB') * 16384 +
    (N * 1 * (q3 / 2) + q3 / 2)) + (N + 2) * (q3 / 2)) / q3 + 4 * (N * 1 * BS)) * (32768 / 2))

/-- The smudge of decoder `1`: `Σ_o X^⌊o/20⌋ w_(o mod 20) T'(z_o)`. -/
noncomputable def smudgeValue1 (z : Fin 160 → Nat) : ErrorPoly N :=
  ∑ o : Fin 160, Xe (n := N) (o / 20) *
    (((smudgeWeight615 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
      ((1 : ErrorPoly N) * ((((-1 : Int) ^ (z o / 4) : Int) : ErrorPoly N) *
        smudgeTable ⟨z o % 4, Nat.mod_lt _ (by norm_num)⟩)))

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem smudgeTotal1 {model : HashModel} {params : Stage_online_eval_0_1.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h609 : ∀ i : Fin 800, Stage_online_eval_0_1.parallel_generatedRoot_609 Backend.backend model
      params i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h610 : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_610 Backend.backend model
      params o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {bot0' bot1' g0 g1 g2 g3 g4 g5 : Fin 800 → ExactMatrix Q0 N 1 6}
    (h613 : ∀ i : Fin 800, Stage_online_eval_0_1.parallel_generatedRoot_613 Backend.backend model
      params i (ca0, aA00, aA10, ca1, aA01, aA11, m0, m2, m4, apay, m1, m3, m5, key, bot0' i,
        bot1' i, ()) (g0 i, g1 i, g2 i, g3 i, g4 i, g5 i, ()))
    {n0 n1 n2 n3 n4 n5 : Fin 160 → ExactMatrix Q0 N 1 6}
    (h614 : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_614 Backend.backend model
      params o (g0, g1, g2, g3, g4, g5, ()) (n0 o, n1 o, n2 o, n3 o, n4 o, n5 o, ()))
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
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 smudge1Rhs0 i j +
        ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int} {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6} {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model key Q0 6 smudge1Rhs1 i j +
        ![bot0' i, bot1' i] j =
      reducePoly Q0 N (((q3 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {hl0 hl1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {u0 u1 u2 u3 u4 u5 : Fin 160 → ExactMatrix Q1 N 1 1}
    (hlut : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_615 Backend.backend model
      params o (fam00, fam01, fam10, fam11, hl0, key, hl1, n4 o, n0 o, n2 o, n5 o, n1 o, n3 o, ())
        (u0 o, u1 o, u2 o, u3 o, u4 o, u5 o, ()))
    {g1' : ExactMatrix Q1 N 1 6} (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1')
    (hh5 : centeredRebaseRuns g1' h5)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![hl0 i, hl1 i] Bh)
    {gq : ExactMatrix Q1 N 1 6} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (smudgeTable k)) (sigma i).toNat *
          reducePoly Q1 N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    Rel sm ![![∑ o, u0 o, ∑ o, u1 o], ![∑ o, u2 o, ∑ o, u3 o]]
      (fun j ↦ reducePoly Q1 N (smudgeValue1 (fun o ↦ a (sel' o) + a (sel o)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![∑ o, u4 o, ∑ o, u5 o] (160 * smudge1LutBound Ba Bs BB BB' Bh BS) := by
  have hmem : ∀ o ∈ (Finset.univ : Finset (Fin 160)), Rel sm ![![u0 o, u1 o], ![u2 o, u3 o]]
      (fun j ↦ reducePoly Q1 N (Xe (n := N) (o / 20)) • reducePoly Q1 N
        (((smudgeWeight615 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
          ((1 : ErrorPoly N) * ((((-1 : Int) ^ ((a (sel' o) + a (sel o)) / 4) : Int) : ErrorPoly N) *
            smudgeTable ⟨(a (sel' o) + a (sel o)) % 4, Nat.mod_lt _ (by norm_num)⟩)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![u4 o, u5 o] (smudge1LutBound Ba Bs BB BB' Bh BS) := by
    intro o _
    have hm := smudge1Member1 h609 h610 h613 h614 hsm hG hg6' hg7 hg5' hh6 ha hpay hone hrhs heB hrhs'
      heB' hsel hsel' o
    exact lut615 (hlut o) hsm hg1' hh5 hm hhelp o.isLt hgq hinverse hswitch
  have hsum := Rel.sum Finset.univ hmem
  rw [blocks_sum, pair_sum] at hsum
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  rw [Finset.sum_apply, smudgeValue1, Finset.sum_mul, map_sum, Finset.sum_smul]
  refine Finset.sum_congr rfl fun o _ ↦ ?_
  rw [smul_smul, ← map_mul]
  congr 2
  ring

/-- The bound of one member's LUT output of decoder `2`. -/
abbrev smudge2LutBound (Ba Bs BB BB' Bh BS : Nat) : Nat :=
  (6 * ((3 * lutBranchBound669 (5 * smudge2Cell1Bound Ba (5 * smudge2Cell0Bound Ba Bs BB) BB') Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * (5 * smudge2Cell1Bound Ba (5 * smudge2Cell0Bound Ba Bs BB) BB') * 16384 +
    (N * 1 * (q3 / 2) + q3 / 2)) + (N + 2) * (q3 / 2)) / q3 + 4 * (N * 1 * BS)) * (32768 / 2))

/-- The smudge of decoder `2`: `Σ_o X^⌊o/20⌋ w_(o mod 20) T'(z_o)`. -/
noncomputable def smudgeValue2 (z : Fin 160 → Nat) : ErrorPoly N :=
  ∑ o : Fin 160, Xe (n := N) (o / 20) *
    (((smudgeWeight669 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
      ((1 : ErrorPoly N) * ((((-1 : Int) ^ (z o / 4) : Int) : ErrorPoly N) *
        smudgeTable ⟨z o % 4, Nat.mod_lt _ (by norm_num)⟩)))

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem smudgeTotal2 {model : HashModel} {params : Stage_online_eval_0_1.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h663 : ∀ i : Fin 800, Stage_online_eval_0_1.parallel_generatedRoot_663 Backend.backend model
      params i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h664 : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_664 Backend.backend model
      params o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {bot0' bot1' g0 g1 g2 g3 g4 g5 : Fin 800 → ExactMatrix Q0 N 1 6}
    (h667 : ∀ i : Fin 800, Stage_online_eval_0_1.parallel_generatedRoot_667 Backend.backend model
      params i (ca0, aA00, aA10, ca1, aA01, aA11, m0, m2, m4, apay, m1, m3, m5, key, bot0' i,
        bot1' i, ()) (g0 i, g1 i, g2 i, g3 i, g4 i, g5 i, ()))
    {n0 n1 n2 n3 n4 n5 : Fin 160 → ExactMatrix Q0 N 1 6}
    (h668 : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_668 Backend.backend model
      params o (g0, g1, g2, g3, g4, g5, ()) (n0 o, n1 o, n2 o, n3 o, n4 o, n5 o, ()))
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
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 smudge2Rhs0 i j +
        ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int} {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6} {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model key Q0 6 smudge2Rhs1 i j +
        ![bot0' i, bot1' i] j =
      reducePoly Q0 N (((q3 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {hl0 hl1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {u0 u1 u2 u3 u4 u5 : Fin 160 → ExactMatrix Q1 N 1 1}
    (hlut : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_669 Backend.backend model
      params o (fam00, fam01, fam10, fam11, hl0, key, hl1, n4 o, n0 o, n2 o, n5 o, n1 o, n3 o, ())
        (u0 o, u1 o, u2 o, u3 o, u4 o, u5 o, ()))
    {g1' : ExactMatrix Q1 N 1 6} (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1')
    (hh5 : centeredRebaseRuns g1' h5)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![hl0 i, hl1 i] Bh)
    {gq : ExactMatrix Q1 N 1 6} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (smudgeTable k)) (sigma i).toNat *
          reducePoly Q1 N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    Rel sm ![![∑ o, u0 o, ∑ o, u1 o], ![∑ o, u2 o, ∑ o, u3 o]]
      (fun j ↦ reducePoly Q1 N (smudgeValue2 (fun o ↦ a (sel' o) + a (sel o)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![∑ o, u4 o, ∑ o, u5 o] (160 * smudge2LutBound Ba Bs BB BB' Bh BS) := by
  have hmem : ∀ o ∈ (Finset.univ : Finset (Fin 160)), Rel sm ![![u0 o, u1 o], ![u2 o, u3 o]]
      (fun j ↦ reducePoly Q1 N (Xe (n := N) (o / 20)) • reducePoly Q1 N
        (((smudgeWeight669 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
          ((1 : ErrorPoly N) * ((((-1 : Int) ^ ((a (sel' o) + a (sel o)) / 4) : Int) : ErrorPoly N) *
            smudgeTable ⟨(a (sel' o) + a (sel o)) % 4, Nat.mod_lt _ (by norm_num)⟩)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![u4 o, u5 o] (smudge2LutBound Ba Bs BB BB' Bh BS) := by
    intro o _
    have hm := smudge2Member1 h663 h664 h667 h668 hsm hG hg6' hg7 hg5' hh6 ha hpay hone hrhs heB hrhs'
      heB' hsel hsel' o
    exact lut669 (hlut o) hsm hg1' hh5 hm hhelp o.isLt hgq hinverse hswitch
  have hsum := Rel.sum Finset.univ hmem
  rw [blocks_sum, pair_sum] at hsum
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  rw [Finset.sum_apply, smudgeValue2, Finset.sum_mul, map_sum, Finset.sum_smul]
  refine Finset.sum_congr rfl fun o _ ↦ ?_
  rw [smul_smul, ← map_mul]
  congr 2
  ring

/-- The bound of one member's LUT output of decoder `3`. -/
abbrev smudge3LutBound (Ba Bs BB BB' Bh BS : Nat) : Nat :=
  (6 * ((3 * lutBranchBound723 (5 * smudge3Cell1Bound Ba (5 * smudge3Cell0Bound Ba Bs BB) BB') Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * (5 * smudge3Cell1Bound Ba (5 * smudge3Cell0Bound Ba Bs BB) BB') * 16384 +
    (N * 1 * (q3 / 2) + q3 / 2)) + (N + 2) * (q3 / 2)) / q3 + 4 * (N * 1 * BS)) * (32768 / 2))

/-- The smudge of decoder `3`: `Σ_o X^⌊o/20⌋ w_(o mod 20) T'(z_o)`. -/
noncomputable def smudgeValue3 (z : Fin 160 → Nat) : ErrorPoly N :=
  ∑ o : Fin 160, Xe (n := N) (o / 20) *
    (((smudgeWeight723 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
      ((1 : ErrorPoly N) * ((((-1 : Int) ^ (z o / 4) : Int) : ErrorPoly N) *
        smudgeTable ⟨z o % 4, Nat.mod_lt _ (by norm_num)⟩)))

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem smudgeTotal3 {model : HashModel} {params : Stage_online_eval_0_1.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h717 : ∀ i : Fin 800, Stage_online_eval_0_1.parallel_generatedRoot_717 Backend.backend model
      params i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h718 : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_718 Backend.backend model
      params o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {bot0' bot1' g0 g1 g2 g3 g4 g5 : Fin 800 → ExactMatrix Q0 N 1 6}
    (h721 : ∀ i : Fin 800, Stage_online_eval_0_1.parallel_generatedRoot_721 Backend.backend model
      params i (ca0, aA00, aA10, ca1, aA01, aA11, m0, m2, m4, apay, m1, m3, m5, key, bot0' i,
        bot1' i, ()) (g0 i, g1 i, g2 i, g3 i, g4 i, g5 i, ()))
    {n0 n1 n2 n3 n4 n5 : Fin 160 → ExactMatrix Q0 N 1 6}
    (h722 : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_722 Backend.backend model
      params o (g0, g1, g2, g3, g4, g5, ()) (n0 o, n1 o, n2 o, n3 o, n4 o, n5 o, ()))
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
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 smudge3Rhs0 i j +
        ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int} {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6} {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model key Q0 6 smudge3Rhs1 i j +
        ![bot0' i, bot1' i] j =
      reducePoly Q0 N (((q3 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {hl0 hl1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {u0 u1 u2 u3 u4 u5 : Fin 160 → ExactMatrix Q1 N 1 1}
    (hlut : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_723 Backend.backend model
      params o (fam00, fam01, fam10, fam11, hl0, key, hl1, n4 o, n0 o, n2 o, n5 o, n1 o, n3 o, ())
        (u0 o, u1 o, u2 o, u3 o, u4 o, u5 o, ()))
    {g1' : ExactMatrix Q1 N 1 6} (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1')
    (hh5 : centeredRebaseRuns g1' h5)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![hl0 i, hl1 i] Bh)
    {gq : ExactMatrix Q1 N 1 6} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (smudgeTable k)) (sigma i).toNat *
          reducePoly Q1 N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    Rel sm ![![∑ o, u0 o, ∑ o, u1 o], ![∑ o, u2 o, ∑ o, u3 o]]
      (fun j ↦ reducePoly Q1 N (smudgeValue3 (fun o ↦ a (sel' o) + a (sel o)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![∑ o, u4 o, ∑ o, u5 o] (160 * smudge3LutBound Ba Bs BB BB' Bh BS) := by
  have hmem : ∀ o ∈ (Finset.univ : Finset (Fin 160)), Rel sm ![![u0 o, u1 o], ![u2 o, u3 o]]
      (fun j ↦ reducePoly Q1 N (Xe (n := N) (o / 20)) • reducePoly Q1 N
        (((smudgeWeight723 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
          ((1 : ErrorPoly N) * ((((-1 : Int) ^ ((a (sel' o) + a (sel o)) / 4) : Int) : ErrorPoly N) *
            smudgeTable ⟨(a (sel' o) + a (sel o)) % 4, Nat.mod_lt _ (by norm_num)⟩)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![u4 o, u5 o] (smudge3LutBound Ba Bs BB BB' Bh BS) := by
    intro o _
    have hm := smudge3Member1 h717 h718 h721 h722 hsm hG hg6' hg7 hg5' hh6 ha hpay hone hrhs heB hrhs'
      heB' hsel hsel' o
    exact lut723 (hlut o) hsm hg1' hh5 hm hhelp o.isLt hgq hinverse hswitch
  have hsum := Rel.sum Finset.univ hmem
  rw [blocks_sum, pair_sum] at hsum
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  rw [Finset.sum_apply, smudgeValue3, Finset.sum_mul, map_sum, Finset.sum_smul]
  refine Finset.sum_congr rfl fun o _ ↦ ?_
  rw [smul_smul, ← map_mul]
  congr 2
  ring

/-- The bound of one member's LUT output of decoder `4`. -/
abbrev smudge4LutBound (Ba Bs BB BB' Bh BS : Nat) : Nat :=
  (6 * ((3 * lutBranchBound777 (5 * smudge4Cell1Bound Ba (5 * smudge4Cell0Bound Ba Bs BB) BB') Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * (5 * smudge4Cell1Bound Ba (5 * smudge4Cell0Bound Ba Bs BB) BB') * 16384 +
    (N * 1 * (q3 / 2) + q3 / 2)) + (N + 2) * (q3 / 2)) / q3 + 4 * (N * 1 * BS)) * (32768 / 2))

/-- The smudge of decoder `4`: `Σ_o X^⌊o/20⌋ w_(o mod 20) T'(z_o)`. -/
noncomputable def smudgeValue4 (z : Fin 160 → Nat) : ErrorPoly N :=
  ∑ o : Fin 160, Xe (n := N) (o / 20) *
    (((smudgeWeight777 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
      ((1 : ErrorPoly N) * ((((-1 : Int) ^ (z o / 4) : Int) : ErrorPoly N) *
        smudgeTable ⟨z o % 4, Nat.mod_lt _ (by norm_num)⟩)))

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem smudgeTotal4 {model : HashModel} {params : Stage_online_eval_0_1.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h771 : ∀ i : Fin 800, Stage_online_eval_0_1.parallel_generatedRoot_771 Backend.backend model
      params i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h772 : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_772 Backend.backend model
      params o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {bot0' bot1' g0 g1 g2 g3 g4 g5 : Fin 800 → ExactMatrix Q0 N 1 6}
    (h775 : ∀ i : Fin 800, Stage_online_eval_0_1.parallel_generatedRoot_775 Backend.backend model
      params i (ca0, aA00, aA10, ca1, aA01, aA11, m0, m2, m4, apay, m1, m3, m5, key, bot0' i,
        bot1' i, ()) (g0 i, g1 i, g2 i, g3 i, g4 i, g5 i, ()))
    {n0 n1 n2 n3 n4 n5 : Fin 160 → ExactMatrix Q0 N 1 6}
    (h776 : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_776 Backend.backend model
      params o (g0, g1, g2, g3, g4, g5, ()) (n0 o, n1 o, n2 o, n3 o, n4 o, n5 o, ()))
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
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 smudge4Rhs0 i j +
        ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int} {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6} {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model key Q0 6 smudge4Rhs1 i j +
        ![bot0' i, bot1' i] j =
      reducePoly Q0 N (((q3 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {hl0 hl1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {u0 u1 u2 u3 u4 u5 : Fin 160 → ExactMatrix Q1 N 1 1}
    (hlut : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_777 Backend.backend model
      params o (fam00, fam01, fam10, fam11, hl0, key, hl1, n4 o, n0 o, n2 o, n5 o, n1 o, n3 o, ())
        (u0 o, u1 o, u2 o, u3 o, u4 o, u5 o, ()))
    {g1' : ExactMatrix Q1 N 1 6} (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1')
    (hh5 : centeredRebaseRuns g1' h5)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![hl0 i, hl1 i] Bh)
    {gq : ExactMatrix Q1 N 1 6} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (smudgeTable k)) (sigma i).toNat *
          reducePoly Q1 N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    Rel sm ![![∑ o, u0 o, ∑ o, u1 o], ![∑ o, u2 o, ∑ o, u3 o]]
      (fun j ↦ reducePoly Q1 N (smudgeValue4 (fun o ↦ a (sel' o) + a (sel o)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![∑ o, u4 o, ∑ o, u5 o] (160 * smudge4LutBound Ba Bs BB BB' Bh BS) := by
  have hmem : ∀ o ∈ (Finset.univ : Finset (Fin 160)), Rel sm ![![u0 o, u1 o], ![u2 o, u3 o]]
      (fun j ↦ reducePoly Q1 N (Xe (n := N) (o / 20)) • reducePoly Q1 N
        (((smudgeWeight777 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
          ((1 : ErrorPoly N) * ((((-1 : Int) ^ ((a (sel' o) + a (sel o)) / 4) : Int) : ErrorPoly N) *
            smudgeTable ⟨(a (sel' o) + a (sel o)) % 4, Nat.mod_lt _ (by norm_num)⟩)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![u4 o, u5 o] (smudge4LutBound Ba Bs BB BB' Bh BS) := by
    intro o _
    have hm := smudge4Member1 h771 h772 h775 h776 hsm hG hg6' hg7 hg5' hh6 ha hpay hone hrhs heB hrhs'
      heB' hsel hsel' o
    exact lut777 (hlut o) hsm hg1' hh5 hm hhelp o.isLt hgq hinverse hswitch
  have hsum := Rel.sum Finset.univ hmem
  rw [blocks_sum, pair_sum] at hsum
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  rw [Finset.sum_apply, smudgeValue4, Finset.sum_mul, map_sum, Finset.sum_smul]
  refine Finset.sum_congr rfl fun o _ ↦ ?_
  rw [smul_smul, ← map_mul]
  congr 2
  ring

/-- The bound of one member's LUT output of decoder `5`. -/
abbrev smudge5LutBound (Ba Bs BB BB' Bh BS : Nat) : Nat :=
  (6 * ((3 * lutBranchBound831 (5 * smudge5Cell1Bound Ba (5 * smudge5Cell0Bound Ba Bs BB) BB') Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * (5 * smudge5Cell1Bound Ba (5 * smudge5Cell0Bound Ba Bs BB) BB') * 16384 +
    (N * 1 * (q3 / 2) + q3 / 2)) + (N + 2) * (q3 / 2)) / q3 + 4 * (N * 1 * BS)) * (32768 / 2))

/-- The smudge of decoder `5`: `Σ_o X^⌊o/20⌋ w_(o mod 20) T'(z_o)`. -/
noncomputable def smudgeValue5 (z : Fin 160 → Nat) : ErrorPoly N :=
  ∑ o : Fin 160, Xe (n := N) (o / 20) *
    (((smudgeWeight831 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
      ((1 : ErrorPoly N) * ((((-1 : Int) ^ (z o / 4) : Int) : ErrorPoly N) *
        smudgeTable ⟨z o % 4, Nat.mod_lt _ (by norm_num)⟩)))

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem smudgeTotal5 {model : HashModel} {params : Stage_online_eval_0_1.Params} {key : ByteArray}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12}
    {apay : Fin 5 → ExactMatrix Q6 N 1 1} {one00 one10 onec0 one01 one11 onec1 : ExactMatrix Q6 N 1 10}
    {bot0 bot1 f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8}
    (h825 : ∀ i : Fin 800, Stage_online_eval_0_1.parallel_generatedRoot_825 Backend.backend model
      params i (ca0, aA00, aA10, ca1, aA01, aA11, one00, one10, onec0, apay, one01, one11, onec1,
        key, bot0 i, bot1 i, ()) (f0 i, f1 i, f2 i, f3 i, f4 i, f5 i, ()))
    {m0 m1 m2 m3 m4 m5 : Fin 160 → ExactMatrix Q5 N 1 8}
    (h826 : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_826 Backend.backend model
      params o (f0, f1, f2, f3, f4, f5, ()) (m0 o, m1 o, m2 o, m3 o, m4 o, m5 o, ()))
    {bot0' bot1' g0 g1 g2 g3 g4 g5 : Fin 800 → ExactMatrix Q0 N 1 6}
    (h829 : ∀ i : Fin 800, Stage_online_eval_0_1.parallel_generatedRoot_829 Backend.backend model
      params i (ca0, aA00, aA10, ca1, aA01, aA11, m0, m2, m4, apay, m1, m3, m5, key, bot0' i,
        bot1' i, ()) (g0 i, g1 i, g2 i, g3 i, g4 i, g5 i, ()))
    {n0 n1 n2 n3 n4 n5 : Fin 160 → ExactMatrix Q0 N 1 6}
    (h830 : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_830 Backend.backend model
      params o (g0, g1, g2, g3, g4, g5, ()) (n0 o, n1 o, n2 o, n3 o, n4 o, n5 o, ()))
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
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model key Q5 8 smudge5Rhs0 i j +
        ![bot0 i, bot1 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int} {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6} {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model key Q0 6 smudge5Rhs1 i j +
        ![bot0' i, bot1' i] j =
      reducePoly Q0 N (((q3 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {hl0 hl1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {u0 u1 u2 u3 u4 u5 : Fin 160 → ExactMatrix Q1 N 1 1}
    (hlut : ∀ o : Fin 160, Stage_online_eval_0_1.parallel_generatedRoot_831 Backend.backend model
      params o (fam00, fam01, fam10, fam11, hl0, key, hl1, n4 o, n0 o, n2 o, n5 o, n1 o, n3 o, ())
        (u0 o, u1 o, u2 o, u3 o, u4 o, u5 o, ()))
    {g1' : ExactMatrix Q1 N 1 6} (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1')
    (hh5 : centeredRebaseRuns g1' h5)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![hl0 i, hl1 i] Bh)
    {gq : ExactMatrix Q1 N 1 6} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (smudgeTable k)) (sigma i).toNat *
          reducePoly Q1 N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    Rel sm ![![∑ o, u0 o, ∑ o, u1 o], ![∑ o, u2 o, ∑ o, u3 o]]
      (fun j ↦ reducePoly Q1 N (smudgeValue5 (fun o ↦ a (sel' o) + a (sel o)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![∑ o, u4 o, ∑ o, u5 o] (160 * smudge5LutBound Ba Bs BB BB' Bh BS) := by
  have hmem : ∀ o ∈ (Finset.univ : Finset (Fin 160)), Rel sm ![![u0 o, u1 o], ![u2 o, u3 o]]
      (fun j ↦ reducePoly Q1 N (Xe (n := N) (o / 20)) • reducePoly Q1 N
        (((smudgeWeight831 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
          ((1 : ErrorPoly N) * ((((-1 : Int) ^ ((a (sel' o) + a (sel o)) / 4) : Int) : ErrorPoly N) *
            smudgeTable ⟨(a (sel' o) + a (sel o)) % 4, Nat.mod_lt _ (by norm_num)⟩)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![u4 o, u5 o] (smudge5LutBound Ba Bs BB BB' Bh BS) := by
    intro o _
    have hm := smudge5Member1 h825 h826 h829 h830 hsm hG hg6' hg7 hg5' hh6 ha hpay hone hrhs heB hrhs'
      heB' hsel hsel' o
    exact lut831 (hlut o) hsm hg1' hh5 hm hhelp o.isLt hgq hinverse hswitch
  have hsum := Rel.sum Finset.univ hmem
  rw [blocks_sum, pair_sum] at hsum
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  rw [Finset.sum_apply, smudgeValue5, Finset.sum_mul, map_sum, Finset.sum_smul]
  refine Finset.sum_congr rfl fun o _ ↦ ?_
  rw [smul_smul, ← map_mul]
  congr 2
  ring

end RefreshTopProof.Part_0_1
