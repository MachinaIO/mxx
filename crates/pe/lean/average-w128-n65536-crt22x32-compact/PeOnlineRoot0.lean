import PeOnlineGlue
import PeOnlineStep0

/-!
# The first online step at the root

The root wires of the first step: the state `(w₈, w₉)` under the hashed input key, the product with
the attribute ciphertext of the first position, and the 128 branches with the first table.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxPe Stage_online_eval

variable {model : HashModel} {tape : SampleTape} {path : List Nat} {params : Params}

/-- The first-depth branch data of the first step. -/
noncomputable def data1_0 (ctx : generatedRoot.Ctx Backend.backend model tape path params) : BranchData :=
  ⟨generatedRoot.w_55_0 ctx, generatedRoot.w_56_0 ctx, generatedRoot.w_57_0 ctx, generatedRoot.w_58_0 ctx,
    generatedRoot.w_11_0 ctx, generatedRoot.w_35_0 ctx, generatedRoot.w_45_0 ctx, generatedRoot.w_64_0 ctx⟩

/-- The second-depth leaf data of the first step. -/
noncomputable def data2_0 (ctx : generatedRoot.Ctx Backend.backend model tape path params) : Leaf2Data :=
  ⟨generatedRoot.w_55_0 ctx, generatedRoot.w_56_0 ctx, generatedRoot.w_57_0 ctx, generatedRoot.w_58_0 ctx,
    generatedRoot.w_11_0 ctx, generatedRoot.w_65_0 ctx, generatedRoot.w_65_1 ctx, generatedRoot.w_69_0 ctx⟩

/-- The preprocessed product key of the first step. -/
noncomputable def mKey_0 (ctx : generatedRoot.Ctx Backend.backend model tape path params) :
    Fin 2 → Fin 2 → ExactMatrix Q N 1 44 :=
  ![![generatedRoot.w_47_0 ctx, generatedRoot.w_49_0 ctx], ![generatedRoot.w_51_0 ctx, generatedRoot.w_53_0 ctx]]

set_option maxRecDepth 200000 in
set_option maxHeartbeats 0 in
theorem online_root0 {ctx : generatedRoot.Ctx Backend.backend model tape path params}
    (F0 : generatedRoot.Facts_0 ctx) (F1 : generatedRoot.Facts_1 ctx)
    {s : ErrorPoly N} (hs : polyNorm s ≤ 1) {v B : Nat}
    (henc : Rel s (inputKey model (generatedRoot.w_11_0 ctx))
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * Xe v * unitPair s j) • g0)
      ![generatedRoot.w_8_0 ctx, generatedRoot.w_9_0 ctx] B)
    (p : Fin 4) (hp : generatedRoot.w_13_0 ctx 0 = p) {xv : Nat} {eB : Fin 2 → ErrorMatrix N 1 44}
    (hrhs : ∀ j, reducePoly Q N s • ![rhsTop model (generatedRoot.w_11_0 ctx) p 0,
        rhsTop model (generatedRoot.w_11_0 ctx) p 1] j +
      ![generatedRoot.w_18_0 ctx p, generatedRoot.w_20_0 ctx p] j =
      reducePoly Q N (Xe xv * unitPair s j) • g0 + reduceMatrix Q N 1 44 (eB j))
    (heB : ∀ j, CoeffBound (eB j) 31)
    {ez0 : Fin 15488 → Fin 2 → ErrorMatrix N 1 1}
    (hcomp0 : ∀ i : Fin 15488,
      RelE s (rhsCompKey model (generatedRoot.w_11_0 ctx) ((i / 3872 : Nat) : Int) ((i % 88 : Nat) : Int)
          ((44 + (i / 88) % 44 : Nat) : Int))
        (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
          (entryDigit (joinedEntry ![generatedRoot.w_18_0 ctx ⟨i / 3872, by omega⟩,
              generatedRoot.w_20_0 ctx ⟨i / 3872, by omega⟩] ⟨i % 88, by omega⟩)
            ⟨(i / 88) % 44, by omega⟩ * unitPair s j)))
        ![generatedRoot.w_25_0 ctx i, generatedRoot.w_26_0 ctx i] (ez0 i) ∧ ∀ j, CoeffBound (ez0 i j) 31)
    (hM : ∀ r (j : Fin 2) (l : Fin 44), mKey_0 ctx r j 0 l =
      ∑ a, keyCompanionSum model (generatedRoot.w_11_0 ctx) p (inputKey model (generatedRoot.w_11_0 ctx)) r
        ⟨44 * (j : Nat) + l, by omega⟩ a +
      ∑ a : Fin 44, inputKey model (generatedRoot.w_11_0 ctx) r 0 0 a *
        rowDigits (joinRow ![rhsTop model (generatedRoot.w_11_0 ctx) p 0,
          rhsTop model (generatedRoot.w_11_0 ctx) p 1]) a ⟨44 * (j : Nat) + l, by omega⟩)
    (hhelp : ∀ (i : Fin 37) (k : Nat) (hk : Odd k), kHelp i = k →
      Rel s (helperKey model (generatedRoot.w_11_0 ctx) k)
        (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * unitPair (substitution k hk s) j) • g0)
        ![generatedRoot.w_55_0 ctx i, generatedRoot.w_56_0 ctx i] 31)
    {ez : Fin 85184 → Fin 2 → ErrorMatrix N 1 1}
    (hcomp : ∀ i : Fin 85184,
      RelE (substitution (kSw (i / 3872)) (kSw_odd ⟨i / 3872, by omega⟩) s)
        (lutCompKey model (generatedRoot.w_11_0 ctx) (kSw (i / 3872)) ((i % 88 : Nat) : Int)
          ((44 + (i / 88) % 44 : Nat) : Int))
        (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
          (entryDigit (joinedEntry ![switchBot model (generatedRoot.w_11_0 ctx) (generatedRoot.w_55_0 ctx)
                (generatedRoot.w_56_0 ctx) ⟨i / 3872, by omega⟩ 0,
              switchBot model (generatedRoot.w_11_0 ctx) (generatedRoot.w_55_0 ctx)
                (generatedRoot.w_56_0 ctx) ⟨i / 3872, by omega⟩ 1] ⟨i % 88, by omega⟩)
            ⟨(i / 88) % 44, by omega⟩ *
            unitPair (substitution (kSw (i / 3872)) (kSw_odd ⟨i / 3872, by omega⟩) s) j)))
        ![generatedRoot.w_60_0 ctx i, generatedRoot.w_61_0 ctx i] (ez i) ∧ ∀ j, CoeffBound (ez i j) 31)
    (hT1 : ∀ (b : Fin 22) r jj, ![(data1_0 ctx).Tb0 b b.isLt, (data1_0 ctx).Tb1 b b.isLt] r 0 jj =
      ∑ a, lutKeyCompanionSum model (generatedRoot.w_11_0 ctx) b (mKey_0 ctx) r jj a +
      ∑ a : Fin 44, substitution (kSw b) (kSw_odd b) (mKey_0 ctx r 0 0 a) *
        rowDigits (switchTopRow model (generatedRoot.w_11_0 ctx) b) a jj)
    (hT2 : ∀ (l : Fin 105) r jj, ![(data2_0 ctx).Tb0 l l.isLt, (data2_0 ctx).Tb1 l l.isLt] r 0 jj =
      ∑ a, lutKeyCompanionSum model (generatedRoot.w_11_0 ctx) (member2 l l.isLt)
        ((data1_0 ctx).key1 model ⟨l % 15, by omega⟩) r jj a +
      ∑ a : Fin 44, substitution (kSw (15 + l / 15)) (kSw_odd (member2 l l.isLt))
        ((data1_0 ctx).key1 model ⟨l % 15, by omega⟩ r 0 0 a) *
        rowDigits (switchTopRow model (generatedRoot.w_11_0 ctx) (member2 l l.isLt)) a jj)
    (ht : ∀ j : Fin 128, 0 ≤ generatedRoot.w_38_0 ctx ⟨0 + j, by omega⟩ ∧
      generatedRoot.w_38_0 ctx ⟨0 + j, by omega⟩ < 2 * N)
    (he : v + xv < 128) :
    Rel s (stepKey model (mKey_0 ctx) (data1_0 ctx) (data2_0 ctx)
        (fun k ↦ branchCoef (generatedRoot.w_38_0 ctx) 0 k (by norm_num)))
      (fun j ↦ (reducePoly Q N (unitPair s j) *
        tableMono (generatedRoot.w_38_0 ctx) 0 (by norm_num) ⟨v + xv, he⟩) • g0)
      ![generatedRoot.w_70_0 ctx, generatedRoot.w_70_1 ctx]
      (stepBound (productBound B companionBound) (bound1 (productBound B companionBound))
        (bound2 (productBound B companionBound))) := by
  -- The position of the step.
  have hw14 : generatedRoot.w_14_0 ctx = ((p : Nat) : Int) := by
    obtain ⟨q, hq, hout⟩ := F0.h14_0
    rw [hout, ← hp, Fin.ext (by omega : (q : Nat) = (0 : Fin 4))]
  have getp : ∀ {α : Type} {f : Fin 4 → α} {o : α}, familyGetDynamic f (generatedRoot.w_14_0 ctx) o →
      o = f p := by
    intro α f o ⟨q, hq, ho⟩
    rw [ho, Fin.ext (by omega : (q : Nat) = p)]
  -- The public top row and the bottom row of the attribute ciphertext.
  obtain ⟨a12, b12, h12a, h12b, h12o⟩ := F0.h12_1 p
  simp only [Prod.mk.injEq] at h12o
  have hw15 : generatedRoot.w_15_0 ctx = rhsTop model (generatedRoot.w_11_0 ctx) p 0 := by
    rw [getp F0.h15_0, h12o.1]; exact hashSample_eq h12a
  have hw16 : generatedRoot.w_16_0 ctx = rhsTop model (generatedRoot.w_11_0 ctx) p 1 := by
    rw [getp F0.h16_0, h12o.2.1]; exact hashSample_eq h12b
  have hw17 : generatedRoot.w_17_0 ctx = joinRow ![rhsTop model (generatedRoot.w_11_0 ctx) p 0,
      rhsTop model (generatedRoot.w_11_0 ctx) p 1] := by
    have h := F0.h17_0
    rw [hw15, hw16] at h
    exact concat_joinRow (r := ![_, _]) h
  have hw22 : generatedRoot.w_22_0 ctx = joinRow ![generatedRoot.w_18_0 ctx p, generatedRoot.w_20_0 ctx p] := by
    have h := F0.h22_0
    rw [getp F0.h19_0, getp F0.h21_0] at h
    exact concat_joinRow (r := ![_, _]) h
  -- The hashed input key.
  have hA0 : concatColumns (inputKey model (generatedRoot.w_11_0 ctx) 0 0)
      (inputKey model (generatedRoot.w_11_0 ctx) 0 1) (generatedRoot.w_29_0 ctx) := by
    have h := F0.h29_0
    rw [hashSample_eq F0.h27_0, hashSample_eq F0.h28_0] at h
    exact h
  have hA1 : concatColumns (inputKey model (generatedRoot.w_11_0 ctx) 1 0)
      (inputKey model (generatedRoot.w_11_0 ctx) 1 1) (generatedRoot.w_32_0 ctx) := by
    have h := F0.h32_0
    rw [hashSample_eq F0.h30_0, hashSample_eq F0.h31_0] at h
    exact h
  -- The product.
  have hcol := fun jj ↦ Step0.colLoop0 F0.h23_0 jj
  rw [hw17, hw22] at hcol
  have h33 := F0.h33_0
  rw [hw14] at h33
  have hmul := fun jj ↦ Step0.mulLoop0 hA0 hA1 h33 jj
  have hprod := product_glue hs F0.h10_0 henc p hrhs heB hcol (zc0 := generatedRoot.w_25_0 ctx)
    (zc1 := generatedRoot.w_26_0 ctx) (ez := ez0) (fun i hi ↦ by
      have e : (⟨(i : Nat) / 3872, by omega⟩ : Fin 4) = p := Fin.ext hi
      have h := hcomp0 i
      rw [e] at h
      exact h) hmul hM
  have hw35 : generatedRoot.w_35_0 ctx = rowWindow 44 0 (by omega) (generatedRoot.w_34_0 ctx) :=
    sliceMatrix_window 0 F1.h35_0 rfl (by omega)
  have hw45 : generatedRoot.w_45_0 ctx = rowWindow 44 44 (by omega) (generatedRoot.w_34_0 ctx) :=
    sliceMatrix_window 44 F1.h45_0 rfl (by omega)
  have hP : Rel s (mKey_0 ctx)
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * Xe (v + xv) * unitPair s j) • g0)
      ![(data1_0 ctx).c0, (data1_0 ctx).c1] (productBound B companionBound) := by
    show Rel s _ _ ![generatedRoot.w_35_0 ctx, generatedRoot.w_45_0 ctx] _
    rw [hw35, hw45]
    exact hprod
  -- The identity coefficient and its scalar.
  have hc39 := Step0.coefLoop0 F1.h39_0
  have hV : generatedRoot.w_43_0 ctx = scalarV (branchCoef (generatedRoot.w_38_0 ctx) 0 1 (by norm_num)) := by
    refine scalarV_eq F1.h41_0 ?_
    have h43 := F1.h43_0
    change gadgetDecomposeRuns _ _ _ (matrixMulScalarLeft (matrixMulScalarLeft winvPoly
      (generatedRoot.w_39_0 ctx)) (generatedRoot.w_41_0 ctx)) _ at h43
    rw [hc39] at h43
    exact h43
  -- The branches.
  have hint : ∀ b : Fin 15, (data1_0 ctx).Runs model b (by omega) ∧
      (∀ j, ![generatedRoot.w_65_0 ctx b, generatedRoot.w_65_1 ctx b] j = (data1_0 ctx).row model b (by omega) j) ∧
      ∀ r j, ![![generatedRoot.w_65_2 ctx b, generatedRoot.w_65_3 ctx b],
        ![generatedRoot.w_65_4 ctx b, generatedRoot.w_65_5 ctx b]] r j =
        (data1_0 ctx).key1 model ⟨b, by omega⟩ r j := fun b ↦ by
    obtain ⟨r1, r2, r3, r4⟩ := Step0.internalBranch0 b.isLt (F1.h65_5 b)
    exact ⟨⟨r1, r2⟩, r3, r4⟩
  obtain ⟨hleaf1, h67⟩ := Step0.leafLoop67 (data1_0 ctx) F1.h67_5
  obtain ⟨hleaf2, h70⟩ := Step0.leafLoop70 (data2_0 ctx) F1.h70_5
  have h66 := Step0.sumLoop66 F1.h66_5
  simp only [Prod.mk.injEq] at h66 h67 h70
  have hR : ∀ j, ![generatedRoot.w_70_0 ctx, generatedRoot.w_70_1 ctx] j =
      ![(data1_0 ctx).c0, (data1_0 ctx).c1] j * scalarV (branchCoef (generatedRoot.w_38_0 ctx) 0 1 (by norm_num)) +
      ∑ b : Fin 15, ![generatedRoot.w_65_0 ctx b, generatedRoot.w_65_1 ctx b] j *
        scalarV (branchCoef (generatedRoot.w_38_0 ctx) 0 (kSw b) (by norm_num)) +
      ∑ i : Fin 7, (data1_0 ctx).row model (15 + i) (by omega) j *
        scalarV (branchCoef (generatedRoot.w_38_0 ctx) 0 (kSw (15 + i)) (by norm_num)) +
      ∑ l : Fin 105, (data2_0 ctx).row model l l.isLt j *
        scalarV (branchCoef (generatedRoot.w_38_0 ctx) 0 (kLeaf2 l) (by norm_num)) := by
    intro j
    fin_cases j
    · simp only [Fin.zero_eta, Fin.isValue, Matrix.cons_val_zero]
      rw [h70.1, h67.1, h66.1, ← hV]
      rfl
    · simp only [Fin.mk_one, Fin.isValue, Matrix.cons_val_one]
      rw [h70.2.1, h67.2.1, h66.2.1, ← hV]
      rfl
  refine online_step hs he (data1_0 ctx) (data2_0 ctx) hP hhelp hcomp ?_ hT1 hint hleaf1 rfl ⟨rfl, rfl⟩
    ⟨rfl, rfl⟩ ?_ hT2 hleaf2 (tableMono (generatedRoot.w_38_0 ctx) 0 (by norm_num)) _ ?_ hR
  · exact fun q ↦ Step0.lutLoop0 F1.h62_0 F1.h63_0 F1.h64_0 q
  · exact fun q ↦ Step0.lutLoop2 F1.h69_0 q
  · exact fun k hk ↦ branchCoef_fourier _ 0 (by norm_num) ht k hk

end PeProof
