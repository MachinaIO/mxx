import W_Stage_online_1_Facts
import Gen_W_Stage_online_1_Scopes
import Dio2Generic
import Dio2Sharp
import Dio2Sampler
import Dio2LutBridge
import GenInstLemmas

/-!
# Step-1 PRF composition with arbitrary bucket widths

The generic layer retains arbitrary family sizes and signed-monomial action gain one.
Generated root theorems connect every exported bucket, readout, corrected fused
placement and packed total. Imported label/seed-state encodings, selector RHS,
public selected-label products, switching and Walsh facts remain explicit;
intermediate PRF or packing encodings are derived, not assumed.
-/
namespace Dio2Proof
open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen
open scoped BigOperators

/-- Raw cell semantics, with gain one for the public signed monomial. -/
alias online1_cell := cell_flat_mono

/-- One-hot selection and row summation; arbitrary members, labels and width. -/
alias online1_bucket := RefreshTopProof.GenInst.bucket_of_cells

/-- Readout LUT semantics with arbitrary branch count. -/
alias online1_readout := LutMemberFlat.rel

/-- Placement and total over arbitrary members, retaining all member errors. -/
alias online1_smudge_total := sampler_flat

/-- The conservative sharp bucket recurrence. -/
def online1PrfBound (action root fresh n : Nat) (width : Nat → Nat) : Nat → Nat
  | 0 => root
  | k + 1 => width k * (action * (action * root +
      online1PrfBound action root fresh n width k) + n * fresh)

/-- Propagate the sharp cell/row-sum invariant without unrolling buckets or members.
The stage adapter supplies `hstep` using `online1_cell` and `online1_bucket`, including the
monomial and one-hot premises; the theorem does not infer monomiality from a norm bound. -/
theorem online1_prf_chain {n q m M buckets action root fresh : Nat}
    (width : Nat → Nat) {sm : ErrorPoly n}
    (keys : Nat → Fin M → Fin 2 → Fin 2 → ExactMatrix q n 1 m)
    (target rows : Nat → Fin M → Fin 2 → ExactMatrix q n 1 m)
    (hroot : ∀ o, Rel sm (keys 0 o) (target 0 o) (rows 0 o) root)
    (hstep : ∀ k, k < buckets → ∀ B,
      (∀ o, Rel sm (keys k o) (target k o) (rows k o) B) →
      ∀ o, Rel sm (keys (k + 1) o) (target (k + 1) o) (rows (k + 1) o)
        (width k * (action * (action * root + B) + n * fresh))) :
    ∀ o, Rel sm (keys buckets o) (target buckets o) (rows buckets o)
      (online1PrfBound action root fresh n width buckets) := by
  have h : ∀ k, k ≤ buckets → ∀ o,
      Rel sm (keys k o) (target k o) (rows k o)
        (online1PrfBound action root fresh n width k) := by
    intro k
    induction k with
    | zero => intro _; exact hroot
    | succ k ih =>
      intro hk
      exact hstep k (by omega) _ (ih (by omega))
  exact h buckets le_rfl
/-- Sum the digit encodings after their gadget actions. The target includes the weights,
while the bound sums the retargeting errors, not plaintext magnitudes. -/
theorem online1_smudge_digits {n q m digits B : Nat} {sm : ErrorPoly n}
    (keys : Fin digits → Fin 2 → Fin 2 → ExactMatrix q n 1 m)
    (target rows : Fin digits → Fin 2 → ExactMatrix q n 1 m)
    (h : ∀ r, Rel sm (keys r) (target r) (rows r) B) :
    Rel sm (∑ r, keys r) (∑ r, target r) (∑ r, rows r) (digits * B) := by
  simpa using Rel.sum Finset.univ (fun r _ ↦ h r)

open RefreshTopProof.GenIface RefreshTopProof.GenInst
variable {n : Nat}

set_option maxHeartbeats 8000000 in
/-- Sharp semantic adapter for a PRF cell; only the fixed two encoding components are split. -/
theorem PrfCellFlat.rel_mono (hn : 0 < n) {q m m' m'' bound : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {hashModel : HashModel} {base digits base' digits' : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    {lo two width lo' two' width' lo'' two'' width'' lo''' two''' width''' : Nat}
    {pre rhsPre : Blob} {a b : Int} {i : Nat}
    {idx : Int} {x0 x1 : ExactMatrix q n 1 m} {seed seed' : ByteArray}
    {As00 As10 As01 As11 cs0 cs1 : ExactMatrix q n 1 m'} {pay : ExactMatrix q n 1 1}
    {bot0 bot1 : ExactMatrix q n 1 m''}
    {outputs : ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' ×
      ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' × Unit}
    (h : Dio2Proof.Comp.PrfCellFlat.Comp q n m' m' m' m' m'' m'' m m q q q q m'' m'' m'' m'' m''
      m'' lo two width pre lo' two' width' pre lo'' two'' width'' pre lo''' two''' width''' pre
      base digits bound m' m' m' base digits bound m' m' m' a b rhsPre base' digits' bound m''
      a b rhsPre base' digits' bound m'' m'' m'' m'' m'' backend hashModel i
      (idx, x0, seed, x1, As00, As10, cs0, pay, As01, As11, cs1, seed', bot0, bot1, ()) outputs)
    {sm sp μ α : ErrorPoly n}
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G) {Ba : Nat}
    (hα : SmulGain α 1) (hαμ : polyNorm (α * μ) ≤ 1)
    (ha : Rel sm (labelKeysG hashModel seed q m pre idx)
      (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ![x0, x1] Ba)
    (hpay : pay = fun _ _ ↦ reducePoly q n α)
    {g' : ExactMatrix q n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm ![![As00, As01], ![As10, As11]]
      (fun j ↦ reducePoly q n (μ * unitPair sp j) • g) ![cs0, cs1] Bs)
    {T : Fin 2 → ExactMatrix q n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q n sp • topPairG hashModel seed' q m'' rhsPre (prfTopTags a b i) j +
      ![bot0, bot1] j = T j + reduceMatrix q n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly q n (α * μ) • T j) ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((m' + m') * n * ((m + m) * n * Ba * bound + Bs) * bound + n * 1 * BB) := by
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
    v38, v39, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
    c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38,
    c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, h⟩ := h
  subst h
  -- The label's keys are its root keys of member `idx`.
  have hkeys : ![![v0, v4], ![v1, v5]] = labelKeysG hashModel seed q m pre idx := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 0 0 (by norm_num) c3
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 0 1 (by norm_num) c11
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 1 0 (by norm_num) c6
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 1 1 (by norm_num) c14
  rw [← hkeys] at ha
  have htop := prf_tops c35 c40
  have hrhs' : ∀ j, reducePoly q n sp • ![v26, v31] j + ![bot0, bot1] j =
      T j + reduceMatrix q n 1 m'' (eB j) := by
    rw [htop]; exact hrhs
  exact cell_flat_mono hn hq1 layout hlayout hexact hG hα hαμ ha hpay hg' hg hs c7 c8 c15 c16 c17
    (Xs := ![v9, v17]) (fun j ↦ by fin_cases j; exacts [c18, c26])
    (D' := ![v10, v18]) (fun j ↦ by fin_cases j; exacts [c19, c27])
    (D := ![v11, v19]) (fun j ↦ by fin_cases j; exacts [c20, c28])
    (Y := ![v12, v20]) (fun j ↦ by fin_cases j; exacts [c21, c29])
    (A' := ![![v13, v21], ![v14, v22]]) (fun j ↦ by fin_cases j; exacts [c22, c30])
    (fun j ↦ by fin_cases j; exacts [c23, c31])
    (c' := ![matrixAdd v12 (matrixMulScalarRight cs0 pay),
      matrixAdd v20 (matrixMulScalarRight cs1 pay)]) (fun j ↦ by fin_cases j <;> rfl)
    c24 c25 c32 c33 c34
    (Xb := ![v27, v32]) (fun j ↦ by fin_cases j; exacts [c36, c41])
    (E' := ![v28, v33]) (fun j ↦ by fin_cases j; exacts [c37, c42])
    (E := ![v29, v34]) (fun j ↦ by fin_cases j; exacts [c38, c43])
    hrhs' heB
    (X := ![v38, v39]) (fun j ↦ by fin_cases j; exacts [c47, c48])
    (Mk := ![![v30, v35], ![v36, v37]]) (fun j ↦ by fin_cases j; exacts [c39, c44])
    (fun j ↦ by fin_cases j; exacts [c45, c46])

set_option maxHeartbeats 8000000 in
/-- Sharp semantic adapter for a PRF cell; only the fixed two encoding components are split. -/
theorem PrfCellFlat.rel_1W_mono (hn : 0 < n) {q m m' m'' bound M W : Nat}
    {w : Int} (hw : w = (W : Int)) (hq1 : 1 < q)
    {backend : BackendContext} {hashModel : HashModel} {base digits base' digits' : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    {lo two width lo' two' width' lo'' two'' width'' lo''' two''' width''' : Nat}
    {l0 h0 l1 h1 l2 h2 l3 h3 l4 h4 l5 h5 : Int}
    {pre rhsPre : Blob} {a b : Int} {i : Nat}
    {idx : Int} {x0 x1 : ExactMatrix q n 1 m} {seed seed' : ByteArray}
    {sA00 sA10 sA01 sA11 scs0 scs1 : Fin M → ExactMatrix q n 1 m'} {pay : ExactMatrix q n 1 1}
    {bot0 bot1 : ExactMatrix q n 1 m''}
    {outputs : ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' ×
      ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' × Unit}
    (h : Dio2Proof.Comp.PrfCellFlat.Comp_1W w q n m' m' m' m' m'' m'' m m M q M M M q M M q q m'' m''
      m'' m'' m'' m'' lo two width pre lo' two' width' pre lo'' two'' width'' pre lo''' two''' width'''
      pre l0 h0 l1 h1 base digits bound m' l2 h2 m' m' l3 h3 l4 h4 base digits bound m' l5 h5 m' m'
      a b rhsPre base' digits' bound m'' a b rhsPre base' digits' bound m'' m'' m'' m'' m''
      backend hashModel i
      (idx, x0, seed, x1, sA00, sA10, scs0, pay, sA01, sA11, scs1, seed', bot0, bot1, ()) outputs)
    (hi : i / W < M)
    {sm sp μ α : ErrorPoly n}
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G) {Ba : Nat}
    (hα : SmulGain α 1) (hαμ : polyNorm (α * μ) ≤ 1)
    (ha : Rel sm (labelKeysG hashModel seed q m pre idx)
      (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ![x0, x1] Ba)
    (hpay : pay = fun _ _ ↦ reducePoly q n α)
    {g' : ExactMatrix q n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm ![![sA00 ⟨i / W, hi⟩, sA01 ⟨i / W, hi⟩], ![sA10 ⟨i / W, hi⟩, sA11 ⟨i / W, hi⟩]]
      (fun j ↦ reducePoly q n (μ * unitPair sp j) • g) ![scs0 ⟨i / W, hi⟩, scs1 ⟨i / W, hi⟩] Bs)
    {T : Fin 2 → ExactMatrix q n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q n sp • topPairG hashModel seed' q m'' rhsPre (prfTopTags a b i) j +
      ![bot0, bot1] j = T j + reduceMatrix q n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly q n (α * μ) • T j) ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((m' + m') * n * ((m + m) * n * Ba * bound + Bs) * bound + n * 1 * BB) := by
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
    v38, v39, v40, v41, v42, v43, v44, v45, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13,
    c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32,
    c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51,
    c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, h⟩ := h
  subst h
  have e9 := familyGetDynamic_div hw c20 hi
  have e10 := familyGetDynamic_div hw c23 hi
  have e15 := familyGetDynamic_div hw c30 hi
  have e20 := familyGetDynamic_div hw c37 hi
  have e21 := familyGetDynamic_div hw c40 hi
  have e26 := familyGetDynamic_div hw c47 hi
  subst e9 e10 e15 e20 e21 e26
  have hkeys : ![![v0, v4], ![v1, v5]] = labelKeysG hashModel seed q m pre idx := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 0 0 (by norm_num) c3
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 0 1 (by norm_num) c11
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 1 0 (by norm_num) c6
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 1 1 (by norm_num) c14
  rw [← hkeys] at ha
  have htop := prf_tops c53 c58
  have hrhs' : ∀ j, reducePoly q n sp • ![v32, v37] j + ![bot0, bot1] j =
      T j + reduceMatrix q n 1 m'' (eB j) := by
    rw [htop]; exact hrhs
  exact cell_flat_mono hn hq1 layout hlayout hexact hG hα hαμ ha hpay hg' hg hs c7 c8 c15 c16 c17
    (Xs := ![v11, v22]) (fun j ↦ by fin_cases j; exacts [c24, c41])
    (D' := ![v12, v23]) (fun j ↦ by fin_cases j; exacts [c25, c42])
    (D := ![v13, v24]) (fun j ↦ by fin_cases j; exacts [c26, c43])
    (Y := ![v14, v25]) (fun j ↦ by fin_cases j; exacts [c27, c44])
    (A' := ![![v16, v27], ![v17, v28]]) (fun j ↦ by fin_cases j; exacts [c31, c48])
    (fun j ↦ by fin_cases j; exacts [c32, c49])
    (c' := ![matrixAdd v14 (matrixMulScalarRight (scs0 ⟨i / W, hi⟩) pay),
      matrixAdd v25 (matrixMulScalarRight (scs1 ⟨i / W, hi⟩) pay)]) (fun j ↦ by fin_cases j <;> rfl)
    c33 c34 c50 c51 c52
    (Xb := ![v33, v38]) (fun j ↦ by fin_cases j; exacts [c54, c59])
    (E' := ![v34, v39]) (fun j ↦ by fin_cases j; exacts [c55, c60])
    (E := ![v35, v40]) (fun j ↦ by fin_cases j; exacts [c56, c61])
    hrhs' heB
    (X := ![v44, v45]) (fun j ↦ by fin_cases j; exacts [c65, c66])
    (Mk := ![![v36, v41], ![v42, v43]]) (fun j ↦ by fin_cases j; exacts [c57, c62])
    (fun j ↦ by fin_cases j; exacts [c63, c64])

/- Generated smudge adapters: gen_online1.py. -/
open W_Stage_online_1
set_option maxRecDepth 16384

abbrev online1_bucket0_rhsPre : Blob := ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49])
abbrev online1PrfLabelPre : Blob := ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48])

set_option maxHeartbeats 16000000 in
theorem online1_bucket0 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} {α : Fin 2 → ErrorPoly 8} (hα : ∀ k, SmulGain (α k) 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1)
    {G : ExactMatrix modulus_0 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G) {Ba : Nat}
    (ha : ∀ k : Fin 2, Rel sm (labelKeysG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 online1PrfLabelPre k)
      (fun j ↦ reducePoly modulus_0 8 (α k * unitPair sm j) • G)
      ![generatedRoot.w_16_0 ctx k, generatedRoot.w_27_0 ctx k] Ba)
    (hpay : ∀ k, generatedRoot.w_47_0 ctx k = fun _ _ ↦ reducePoly modulus_0 8 (α k))
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm ![![generatedRoot.w_141_0 ctx, generatedRoot.w_144_0 ctx],
        ![generatedRoot.w_142_0 ctx, generatedRoot.w_145_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (μ * unitPair sp j) • g)
      ![generatedRoot.w_143_0 ctx, generatedRoot.w_146_0 ctx] Bs)
    {ν : Fin 16 → ErrorPoly 8} {hT : ExactMatrix modulus_0 8 1 25}
    {eB : Fin 16 → Fin 2 → ErrorMatrix 8 1 25} {BB : Nat}
    (hrhs : ∀ (i : Fin 16) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_70_0 ctx) modulus_0 25 online1_bucket0_rhsPre (prfTopTags 36550556566445887949135852474979514307313305075533710697673106717124183540354 84715014674746765542951856319120207364894094043218421459706511318570933778708 i) j +
        ![generatedRoot.w_147_0 ctx i, generatedRoot.w_148_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν i * unitPair sp j) • hT +
        reduceMatrix modulus_0 8 1 25 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (sel : Fin 8 → Fin 2) (hsel : ∀ o j, ν (cellAt o j) = if j = sel o then (1 : ErrorPoly 8) else 0)
    (o : Fin 8) :
    ∃ key : Fin 2, ((key : Nat) : Int) = generatedRoot.w_140_0 ctx (cellAt o (sel o)) ∧
      EncRel sm (generatedRoot.w_150_0 ctx o, (generatedRoot.w_150_1 ctx o,
        (generatedRoot.w_150_2 ctx o, (generatedRoot.w_150_3 ctx o, (generatedRoot.w_150_4 ctx o,
          (generatedRoot.w_150_5 ctx o, ()))))))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α key * μ) * unitPair sp i) • hT)
        (2 * ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + Bs) * 2048 + 8 * 1 * BB)) := by
  have hcell : ∀ (o : Fin 8) (j : Fin 2), ∃ k : Fin 2,
      ((k : Nat) : Int) = generatedRoot.w_140_0 ctx (cellAt o j) ∧
      EncRel sm (Tup.ev (k := 6) (generatedRoot.w_149_0 ctx, (generatedRoot.w_149_1 ctx,
          (generatedRoot.w_149_2 ctx, (generatedRoot.w_149_3 ctx, (generatedRoot.w_149_4 ctx,
            (generatedRoot.w_149_5 ctx, ())))))) (cellAt o j))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α k * μ * ν (cellAt o j)) *
          unitPair sp i) • hT)
        ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + Bs) * 2048 + 8 * 1 * BB) := by
    intro o j
    obtain ⟨x0, x1, xp, -, -, g0, -, -, g1, -, -, g2, hc⟩ := F.c6.h149_5 (cellAt o j)
    obtain ⟨k, hk, rfl⟩ := g0
    obtain ⟨k1, hk1, rfl⟩ := g1
    obtain ⟨k2, hk2, rfl⟩ := g2
    have e1 : k = k1 := Fin.ext (by omega)
    have e2 : k = k2 := Fin.ext (by omega)
    subst e1 e2
    refine ⟨k, hk, ?_⟩
    have hak := ha k
    rw [hk] at hak
    have hrel := PrfCellFlat.rel_mono (by norm_num) hq1 layout hlayout hexact
      (parallel_generatedRoot_149.sem hc) hG (hα k) (hαμ k) hak (hpay k) hg' hg hs
      (hrhs (cellAt o j)) (heB _)
    refine encRel_of_rel (hrel.target_eq ?_) (fun a j ↦ by fin_cases a <;> fin_cases j <;> rfl)
      (fun j ↦ by fin_cases j <;> rfl)
    funext i
    rw [smul_smul, ← map_mul]
    congr 2
    ring
  obtain ⟨key, hkey, -⟩ := hcell o (sel o)
  exact ⟨key, hkey, bucket_of_cells (M := 8) (W := 2) (K := 2) (p := 1) α (fun _ ↦ μ)
    (generatedRoot.w_140_0 ctx) ν hcell o (parallel_generatedRoot_150.sem (F.c6.h150_5 o))
    (sel o) (hsel o) key hkey⟩


abbrev online1_bucket1_rhsPre : Blob := ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50])

set_option maxHeartbeats 16000000 in
theorem online1_bucket1 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp : ErrorPoly 8} {μ : Fin 8 → ErrorPoly 8} {α : Fin 2 → ErrorPoly 8}
    (hα : ∀ k, SmulGain (α k) 1) (hαμ : ∀ k o, polyNorm (α k * μ o) ≤ 1)
    {G : ExactMatrix modulus_0 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G) {Ba : Nat}
    (ha : ∀ k : Fin 2, Rel sm (labelKeysG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 online1PrfLabelPre k)
      (fun j ↦ reducePoly modulus_0 8 (α k * unitPair sm j) • G)
      ![generatedRoot.w_16_0 ctx k, generatedRoot.w_27_0 ctx k] Ba)
    (hpay : ∀ k, generatedRoot.w_47_0 ctx k = fun _ _ ↦ reducePoly modulus_0 8 (α k))
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hstate : ∀ o : Fin 8, EncRel sm (generatedRoot.w_150_0 ctx o, (generatedRoot.w_150_1 ctx o,
        (generatedRoot.w_150_2 ctx o, (generatedRoot.w_150_3 ctx o, (generatedRoot.w_150_4 ctx o,
          (generatedRoot.w_150_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ o * unitPair sp j) • g) Bs)
    {ν : Fin 16 → ErrorPoly 8} {hT : ExactMatrix modulus_0 8 1 25}
    {eB : Fin 16 → Fin 2 → ErrorMatrix 8 1 25} {BB : Nat}
    (hrhs : ∀ (i : Fin 16) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_70_0 ctx) modulus_0 25 online1_bucket1_rhsPre (prfTopTags 36550556566445887949135852474979514307313305075533710697673106717124183540354 84715014674746765542951856319120207364894094043218421459706511318570933778708 i) j +
        ![generatedRoot.w_151_0 ctx i, generatedRoot.w_152_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν i * unitPair sp j) • hT +
        reduceMatrix modulus_0 8 1 25 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (sel : Fin 8 → Fin 2) (hsel : ∀ o j, ν (cellAt o j) = if j = sel o then (1 : ErrorPoly 8) else 0)
    (o : Fin 8) :
    ∃ key : Fin 2, ((key : Nat) : Int) = generatedRoot.w_139_0 ctx (cellAt o (sel o)) ∧
      EncRel sm (generatedRoot.w_154_0 ctx o, (generatedRoot.w_154_1 ctx o,
        (generatedRoot.w_154_2 ctx o, (generatedRoot.w_154_3 ctx o, (generatedRoot.w_154_4 ctx o,
          (generatedRoot.w_154_5 ctx o, ()))))))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α key * μ o) * unitPair sp i) • hT)
        (2 * ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + Bs) * 2048 + 8 * 1 * BB)) := by
  have hcell : ∀ (o : Fin 8) (j : Fin 2), ∃ k : Fin 2,
      ((k : Nat) : Int) = generatedRoot.w_139_0 ctx (cellAt o j) ∧
      EncRel sm (Tup.ev (k := 6) (generatedRoot.w_153_0 ctx, (generatedRoot.w_153_1 ctx,
          (generatedRoot.w_153_2 ctx, (generatedRoot.w_153_3 ctx, (generatedRoot.w_153_4 ctx,
            (generatedRoot.w_153_5 ctx, ())))))) (cellAt o j))
        (fun i ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * (α k * μ o * ν (cellAt o j)) *
          unitPair sp i) • hT)
        ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + Bs) * 2048 + 8 * 1 * BB) := by
    intro o j
    obtain ⟨x0, x1, xp, -, -, g0, -, -, g1, -, -, g2, hc⟩ := F.c6.h153_5 (cellAt o j)
    obtain ⟨k, hk, rfl⟩ := g0
    obtain ⟨k1, hk1, rfl⟩ := g1
    obtain ⟨k2, hk2, rfl⟩ := g2
    have e1 : k = k1 := Fin.ext (by omega)
    have e2 : k = k2 := Fin.ext (by omega)
    subst e1 e2
    refine ⟨k, hk, ?_⟩
    have hak := ha k
    rw [hk] at hak
    have hio : (cellAt o j : Nat) / 2 < 8 := by
      have := o.isLt; have := j.isLt; simp only [cellAt]; omega
    have hoo : (⟨(cellAt o j : Nat) / 2, hio⟩ : Fin 8) = o := by
      apply Fin.ext; have := j.isLt; simp only [cellAt]; omega
    have hs := hstate o
    rw [← hoo] at hs
    unfold EncRel at hs
    have hrel := PrfCellFlat.rel_1W_mono (W := 2) (by norm_num) rfl hq1 layout hlayout hexact
      (parallel_generatedRoot_153.sem hc) hio hG (hα k) (by rw [hoo]; exact hαμ k o) hak (hpay k)
      hg' hg (μ := μ ⟨(cellAt o j : Nat) / 2, hio⟩) (hs.target_eq (by funext j; simp))
      (hrhs (cellAt o j)) (heB _)
    rw [hoo] at hrel
    refine encRel_of_rel (hrel.target_eq ?_) (fun a j ↦ by fin_cases a <;> fin_cases j <;> rfl)
      (fun j ↦ by fin_cases j <;> rfl)
    funext i
    rw [smul_smul, ← map_mul]
    congr 2
    ring
  obtain ⟨key, hkey, -⟩ := hcell o (sel o)
  exact ⟨key, hkey, bucket_of_cells (M := 8) (W := 2) (K := 2) (p := 1) α μ
    (generatedRoot.w_139_0 ctx) ν hcell o (parallel_generatedRoot_154.sem (F.c6.h154_5 o))
    (sel o) (hsel o) key hkey⟩

abbrev online1PrfBudget0 (Ba Bs : Nat) (BB : Fin 2 → Nat) : Nat :=
  (2 * ((25 + 25) * 8 *
      ((25 + 25) * 8 * Ba * 2048 + Bs) *
      2048 + 8 * 1 * BB 0))

abbrev online1PrfBudget1 (Ba Bs : Nat) (BB : Fin 2 → Nat) : Nat :=
  (2 * ((25 + 25) * 8 *
      ((25 + 25) * 8 * Ba * 2048 + online1PrfBudget0 Ba Bs BB) *
      2048 + 8 * 1 * BB 1))

set_option maxHeartbeats 400000000 in
/-- Every exported PRF bucket is derived from label, seed-state and selector-RHS premises.
Selected-label products are public semantics, not assumed intermediate encodings. -/
theorem online1_prf_root {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} {α : Fin 2 → ErrorPoly 8} (hα : ∀ k, SmulGain (α k) 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1)
    {G : ExactMatrix modulus_0 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G) {Ba : Nat}
    (ha : ∀ k : Fin 2, Rel sm (labelKeysG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 online1PrfLabelPre k)
      (fun j ↦ reducePoly modulus_0 8 (α k * unitPair sm j) • G)
      ![generatedRoot.w_16_0 ctx k, generatedRoot.w_27_0 ctx k] Ba)
    (hpay : ∀ k, generatedRoot.w_47_0 ctx k = fun _ _ ↦ reducePoly modulus_0 8 (α k))
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm ![![generatedRoot.w_141_0 ctx, generatedRoot.w_144_0 ctx],
        ![generatedRoot.w_142_0 ctx, generatedRoot.w_145_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (μ * unitPair sp j) • g)
      ![generatedRoot.w_143_0 ctx, generatedRoot.w_146_0 ctx] Bs)
    (BB : Fin 2 → Nat)
    {ν0 : Fin 16 → ErrorPoly 8}
    {eB0 : Fin 16 → Fin 2 → ErrorMatrix 8 1 25}
    (hrhs0 : ∀ (i : Fin 16) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_70_0 ctx) modulus_0 25 online1_bucket0_rhsPre (prfTopTags 36550556566445887949135852474979514307313305075533710697673106717124183540354 84715014674746765542951856319120207364894094043218421459706511318570933778708 i) j +
        ![generatedRoot.w_147_0 ctx i, generatedRoot.w_148_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν0 i * unitPair sp j) • g +
        reduceMatrix modulus_0 8 1 25 (eB0 i j))
    (heB0 : ∀ i j, CoeffBound (eB0 i j) (BB 0))
    (sel0 : Fin 8 → Fin 2) (hsel0 : ∀ o j, ν0 (cellAt o j) = if j = sel0 o then (1 : ErrorPoly 8) else 0)
    (selected0 : Fin 8 → Fin 2)
    (hselected0 : ∀ o, ((selected0 o).val : Int) =
      generatedRoot.w_140_0 ctx (cellAt o (sel0 o)))
    (μ0 : Fin 8 → ErrorPoly 8)
    (hproduct0 : ∀ o, μ0 o = α (selected0 o) * (μ))
    (hαμ1 : ∀ k o, polyNorm (α k * μ0 o) ≤ 1)
    {ν1 : Fin 16 → ErrorPoly 8}
    {eB1 : Fin 16 → Fin 2 → ErrorMatrix 8 1 25}
    (hrhs1 : ∀ (i : Fin 16) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_70_0 ctx) modulus_0 25 online1_bucket1_rhsPre (prfTopTags 36550556566445887949135852474979514307313305075533710697673106717124183540354 84715014674746765542951856319120207364894094043218421459706511318570933778708 i) j +
        ![generatedRoot.w_151_0 ctx i, generatedRoot.w_152_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν1 i * unitPair sp j) • g +
        reduceMatrix modulus_0 8 1 25 (eB1 i j))
    (heB1 : ∀ i j, CoeffBound (eB1 i j) (BB 1))
    (sel1 : Fin 8 → Fin 2) (hsel1 : ∀ o j, ν1 (cellAt o j) = if j = sel1 o then (1 : ErrorPoly 8) else 0)
    (selected1 : Fin 8 → Fin 2)
    (hselected1 : ∀ o, ((selected1 o).val : Int) =
      generatedRoot.w_139_0 ctx (cellAt o (sel1 o)))
    (μ1 : Fin 8 → ErrorPoly 8)
    (hproduct1 : ∀ o, μ1 o = α (selected1 o) * (μ0 o))
    (o : Fin 8) :
    EncRel sm (generatedRoot.w_154_0 ctx o, (generatedRoot.w_154_1 ctx o,
        (generatedRoot.w_154_2 ctx o, (generatedRoot.w_154_3 ctx o, (generatedRoot.w_154_4 ctx o,
          (generatedRoot.w_154_5 ctx o, ()))))))
      (fun port ↦ reducePoly modulus_0 8 (μ1 o * unitPair sp port) • g) (online1PrfBudget1 Ba Bs BB) := by
  have bucket0 : ∀ o : Fin 8,
      EncRel sm (generatedRoot.w_150_0 ctx o, (generatedRoot.w_150_1 ctx o,
        (generatedRoot.w_150_2 ctx o, (generatedRoot.w_150_3 ctx o, (generatedRoot.w_150_4 ctx o,
          (generatedRoot.w_150_5 ctx o, ()))))))
        (fun port ↦ reducePoly modulus_0 8 (μ0 o * unitPair sp port) • g) (online1PrfBudget0 Ba Bs BB) := by
    intro o
    obtain ⟨key, hkey, hrel⟩ := online1_bucket0 ctx F hq1 layout hlayout hexact
      hα hαμ hG ha hpay hg' hg
      hs
      (hrhs0) heB0 sel0 hsel0 o
    have same : key = selected0 o := Fin.ext (by
      exact_mod_cast hkey.trans (hselected0 o).symm)
    subst key
    simpa only [Nat.cast_one, Int.cast_one, one_mul, ← hproduct0 o] using hrel
  have bucket1 : ∀ o : Fin 8,
      EncRel sm (generatedRoot.w_154_0 ctx o, (generatedRoot.w_154_1 ctx o,
        (generatedRoot.w_154_2 ctx o, (generatedRoot.w_154_3 ctx o, (generatedRoot.w_154_4 ctx o,
          (generatedRoot.w_154_5 ctx o, ()))))))
        (fun port ↦ reducePoly modulus_0 8 (μ1 o * unitPair sp port) • g) (online1PrfBudget1 Ba Bs BB) := by
    intro o
    obtain ⟨key, hkey, hrel⟩ := online1_bucket1 ctx F hq1 layout hlayout hexact
      hα hαμ1 hG ha hpay hg' hg
      (fun member ↦ by simpa only [Nat.cast_one, Int.cast_one, one_mul] using bucket0 member)
      (hrhs1) heB1 sel1 hsel1 o
    have same : key = selected1 o := Fin.ext (by
      exact_mod_cast hkey.trans (hselected1 o).symm)
    subst key
    simpa only [Nat.cast_one, Int.cast_one, one_mul, ← hproduct1 o] using hrel
  exact bucket1 o


abbrev online1PrfHelperPre : Blob := ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97])
abbrev online1Sigmas : List Int := [1, 3, 5, 7, 9, 11, 13, 15]

theorem online1Sigmas_odd (i : Fin 8) : Odd (sigmaOf online1Sigmas i).toNat := by
  have hsize : online1Sigmas.length = 8 := rfl
  have hindex : i.val < online1Sigmas.length := by simpa only [hsize] using i.isLt
  have hall : online1Sigmas.all (fun value ↦ value.toNat % 2 == 1) = true := by decide
  have hmod := List.all_eq_true.mp hall _ (List.getElem_mem hindex)
  change Odd ((online1Sigmas[i.val]?.getD 0).toNat)
  rw [List.getElem?_eq_getElem hindex, Option.getD_some]
  exact Nat.odd_iff.mpr (by simpa only [beq_iff_eq] using hmod)

set_option maxHeartbeats 32000000 in
theorem online1_readout_graph {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp : ErrorPoly 8} (hsm : polyNorm sm ≤ 1)
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {μ : Fin 8 → ErrorPoly 8} {B : Nat}
    (hin : ∀ o : Fin 8, EncRel sm (generatedRoot.w_154_0 ctx o, (generatedRoot.w_154_1 ctx o,
        (generatedRoot.w_154_2 ctx o, (generatedRoot.w_154_3 ctx o, (generatedRoot.w_154_4 ctx o,
          (generatedRoot.w_154_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ o * unitPair sp j) • g) B)
    {Bh : Nat}
    (hhelp : ∀ i : Fin 8, Rel sm (keyPairG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 online1PrfHelperPre
        [.integer (sigmaOf online1Sigmas i)])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution (sigmaOf online1Sigmas i).toNat (online1Sigmas_odd i) sm) j) • g)
      ![generatedRoot.w_137_0 ctx i, generatedRoot.w_138_0 ctx i] Bh)
    {Δ' : ErrorPoly 8} {ν : Fin 8 → ErrorPoly 8} {gq : ExactMatrix modulus_0 8 1 25}
    {C : Fin 8 → ExactPoly modulus_0 8} {BB : Nat}
    (hswitch : ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly modulus_0 8 (substitution (sigmaOf online1Sigmas i).toNat (online1Sigmas_odd i) sp) •
          ![![generatedRoot.w_133_0 ctx, generatedRoot.w_134_0 ctx],
            ![generatedRoot.w_135_0 ctx, generatedRoot.w_136_0 ctx]] 0 j i +
        ![![generatedRoot.w_133_0 ctx, generatedRoot.w_134_0 ctx],
          ![generatedRoot.w_135_0 ctx, generatedRoot.w_136_0 ctx]] 1 j i =
      (C i * reducePoly modulus_0 8 (Δ' * unitPair sp j)) • gq + reduceMatrix modulus_0 8 1 25 (e j))
    (hμ : ∀ o i, polyNorm (substitution (sigmaOf online1Sigmas i).toNat (online1Sigmas_odd i) (μ o)) ≤ 1)
    (hwalsh : ∀ o, ∑ i, reducePoly modulus_0 8 (substitution (sigmaOf online1Sigmas i).toNat
      (online1Sigmas_odd i) (μ o)) * C i = reducePoly modulus_0 8 (ν o))
    (o : Fin 8) :
    EncRel sm (generatedRoot.w_155_0 ctx o, (generatedRoot.w_155_1 ctx o,
      (generatedRoot.w_155_2 ctx o, (generatedRoot.w_155_3 ctx o, (generatedRoot.w_155_4 ctx o,
        (generatedRoot.w_155_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (Δ' * ν o * unitPair sp j) • gq)
      (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
        ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 8 * (8 * 1 * BB)) := by
    have he := hin o
    unfold EncRel at he
    have hr := LutMemberFlat.rel (by norm_num) (log := 3) (by norm_num) (L := 7) (by norm_num) hq1
      layout hlayout hexact (fun i a b hb ↦ sequential_parallel_generatedRoot_155_18.sem hb)
      (parallel_generatedRoot_155.sem (F.c6.h155_5 o)) online1Sigmas_odd hsm hg' hg he
      hhelp hswitch (hμ o) (hwalsh o)
    exact encRel_of_rel hr (fun a j ↦ by fin_cases a <;> fin_cases j <;> rfl)
      (fun j ↦ by fin_cases j <;> rfl)

theorem online1_monomial_place_rel {n : Nat} (hn : 1 < n) {q m : Nat} {sm : ErrorPoly n} {o : Nat}
    (ho : o < 2 * n) {x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m}
    {out : Tup (ExactMatrix q n 1 m) 6}
    (h : out = (multiplyMonomial x0 (Int.ofNat o), (multiplyMonomial x1 (Int.ofNat o),
      (multiplyMonomial x2 (Int.ofNat o), (multiplyMonomial x3 (Int.ofNat o),
        (multiplyMonomial x4 (Int.ofNat o), (multiplyMonomial x5 (Int.ofNat o), ())))))))
    {t : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (hin : EncRel sm (x0, (x1, (x2, (x3, (x4, (x5, ())))))) t B) :
    EncRel sm out (fun j ↦ reducePoly q n (Xe (n := n) o) • t j) B := by
  subst h
  have hr := rel_monomial_gen hn hin ho
  have hA : Enc.A (multiplyMonomial x0 (Int.ofNat o), (multiplyMonomial x1 (Int.ofNat o),
      (multiplyMonomial x2 (Int.ofNat o), (multiplyMonomial x3 (Int.ofNat o),
        (multiplyMonomial x4 (Int.ofNat o), (multiplyMonomial x5 (Int.ofNat o), ())))))) =
      fun a j ↦ multiplyMonomial (Enc.A (x0, (x1, (x2, (x3, (x4, (x5, ())))))) a j) (o : Int) := by
    funext a j; fin_cases a <;> fin_cases j <;> rfl
  have hc : Enc.c (multiplyMonomial x0 (Int.ofNat o), (multiplyMonomial x1 (Int.ofNat o),
      (multiplyMonomial x2 (Int.ofNat o), (multiplyMonomial x3 (Int.ofNat o),
        (multiplyMonomial x4 (Int.ofNat o), (multiplyMonomial x5 (Int.ofNat o), ())))))) =
      fun j ↦ multiplyMonomial (Enc.c (x0, (x1, (x2, (x3, (x4, (x5, ())))))) j) (o : Int) := by
    funext j; fin_cases j <;> rfl
  unfold EncRel
  rw [hA, hc]
  exact hr

/-- Extensionality for the fixed-size tuple representation. -/
theorem online1_tuple_ext {α : Type} {size : Nat} {left right : Tup α size}
    (heq : ∀ component, left.get component = right.get component) : left = right := by
  induction size with
  | zero => exact (show Subsingleton Unit from inferInstance).elim _ _
  | succ size ih =>
    apply Prod.ext
    · exact heq 0
    · exact ih (fun component ↦ heq component.succ)

/-- A fused gather/placement selects the same member in every component and
preserves its noise bound under a monomial. The family size is arbitrary. -/
theorem online1_gather_monomial_place_rel (hn : 1 < n) {q m members : Nat}
    {sm : ErrorPoly n} {families : Tup (Fin members → ExactMatrix q n 1 m) 6}
    {gathered output : Tup (ExactMatrix q n 1 m) 6} {index : Int} {exponent : Nat}
    (hexponent : exponent < 2 * n)
    (hgather : GatherUnpack.Spec families index gathered)
    (hplace : ∀ component, output.get component =
      multiplyMonomial (gathered.get component) (exponent : Int))
    {target : Fin members → Fin 2 → ExactMatrix q n 1 m} {budget : Nat}
    (hmember : ∀ member, EncRel sm (Tup.ev families member) (target member) budget) :
    ∃ member : Fin members, member.val = index.toNat ∧ EncRel sm output
      (fun port ↦ reducePoly q n (Xe (n := n) exponent) • target member port) budget := by
  obtain ⟨hbound, hnonneg, hgather⟩ := hgather
  let member : Fin members := ⟨index.toNat, hbound⟩
  have hsource : gathered = Tup.ev families member := by
    apply online1_tuple_ext
    intro component
    simpa only [Tup.get_ev] using hgather component
  have hout : output =
      (multiplyMonomial gathered.1 (exponent : Int),
      (multiplyMonomial gathered.2.1 (exponent : Int),
      (multiplyMonomial gathered.2.2.1 (exponent : Int),
      (multiplyMonomial gathered.2.2.2.1 (exponent : Int),
      (multiplyMonomial gathered.2.2.2.2.1 (exponent : Int),
      (multiplyMonomial gathered.2.2.2.2.2.1 (exponent : Int), ())))))) := by
    apply online1_tuple_ext
    intro component
    fin_cases component <;> exact hplace _
  refine ⟨member, rfl, online1_monomial_place_rel hn hexponent hout ?_⟩
  simpa only [hsource] using hmember member


end Dio2Proof
namespace W_Stage_online_1
open Dio2Proof Mxx.Primitives MxxRuntime MxxExponentLut
  RefreshTopProof.Gen RefreshTopProof.GenInst RefreshTopProof.GenIface
/-- The exported fused gather/monomial placement, without family enumeration. -/
theorem parallel_generatedRoot_156.placement_rel {backend : MxxRuntime.BackendContext} {x_0 : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {inputs : (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × Unit} {outputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × Unit}
    (h : parallel_generatedRoot_156 backend x_0 params i_0 inputs outputs) {sm : ErrorPoly 8}
    {target : Fin 8 → Fin 2 → ExactMatrix modulus_0 8 1 25} {budget : Nat}
    (hmember : ∀ member, EncRel sm (Tup.ev inputs member) (target member) budget) :
    ∃ member : Fin 8, member.val = ((((Int.fdiv ((Int.ofNat i_0)) (8)) + (Int.fmod ((Int.ofNat i_0)) (8))))).toNat ∧ EncRel sm outputs
      (fun port ↦ reducePoly modulus_0 8 (Xe (n := 8) (i_0 % 8)) • target member port) budget := by
  obtain ⟨w_2_0, w_6_0, w_10_0, w_14_0, w_18_0, w_22_0, a0_0, a0_1, a0_2, a0_3, a0_4, a0_5, a0_6, a0_7, a0_8, a0_9, a0_10, a0_11, a0_12, a0_13, a0_14, a0_15, a0_16, a0_17, a0_18, a0_19, a0_20, a0_21, a0_22, a0_23, a0_24, a0_25, a0_26⟩ := h
  apply online1_gather_monomial_place_rel (by norm_num : 1 < 8)
    (families := inputs) (gathered := (w_2_0, (w_6_0, (w_10_0, (w_14_0, (w_18_0, (w_22_0, ())))))))
    (index := (((Int.fdiv ((Int.ofNat i_0)) (8)) + (Int.fmod ((Int.ofNat i_0)) (8))))) (exponent := i_0 % 8) (by omega)
  · apply GatherUnpack.sem (by omega) (by omega)
    intro component
    fin_cases component
    exacts [a0_4, a0_8, a0_12, a0_16, a0_20, a0_24]
  · intro component
    rw [a0_26]
    have hexponent : Int.fmod (Int.ofNat i_0) (8 : Int) = ((i_0 % 8 : Nat) : Int) :=
      fmod_ofNat i_0 8
    fin_cases component <;> rw [hexponent] <;> rfl
  · exact hmember


end W_Stage_online_1
namespace Dio2Proof
open W_Stage_online_1 Mxx.Primitives MxxRuntime MxxExponentLut
  RefreshTopProof.Gen RefreshTopProof.GenInst RefreshTopProof.GenIface
open scoped BigOperators
/-- The actual exported member/column loop, conditional on its member relations. -/
theorem online1_smudge_pack {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    {sm : ErrorPoly 8} {target : Fin 8 → Fin 2 → ExactMatrix modulus_0 8 1 25}
    {budget : Nat}
    (hmember : ∀ member, EncRel sm (Tup.ev ((generatedRoot.w_156_0 ctx), ((generatedRoot.w_156_1 ctx), ((generatedRoot.w_156_2 ctx), ((generatedRoot.w_156_3 ctx), ((generatedRoot.w_156_4 ctx), ((generatedRoot.w_156_5 ctx), ())))))) member) (target member) budget)
    :
    EncRel sm ((generatedRoot.w_175_0 ctx), ((generatedRoot.w_175_1 ctx), ((generatedRoot.w_175_2 ctx), ((generatedRoot.w_175_3 ctx), ((generatedRoot.w_175_4 ctx), ((generatedRoot.w_175_5 ctx), ())))))) (∑ member, target member) (8 * budget) := by
  exact member_total_rel (by norm_num : 2 ≤ 8)
    (fams := ((generatedRoot.w_156_0 ctx), ((generatedRoot.w_156_1 ctx), ((generatedRoot.w_156_2 ctx), ((generatedRoot.w_156_3 ctx), ((generatedRoot.w_156_4 ctx), ((generatedRoot.w_156_5 ctx), ())))))))
    (a := ((generatedRoot.w_157_0 ctx), ((generatedRoot.w_160_0 ctx), ((generatedRoot.w_163_0 ctx), ((generatedRoot.w_166_0 ctx), ((generatedRoot.w_169_0 ctx), ((generatedRoot.w_172_0 ctx), ()))))))) (b := ((generatedRoot.w_158_0 ctx), ((generatedRoot.w_161_0 ctx), ((generatedRoot.w_164_0 ctx), ((generatedRoot.w_167_0 ctx), ((generatedRoot.w_170_0 ctx), ((generatedRoot.w_173_0 ctx), ())))))))
    (fun component ↦ by fin_cases component; exacts [F.c6.h157_0, F.c6.h160_0, F.c6.h163_0, F.c6.h166_0, F.c7.h169_0, F.c7.h172_0])
    (fun component ↦ by fin_cases component; exacts [F.c6.h158_0, F.c6.h161_0, F.c6.h164_0, F.c7.h167_0, F.c7.h170_0, F.c7.h173_0])
    (fun component ↦ by fin_cases component <;> rfl)
    F.c7.h175_5
    (fun index current next hbody ↦ sequential_generatedRoot_175.sem hbody)
    hmember

abbrev online1ReadoutBudget (Ba Bs Bh Bswitch : Nat) (BB : Fin 2 → Nat) : Nat :=
  (7 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * (online1PrfBudget1 Ba Bs BB) * 2048) +
        ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * (online1PrfBudget1 Ba Bs BB) * 2048) + 8 * (8 * 1 * Bswitch))

set_option maxHeartbeats 400000000 in
/-- The smudge readout follows from the complete exported PRF chain and preprocessing facts. -/
theorem online1_readout_root {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} {α : Fin 2 → ErrorPoly 8} (hα : ∀ k, SmulGain (α k) 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1)
    {G : ExactMatrix modulus_0 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G) {Ba : Nat}
    (ha : ∀ k : Fin 2, Rel sm (labelKeysG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 online1PrfLabelPre k)
      (fun j ↦ reducePoly modulus_0 8 (α k * unitPair sm j) • G)
      ![generatedRoot.w_16_0 ctx k, generatedRoot.w_27_0 ctx k] Ba)
    (hpay : ∀ k, generatedRoot.w_47_0 ctx k = fun _ _ ↦ reducePoly modulus_0 8 (α k))
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm ![![generatedRoot.w_141_0 ctx, generatedRoot.w_144_0 ctx],
        ![generatedRoot.w_142_0 ctx, generatedRoot.w_145_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (μ * unitPair sp j) • g)
      ![generatedRoot.w_143_0 ctx, generatedRoot.w_146_0 ctx] Bs)
    (BB : Fin 2 → Nat)
    {ν0 : Fin 16 → ErrorPoly 8}
    {eB0 : Fin 16 → Fin 2 → ErrorMatrix 8 1 25}
    (hrhs0 : ∀ (i : Fin 16) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_70_0 ctx) modulus_0 25 online1_bucket0_rhsPre (prfTopTags 36550556566445887949135852474979514307313305075533710697673106717124183540354 84715014674746765542951856319120207364894094043218421459706511318570933778708 i) j +
        ![generatedRoot.w_147_0 ctx i, generatedRoot.w_148_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν0 i * unitPair sp j) • g +
        reduceMatrix modulus_0 8 1 25 (eB0 i j))
    (heB0 : ∀ i j, CoeffBound (eB0 i j) (BB 0))
    (sel0 : Fin 8 → Fin 2) (hsel0 : ∀ o j, ν0 (cellAt o j) = if j = sel0 o then (1 : ErrorPoly 8) else 0)
    (selected0 : Fin 8 → Fin 2)
    (hselected0 : ∀ o, ((selected0 o).val : Int) =
      generatedRoot.w_140_0 ctx (cellAt o (sel0 o)))
    (μ0 : Fin 8 → ErrorPoly 8)
    (hproduct0 : ∀ o, μ0 o = α (selected0 o) * (μ))
    (hαμ1 : ∀ k o, polyNorm (α k * μ0 o) ≤ 1)
    {ν1 : Fin 16 → ErrorPoly 8}
    {eB1 : Fin 16 → Fin 2 → ErrorMatrix 8 1 25}
    (hrhs1 : ∀ (i : Fin 16) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_70_0 ctx) modulus_0 25 online1_bucket1_rhsPre (prfTopTags 36550556566445887949135852474979514307313305075533710697673106717124183540354 84715014674746765542951856319120207364894094043218421459706511318570933778708 i) j +
        ![generatedRoot.w_151_0 ctx i, generatedRoot.w_152_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν1 i * unitPair sp j) • g +
        reduceMatrix modulus_0 8 1 25 (eB1 i j))
    (heB1 : ∀ i j, CoeffBound (eB1 i j) (BB 1))
    (sel1 : Fin 8 → Fin 2) (hsel1 : ∀ o j, ν1 (cellAt o j) = if j = sel1 o then (1 : ErrorPoly 8) else 0)
    (selected1 : Fin 8 → Fin 2)
    (hselected1 : ∀ o, ((selected1 o).val : Int) =
      generatedRoot.w_139_0 ctx (cellAt o (sel1 o)))
    (μ1 : Fin 8 → ErrorPoly 8)
    (hproduct1 : ∀ o, μ1 o = α (selected1 o) * (μ0 o))
    (hsm : polyNorm sm ≤ 1)
    {Bh : Nat}
    (hhelp : ∀ i : Fin 8, Rel sm (keyPairG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 online1PrfHelperPre
        [.integer (sigmaOf online1Sigmas i)])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution (sigmaOf online1Sigmas i).toNat (online1Sigmas_odd i) sm) j) • g)
      ![generatedRoot.w_137_0 ctx i, generatedRoot.w_138_0 ctx i] Bh)
    {Δ' : ErrorPoly 8} {ρ : Fin 8 → ErrorPoly 8} {gq : ExactMatrix modulus_0 8 1 25}
    {C : Fin 8 → ExactPoly modulus_0 8} {Bswitch : Nat}
    (hswitch : ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) Bswitch) ∧ ∀ j,
      reducePoly modulus_0 8 (substitution (sigmaOf online1Sigmas i).toNat (online1Sigmas_odd i) sp) •
          ![![generatedRoot.w_133_0 ctx, generatedRoot.w_134_0 ctx],
            ![generatedRoot.w_135_0 ctx, generatedRoot.w_136_0 ctx]] 0 j i +
        ![![generatedRoot.w_133_0 ctx, generatedRoot.w_134_0 ctx],
          ![generatedRoot.w_135_0 ctx, generatedRoot.w_136_0 ctx]] 1 j i =
      (C i * reducePoly modulus_0 8 (Δ' * unitPair sp j)) • gq + reduceMatrix modulus_0 8 1 25 (e j))
    (hμ : ∀ o i, polyNorm (substitution (sigmaOf online1Sigmas i).toNat (online1Sigmas_odd i) (μ1 o)) ≤ 1)
    (hwalsh : ∀ o, ∑ i, reducePoly modulus_0 8 (substitution (sigmaOf online1Sigmas i).toNat
      (online1Sigmas_odd i) (μ1 o)) * C i = reducePoly modulus_0 8 (ρ o))
    (o : Fin 8) :
    EncRel sm (generatedRoot.w_155_0 ctx o, (generatedRoot.w_155_1 ctx o,
      (generatedRoot.w_155_2 ctx o, (generatedRoot.w_155_3 ctx o, (generatedRoot.w_155_4 ctx o,
        (generatedRoot.w_155_5 ctx o, ()))))))
      (fun j ↦ reducePoly modulus_0 8 (Δ' * ρ o * unitPair sp j) • gq)
      (online1ReadoutBudget Ba Bs Bh Bswitch BB) := by
  exact online1_readout_graph ctx F hq1 layout hlayout hexact hsm hg' hg
    (fun member ↦ by simpa only [Nat.cast_one, Int.cast_one, one_mul] using online1_prf_root ctx F hq1 layout hlayout hexact hα hαμ hG ha hpay hg' hg hs BB hrhs0 heB0 sel0 hsel0 selected0 hselected0 μ0 hproduct0 hαμ1 hrhs1 heB1 sel1 hsel1 selected1 hselected1 μ1 hproduct1 member)
    hhelp hswitch hμ hwalsh o

/-- The corrected export selects quotient plus remainder, not a flattened coefficient index. -/
def online1PlacementSource (member : Fin 8) : Fin 8 :=
  ⟨member.val / 8 + member.val % 8, by have := member.isLt; omega⟩

/-- Each digit's plaintext is the actual corrected placement and column sum. -/
noncomputable def online1PackedReadout (Δ : ErrorPoly 8) (ρ : Fin 8 → ErrorPoly 8) :
    Fin 1 → Fin 1 → ErrorPoly 8 := fun digit _ ↦
  ∑ coefficient : Fin 8, Xe (n := 8) coefficient.val *
    (Δ * ρ (online1PlacementSource ⟨8 * digit.val + coefficient.val,
      by have := digit.isLt; have := coefficient.isLt; omega⟩))

set_option maxHeartbeats 400000000 in
/-- Actual placement and packed totals preserve the readout relation without member enumeration. -/
theorem online1_pack_readout {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) {sm sp Δ : ErrorPoly 8}
    (ρ : Fin 8 → ErrorPoly 8) (gq : ExactMatrix modulus_0 8 1 25) {B : Nat}
    (hread : ∀ member, EncRel sm
      ((generatedRoot.w_155_0 ctx member), ((generatedRoot.w_155_1 ctx member), ((generatedRoot.w_155_2 ctx member), ((generatedRoot.w_155_3 ctx member), ((generatedRoot.w_155_4 ctx member), ((generatedRoot.w_155_5 ctx member), ()))))))
      (fun port ↦ reducePoly modulus_0 8 (Δ * ρ member * unitPair sp port) • gq) B)
    (digit : Fin 1) :
    EncRel sm (blkT (C := 1) (m := 25) ((generatedRoot.w_175_0 ctx), ((generatedRoot.w_175_1 ctx), ((generatedRoot.w_175_2 ctx), ((generatedRoot.w_175_3 ctx), ((generatedRoot.w_175_4 ctx), ((generatedRoot.w_175_5 ctx), ())))))) digit)
      (fun port ↦ reducePoly modulus_0 8
        (online1PackedReadout Δ ρ digit 0 * unitPair sp port) • gq) (8 * B) := by
  let target : Fin 8 → Fin 2 → ExactMatrix modulus_0 8 1 25 := fun member port ↦
    reducePoly modulus_0 8 (Xe (n := 8) (member.val % 8) *
      (Δ * ρ (online1PlacementSource member) * unitPair sp port)) • gq
  have hplace : ∀ member : Fin 8, EncRel sm
      ((generatedRoot.w_156_0 ctx member), ((generatedRoot.w_156_1 ctx member), ((generatedRoot.w_156_2 ctx member), ((generatedRoot.w_156_3 ctx member), ((generatedRoot.w_156_4 ctx member), ((generatedRoot.w_156_5 ctx member), ()))))))
      (target member) B := by
    intro member
    obtain ⟨source, hsource, hrel⟩ := parallel_generatedRoot_156.placement_rel
      (F.c6.h156_5 member) hread
    have same : source = online1PlacementSource member := by
      apply Fin.ext
      rw [hsource]
      have hdiv := (Int.ofNat_fdiv member.val 8).symm
      have hmod := fmod_ofNat member.val 8
      norm_num only [Nat.cast_ofNat] at hdiv hmod
      simp only [Int.ofNat_eq_natCast] at hmod ⊢
      rw [hdiv, hmod]
      simp only [online1PlacementSource, ← Int.natCast_add, Int.toNat_natCast]
    subst source
    simpa only [target, smul_smul, ← map_mul] using hrel
  have htotal := online1_smudge_pack ctx F hplace
  have hdigit : digit = 0 := Subsingleton.elim _ _
  subst digit
  have heq : blkT (C := 1) (m := 25) ((generatedRoot.w_175_0 ctx), ((generatedRoot.w_175_1 ctx), ((generatedRoot.w_175_2 ctx), ((generatedRoot.w_175_3 ctx), ((generatedRoot.w_175_4 ctx), ((generatedRoot.w_175_5 ctx), ())))))) 0 = ((generatedRoot.w_175_0 ctx), ((generatedRoot.w_175_1 ctx), ((generatedRoot.w_175_2 ctx), ((generatedRoot.w_175_3 ctx), ((generatedRoot.w_175_4 ctx), ((generatedRoot.w_175_5 ctx), ())))))) := by
    apply online1_tuple_ext
    intro component
    rw [blkT_get]
    funext row column
    simp [blk]
  rw [heq]
  apply htotal.target_eq
  funext port
  simp only [target, online1PackedReadout, online1PlacementSource, Finset.sum_apply,
    Finset.sum_smul, map_sum, Finset.sum_mul, map_mul, Fin.val_zero, Nat.zero_mul, Nat.zero_add]
  apply Finset.sum_congr rfl
  intro coefficient _
  simp only [Nat.add_mod, Nat.mul_mod_right, Nat.zero_add, Nat.mod_eq_of_lt coefficient.isLt]
  apply congrArg (fun scalar : ExactPoly modulus_0 8 ↦ scalar • gq)
  ring

set_option maxHeartbeats 400000000 in
/-- The full root Facts discharge the packed PRF invariant used by the smudge digit loop. -/
theorem online1_packed_prf_root {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} {α : Fin 2 → ErrorPoly 8} (hα : ∀ k, SmulGain (α k) 1)
    (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1)
    {G : ExactMatrix modulus_0 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G) {Ba : Nat}
    (ha : ∀ k : Fin 2, Rel sm (labelKeysG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 online1PrfLabelPre k)
      (fun j ↦ reducePoly modulus_0 8 (α k * unitPair sm j) • G)
      ![generatedRoot.w_16_0 ctx k, generatedRoot.w_27_0 ctx k] Ba)
    (hpay : ∀ k, generatedRoot.w_47_0 ctx k = fun _ _ ↦ reducePoly modulus_0 8 (α k))
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm ![![generatedRoot.w_141_0 ctx, generatedRoot.w_144_0 ctx],
        ![generatedRoot.w_142_0 ctx, generatedRoot.w_145_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (μ * unitPair sp j) • g)
      ![generatedRoot.w_143_0 ctx, generatedRoot.w_146_0 ctx] Bs)
    (BB : Fin 2 → Nat)
    {ν0 : Fin 16 → ErrorPoly 8}
    {eB0 : Fin 16 → Fin 2 → ErrorMatrix 8 1 25}
    (hrhs0 : ∀ (i : Fin 16) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_70_0 ctx) modulus_0 25 online1_bucket0_rhsPre (prfTopTags 36550556566445887949135852474979514307313305075533710697673106717124183540354 84715014674746765542951856319120207364894094043218421459706511318570933778708 i) j +
        ![generatedRoot.w_147_0 ctx i, generatedRoot.w_148_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν0 i * unitPair sp j) • g +
        reduceMatrix modulus_0 8 1 25 (eB0 i j))
    (heB0 : ∀ i j, CoeffBound (eB0 i j) (BB 0))
    (sel0 : Fin 8 → Fin 2) (hsel0 : ∀ o j, ν0 (cellAt o j) = if j = sel0 o then (1 : ErrorPoly 8) else 0)
    (selected0 : Fin 8 → Fin 2)
    (hselected0 : ∀ o, ((selected0 o).val : Int) =
      generatedRoot.w_140_0 ctx (cellAt o (sel0 o)))
    (μ0 : Fin 8 → ErrorPoly 8)
    (hproduct0 : ∀ o, μ0 o = α (selected0 o) * (μ))
    (hαμ1 : ∀ k o, polyNorm (α k * μ0 o) ≤ 1)
    {ν1 : Fin 16 → ErrorPoly 8}
    {eB1 : Fin 16 → Fin 2 → ErrorMatrix 8 1 25}
    (hrhs1 : ∀ (i : Fin 16) j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_70_0 ctx) modulus_0 25 online1_bucket1_rhsPre (prfTopTags 36550556566445887949135852474979514307313305075533710697673106717124183540354 84715014674746765542951856319120207364894094043218421459706511318570933778708 i) j +
        ![generatedRoot.w_151_0 ctx i, generatedRoot.w_152_0 ctx i] j =
      reducePoly modulus_0 8 (((1 : Nat) : Int) * ν1 i * unitPair sp j) • g +
        reduceMatrix modulus_0 8 1 25 (eB1 i j))
    (heB1 : ∀ i j, CoeffBound (eB1 i j) (BB 1))
    (sel1 : Fin 8 → Fin 2) (hsel1 : ∀ o j, ν1 (cellAt o j) = if j = sel1 o then (1 : ErrorPoly 8) else 0)
    (selected1 : Fin 8 → Fin 2)
    (hselected1 : ∀ o, ((selected1 o).val : Int) =
      generatedRoot.w_139_0 ctx (cellAt o (sel1 o)))
    (μ1 : Fin 8 → ErrorPoly 8)
    (hproduct1 : ∀ o, μ1 o = α (selected1 o) * (μ0 o))
    (hsm : polyNorm sm ≤ 1)
    {Bh : Nat}
    (hhelp : ∀ i : Fin 8, Rel sm (keyPairG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 online1PrfHelperPre
        [.integer (sigmaOf online1Sigmas i)])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution (sigmaOf online1Sigmas i).toNat (online1Sigmas_odd i) sm) j) • g)
      ![generatedRoot.w_137_0 ctx i, generatedRoot.w_138_0 ctx i] Bh)
    {Δ' : ErrorPoly 8} {ρ : Fin 8 → ErrorPoly 8} {gq : ExactMatrix modulus_0 8 1 25}
    {C : Fin 8 → ExactPoly modulus_0 8} {Bswitch : Nat}
    (hswitch : ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) Bswitch) ∧ ∀ j,
      reducePoly modulus_0 8 (substitution (sigmaOf online1Sigmas i).toNat (online1Sigmas_odd i) sp) •
          ![![generatedRoot.w_133_0 ctx, generatedRoot.w_134_0 ctx],
            ![generatedRoot.w_135_0 ctx, generatedRoot.w_136_0 ctx]] 0 j i +
        ![![generatedRoot.w_133_0 ctx, generatedRoot.w_134_0 ctx],
          ![generatedRoot.w_135_0 ctx, generatedRoot.w_136_0 ctx]] 1 j i =
      (C i * reducePoly modulus_0 8 (Δ' * unitPair sp j)) • gq + reduceMatrix modulus_0 8 1 25 (e j))
    (hμ : ∀ o i, polyNorm (substitution (sigmaOf online1Sigmas i).toNat (online1Sigmas_odd i) (μ1 o)) ≤ 1)
    (hwalsh : ∀ o, ∑ i, reducePoly modulus_0 8 (substitution (sigmaOf online1Sigmas i).toNat
      (online1Sigmas_odd i) (μ1 o)) * C i = reducePoly modulus_0 8 (ρ o))
    (digit : Fin 1) :
    EncRel sm (blkT (C := 1) (m := 25) ((generatedRoot.w_175_0 ctx), ((generatedRoot.w_175_1 ctx), ((generatedRoot.w_175_2 ctx), ((generatedRoot.w_175_3 ctx), ((generatedRoot.w_175_4 ctx), ((generatedRoot.w_175_5 ctx), ())))))) digit)
      (fun port ↦ reducePoly modulus_0 8
        (online1PackedReadout Δ' ρ digit 0 * unitPair sp port) • gq)
      (8 * (online1ReadoutBudget Ba Bs Bh Bswitch BB)) := by
  exact online1_pack_readout ctx F ρ gq
    (fun member ↦ online1_readout_root ctx F hq1 layout hlayout hexact hα hαμ hG ha hpay hg' hg hs BB hrhs0 heB0 sel0 hsel0 selected0 hselected0 μ0 hproduct0 hαμ1 hrhs1 heB1 sel1 hsel1 selected1 hselected1 μ1 hproduct1 hsm hhelp hswitch hμ hwalsh member) digit


end Dio2Proof
