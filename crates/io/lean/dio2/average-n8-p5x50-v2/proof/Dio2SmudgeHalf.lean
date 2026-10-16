import Dio2Online1Prf
import Gen_W_Stage_online_1_Scopes
import W_Stage_online_1_Facts
import Dio2SmudgeValues

/-!
# Multi-digit hTH witnesses

The generic residual proof applies to every digit, prime and part symbolically.
Generated adapters prove the actual block-diagonal/gadget digit loop, indexed
weights, weighted target and half offset. `online1_smudge_root` composes the full
root PRF/packing proof with this loop and produces both hs and hTH.
Imported encodings, preprocessing semantics and odd bounded readout digits remain
explicit; internal PRF, placement, packing and target identities do not.
-/
namespace Dio2Proof
open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen
open scoped BigOperators

/-- Weighted packed readouts minus their coefficientwise half have the full smudge bound. -/
theorem smudge_hTH {q n P d columns : Nat} (hn : 0 < n) (hP : P % 2 = 0)
    {half : Int} (hhalf : ((2 * half : Int) : ZMod q) = 1)
    (digit : Fin (d + 1) → Fin columns → Fin n → Int)
    (readout : Fin (d + 1) → Fin columns → ErrorPoly n)
    (hreadout : ∀ r c, reducePoly q n (readout r c) =
      reducePoly q n (intPoly (digit r c)))
    (hodd : ∀ r c i, digit r c i % 2 = 1)
    (hbound : ∀ r c i, |digit r c i| ≤ (P : Int) - 1)
    {T H : ExactMatrix q n 1 columns}
    (hT : T = fun _ c ↦ ∑ r : Fin (d + 1), reducePoly q n (readout r c) *
      ((half : ExactPoly q n) * (P : ExactPoly q n) ^ r.val))
    (hH : H = allCoefficients half) :
    ∃ M : ErrorMatrix n 1 columns, CoeffBound M (P ^ (d + 1) / 2) ∧
      T - H = reduceMatrix q n 1 columns M := by
  refine ⟨smudgeResidual P digit, smudgeResidual_bound hn hP digit hodd hbound, ?_⟩
  rw [hT, hH, allCoefficients_reduce]
  funext row c
  simp only [Matrix.sub_apply, reduceMatrix, smudgeResidual, hreadout]
  exact smudge_weighted_poly hP hhalf (fun r ↦ digit r c) (hodd 0 c)
/- Generated final-output adapter: gen_online1.py. -/
open W_Stage_online_1 in
/-- Multi-digit hTH with the coefficientwise offset actually exported by online step 1.
The readout and weighted-target identities remain the PRF/digit-loop adapter obligations. -/
theorem online1_smudge_hTH {backend : BackendContext} {hashModel : HashModel}
    {params : Params} (ctx : generatedRoot.Ctx backend hashModel params)
    (digit : Fin 1 → Fin 1 → Fin 8 → Int)
    (readout : Fin 1 → Fin 1 → ErrorPoly 8)
    (hreadout : ∀ r c, reducePoly 1809251394327365744250316798798350396453362684746139641520435240552807944993 8 (readout r c) =
      reducePoly 1809251394327365744250316798798350396453362684746139641520435240552807944993 8 (intPoly (digit r c)))
    (hodd : ∀ r c i, digit r c i % 2 = 1)
    (hbound : ∀ r c i, |digit r c i| ≤ (8 : Int) - 1)
    {T : ExactMatrix 1809251394327365744250316798798350396453362684746139641520435240552807944993 8 1 1}
    (hT : T = fun _ c ↦ ∑ r : Fin 1, reducePoly 1809251394327365744250316798798350396453362684746139641520435240552807944993 8 (readout r c) *
      ((904625697163682872125158399399175198226681342373069820760217620276403972497 : ExactPoly 1809251394327365744250316798798350396453362684746139641520435240552807944993 8) * (8 : ExactPoly 1809251394327365744250316798798350396453362684746139641520435240552807944993 8) ^ r.val)) :
    ∃ M : ErrorMatrix 8 1 1, CoeffBound M (8 ^ 1 / 2) ∧
      T - generatedRoot.w_181_0 ctx = reduceMatrix 1809251394327365744250316798798350396453362684746139641520435240552807944993 8 1 1 M := by
  exact smudge_hTH (d := 0) (by norm_num) (by norm_num)
    (half := 904625697163682872125158399399175198226681342373069820760217620276403972497) (by
      apply (ZMod.intCast_eq_intCast_iff (2 * 904625697163682872125158399399175198226681342373069820760217620276403972497) 1 1809251394327365744250316798798350396453362684746139641520435240552807944993).2
      norm_num [Int.ModEq]) digit readout hreadout hodd hbound hT rfl

open W_Stage_online_1 RefreshTopProof.GenInst RefreshTopProof.GenIface
/-- One exported digit step uses the digit-indexed weight and the matching packed blocks. -/
theorem online1_digit_step_graph {backend : MxxRuntime.BackendContext} {x_0 : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {inputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Fin 1 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × Unit} {outputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × Unit}
    (hq : 1 < modulus_0) (h : sequential_generatedRoot_178 backend x_0 params i_0 inputs outputs) (hr : i_0 < 1) :
    ∀ component : Fin 3, Tup.get (k := 3) outputs component = ![inputs.1, inputs.2.1, inputs.2.2.1] component +
      ![blk inputs.2.2.2.1 ⟨i_0, hr⟩, blk inputs.2.2.2.2.2.1 ⟨i_0, hr⟩,
        blk inputs.2.2.2.2.2.2.1 ⟨i_0, hr⟩] component *
        decG (inner := 25) backend 4096 25
          (inputs.2.2.2.2.1 ⟨i_0, hr⟩) := by
  obtain ⟨w_2_0, w_5_0, w_6_0, w_7_0, w_10_0, w_11_0, w_16_0, w_21_0, a0_0, a0_1, a0_2, a0_3, a0_4, a0_5, a0_6, a0_7, a0_8, a0_9, a0_10, a0_11, a0_12, a0_13, a0_14, a0_15, a0_16, a0_17, a0_18, a0_19, a0_20, a0_21, a0_22, a0_23, a0_24, a0_25, a0_26, a0_27, a0_28⟩ := h
  obtain ⟨_, hweight⟩ := familyGet_atG i_0 a0_9
  have hblock : w_6_0 = w_5_0 := by
    funext row column
    simpa only [Fin.isValue, Fin.val_zero, Nat.zero_mul, Nat.zero_add, ite_true]
      using a0_10.2 0 0 row column
  have hrebased : w_7_0 = w_5_0 :=
    (centeredRebaseRuns_unique a0_11 (centeredRebaseRuns_self hq (by norm_num) _)).trans hblock
  have hscalar : matrixMulScalarLeft w_7_0
      (matrixPolynomial [1]) = inputs.2.2.2.2.1 ⟨i_0, hr⟩ := by
    rw [hrebased, hweight]
    funext row column
    have hrow : row = 0 := Subsingleton.elim _ _
    have hcolumn : column = 0 := Subsingleton.elim _ _
    subst row column
    simp [matrixMulScalarLeft, matrixPolynomial_single]
  have hdigits : w_11_0 = decG (inner := 25) backend 4096 25
      (inputs.2.2.2.2.1 ⟨i_0, hr⟩) := by
    have hd := (RefreshTopProof.GenFinal.compactRebase_self_gen (by norm_num) hq a0_13).trans
      (decG_eq a0_12)
    simpa only [hscalar] using hd
  have hs0 : w_2_0 = blk inputs.2.2.2.1 ⟨i_0, hr⟩ := slice_blk ⟨i_0, hr⟩ a0_6 rfl
  have hs1 : w_16_0 = blk inputs.2.2.2.2.2.1 ⟨i_0, hr⟩ := slice_blk ⟨i_0, hr⟩ a0_20 rfl
  have hs2 : w_21_0 = blk inputs.2.2.2.2.2.2.1 ⟨i_0, hr⟩ := slice_blk ⟨i_0, hr⟩ a0_27 rfl
  rw [a0_28]
  intro component
  fin_cases component
  · change matrixAdd inputs.1 (matrixMul w_2_0 w_11_0) =
      inputs.1 + blk inputs.2.2.2.1 ⟨i_0, hr⟩ *
        decG (inner := 25) backend 4096 25
          (inputs.2.2.2.2.1 ⟨i_0, hr⟩)
    rw [hs0, hdigits]
    rfl
  · change matrixAdd inputs.2.1 (matrixMul w_16_0 w_11_0) =
      inputs.2.1 + blk inputs.2.2.2.2.2.1 ⟨i_0, hr⟩ *
        decG (inner := 25) backend 4096 25
          (inputs.2.2.2.2.1 ⟨i_0, hr⟩)
    rw [hs1, hdigits]
    rfl
  · change matrixAdd inputs.2.2.1 (matrixMul w_21_0 w_11_0) =
      inputs.2.2.1 + blk inputs.2.2.2.2.2.2.1 ⟨i_0, hr⟩ *
        decG (inner := 25) backend 4096 25
          (inputs.2.2.2.2.1 ⟨i_0, hr⟩)
    rw [hs2, hdigits]
    rfl

/-- The root smudge accumulator is the sum of all indexed digit actions. -/
theorem online1_digit_sums {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) (hq : 1 < modulus_0) :
    ∀ component : Fin 3, Tup.get (k := 3) ((generatedRoot.w_178_0 ctx, (generatedRoot.w_178_1 ctx, (generatedRoot.w_178_2 ctx, ())))) component =
      ∑ r : Fin 1, ![blk (generatedRoot.w_175_5 ctx) r, blk (generatedRoot.w_175_1 ctx) r,
        blk (generatedRoot.w_175_3 ctx) r] component *
      decG (inner := 25) backend 4096 25 (generatedRoot.w_177_0 ctx r) := by
  let term : Nat → Fin 3 → ExactMatrix modulus_0 8 1 1 := fun index component ↦
    if valid : index < 1 then
      let r : Fin 1 := ⟨index, valid⟩
      ![blk (generatedRoot.w_175_5 ctx) r, blk (generatedRoot.w_175_1 ctx) r,
        blk (generatedRoot.w_175_3 ctx) r] component *
      decG (inner := 25) backend 4096 25 (generatedRoot.w_177_0 ctx r)
    else 0
  have result := online1_digit_fold term (fun index current next hi hstep ↦ by
    change index < 1 at hi
    have hs := online1_digit_step_graph hq hstep hi
    intro component
    have hcurrent : Tup.get (k := 3) current component =
        ![current.1, current.2.1, current.2.2.1] component := by
      fin_cases component <;> rfl
    dsimp only [term]
    simp only [dif_pos hi, if_pos hi, hcurrent]
    exact hs component) (F.c7.h178_2)
  intro component
  simpa only [Finset.sum_range, term, dif_pos, Fin.is_lt] using result component

/-- Every exported digit uses its own radix power, not a repeated first-digit weight. -/
theorem online1_digit_weights {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) (r : Fin 1) :
    generatedRoot.w_177_0 ctx r = fun _ _ ↦ (904625697163682872125158399399175198226681342373069820760217620276403972497 : ExactPoly modulus_0 8) *
      (8 : ExactPoly modulus_0 8) ^ r.val := by
  have equation := (F.c7.h177_0 r).2.2
  change generatedRoot.w_177_0 ctx r = matrixMulScalarLeft
    (liftInteger ((8 : Int) ^ r.val)) (matrixPolynomial [(904625697163682872125158399399175198226681342373069820760217620276403972497)]) at equation
  rw [equation]
  funext row column
  simp only [matrixMulScalarLeft, liftInteger, matrixPolynomial_single, Int.cast_pow,
    Int.cast_natCast]
  ring


/-- The actual digit loop discharges the smudge row equation from packed readouts.
The packed PRF relation is an explicit composition boundary, not an imported-input fact. -/
theorem online1_smudge_of_packed {backend : BackendContext} {hashModel : HashModel}
    {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) (hq : 1 < modulus_0)
    (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    {sm : ErrorPoly 8} {g : ExactMatrix modulus_0 8 1 25}
    (hg : gadgetMatrixRuns backend 4096 25 g)
    (target : Fin 1 → Fin 2 → ExactPoly modulus_0 8) {budget : Nat}
    (hpacked : ∀ r, EncRel sm (blkT ((generatedRoot.w_175_0 ctx, (generatedRoot.w_175_1 ctx, (generatedRoot.w_175_2 ctx, (generatedRoot.w_175_3 ctx, (generatedRoot.w_175_4 ctx, (generatedRoot.w_175_5 ctx, ()))))))) r)
      (fun port ↦ target r port • g) budget) :
    ∃ error : ErrorMatrix 8 1 1,
      CoeffBound error (1 * (25 * 8 * budget * 2048)) ∧
      generatedRoot.w_178_0 ctx = Mxx.ExponentLut.Refresh.maskPart (reducePoly modulus_0 8 sm)
        ![generatedRoot.w_178_1 ctx, generatedRoot.w_178_2 ctx] -
        (∑ r, target r 1 • generatedRoot.w_177_0 ctx r) + reduceMatrix modulus_0 8 1 1 error := by
  obtain ⟨error, herror, equation⟩ := online1_digit_encoding (by norm_num) hq layout
    hlayout hexact hg ((generatedRoot.w_175_0 ctx, (generatedRoot.w_175_1 ctx, (generatedRoot.w_175_2 ctx, (generatedRoot.w_175_3 ctx, (generatedRoot.w_175_4 ctx, (generatedRoot.w_175_5 ctx, ()))))))) target hpacked (generatedRoot.w_177_0 ctx)
    (fun r ↦ decG (inner := 25) backend 4096 25
      (generatedRoot.w_177_0 ctx r))
    (fun r ↦ online1_weight_decomposition (by norm_num) hq layout hlayout hexact hg _)
  have row_eq := online1_digit_sums ctx F hq 0
  have key0_eq := online1_digit_sums ctx F hq 1
  have key1_eq := online1_digit_sums ctx F hq 2
  change generatedRoot.w_178_0 ctx = ∑ r, blk (generatedRoot.w_175_5 ctx) r *
    decG (inner := 25) backend 4096 25 (generatedRoot.w_177_0 ctx r) at row_eq
  change generatedRoot.w_178_1 ctx = ∑ r, blk (generatedRoot.w_175_1 ctx) r *
    decG (inner := 25) backend 4096 25 (generatedRoot.w_177_0 ctx r) at key0_eq
  change generatedRoot.w_178_2 ctx = ∑ r, blk (generatedRoot.w_175_3 ctx) r *
    decG (inner := 25) backend 4096 25 (generatedRoot.w_177_0 ctx r) at key1_eq
  rw [← row_eq, ← key0_eq, ← key1_eq] at equation
  exact ⟨error, herror, equation⟩

/-- The weighted target produced by the digit loop has the multi-digit residual bound.
No internal weighted-target identity is assumed. Odd digit witnesses remain a readout premise. -/
theorem online1_digit_target_hTH {backend : BackendContext} {hashModel : HashModel}
    {params : Params} (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (digit : Fin 1 → Fin 1 → Fin 8 → Int)
    (readout : Fin 1 → Fin 1 → ErrorPoly 8)
    (hreadout : ∀ r c, reducePoly modulus_0 8 (readout r c) =
      reducePoly modulus_0 8 (intPoly (digit r c)))
    (hodd : ∀ r c i, digit r c i % 2 = 1)
    (hbound : ∀ r c i, |digit r c i| ≤ (8 : Int) - 1) :
    ∃ error : ErrorMatrix 8 1 1, CoeffBound error (8 ^ 1 / 2) ∧
      (∑ r, reducePoly modulus_0 8 (readout r 0) • generatedRoot.w_177_0 ctx r) -
        generatedRoot.w_181_0 ctx = reduceMatrix modulus_0 8 1 1 error := by
  apply online1_smudge_hTH ctx digit readout hreadout hodd hbound
  funext row column
  have hcolumn : column = 0 := Subsingleton.elim _ _
  subst column
  simp only [Matrix.sum_apply, Matrix.smul_apply, smul_eq_mul,
    online1_digit_weights ctx F]
set_option maxHeartbeats 400000000 in
/-- Full exported step-1 smudge assembly. No internal PRF, packing, digit or target identity is assumed. -/
theorem online1_smudge_root {backend : BackendContext} {hashModel : HashModel} {params : Params}
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

    (hgq : gadgetMatrixRuns backend 4096 25 gq)
    (digit : Fin 1 → Fin 1 → Fin 8 → Int)
    (hreadout : ∀ r c, reducePoly modulus_0 8 (online1PackedReadout Δ' ρ r c) =
      reducePoly modulus_0 8 (intPoly (digit r c)))
    (hodd : ∀ r c coefficient, digit r c coefficient % 2 = 1)
    (hbound : ∀ r c coefficient, |digit r c coefficient| ≤ (8 : Int) - 1) :
    ∃ es : ErrorMatrix 8 1 1, ∃ M : ErrorMatrix 8 1 1,
      CoeffBound es (1 * (25 * 8 * (8 * (online1ReadoutBudget Ba Bs Bh Bswitch BB)) * 2048)) ∧ CoeffBound M (8 ^ 1 / 2) ∧
      generatedRoot.w_178_0 ctx = Mxx.ExponentLut.Refresh.maskPart (reducePoly modulus_0 8 sm)
        ![generatedRoot.w_178_1 ctx, generatedRoot.w_178_2 ctx] -
        (∑ r, reducePoly modulus_0 8 (online1PackedReadout Δ' ρ r 0) • generatedRoot.w_177_0 ctx r) +
        reduceMatrix modulus_0 8 1 1 es ∧
      (∑ r, reducePoly modulus_0 8 (online1PackedReadout Δ' ρ r 0) • generatedRoot.w_177_0 ctx r) -
        generatedRoot.w_181_0 ctx = reduceMatrix modulus_0 8 1 1 M := by
  have hpacked := fun r ↦ online1_packed_prf_root ctx F hq1 layout hlayout hexact hα hαμ hG ha hpay hg' hg hs BB hrhs0 heB0 sel0 hsel0 selected0 hselected0 μ0 hproduct0 hαμ1 hrhs1 heB1 sel1 hsel1 selected1 hselected1 μ1 hproduct1 hsm hhelp hswitch hμ hwalsh r
  obtain ⟨es, hes, hs⟩ := online1_smudge_of_packed ctx F hq1 layout hlayout hexact hgq
    (fun r port ↦ reducePoly modulus_0 8 (online1PackedReadout Δ' ρ r 0 * unitPair sp port)) hpacked
  obtain ⟨M, hM, hTH⟩ := online1_digit_target_hTH ctx F digit (online1PackedReadout Δ' ρ) hreadout hodd hbound
  refine ⟨es, M, hes, hM, ?_, hTH⟩
  simpa only [unitPair, Matrix.cons_val_one, Matrix.cons_val_zero, mul_one] using hs

end Dio2Proof
