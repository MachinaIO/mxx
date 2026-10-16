import Dio2Rel
import Dio2Online0Sigma
import Dio2Online0Row

/-!
# The refresh inputs of step 0

The rows the refresh parts of step 0 combine: the σ carrier `g` (component `1` of the σ sampler,
re-targeted to the gadget) and the one-public row `c₁`.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof
  RefreshTopProof.Gen RefreshTopProof.GenIface RefreshTopProof.GenFinal

variable {n : Nat}

/-- Component `j` of an encoding relation in the masked form. -/
theorem Rel.comp {q m : Nat} {sm : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {t c : Fin 2 → ExactMatrix q n 1 m} {B : Nat} (h : Rel sm A t c B) (j : Fin 2) :
    ∃ e : ErrorMatrix n 1 m, CoeffBound e B ∧
      c j = maskPart (reducePoly q n sm) (fun k ↦ A k j) - t j + reduceMatrix q n 1 m e := by
  obtain ⟨e, he, hc⟩ := h
  exact ⟨e j, he j, by rw [hc j, maskPart]⟩

set_option maxHeartbeats 8000000 in
/-- **A placed and summed sampler, re-targeted at component `1`.** A sampler total encoding
`Σ_o X^o (Δ' ν_o (s_p, 1)) G` acted on by the digits `D` of `X` is, at component `1`,
`s̄ (K D) - (Δ' Σ_o X^o ν_o) X + e`. -/
theorem sampler_gamma_one (hn : 0 < n) {q m m' M : Nat} (hq : 1 < q) {backend : BackendContext}
    {base digits : Int} (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0) {G : ExactMatrix q n 1 m}
    (hG : gadgetMatrixRuns backend base digits G) {X : ExactMatrix q n 1 m'}
    {D : ExactMatrix q n m m'} (hD : gadgetDecomposeRuns backend base digits X D)
    {sm sp Δ' : ErrorPoly n} {ν : Fin M → ErrorPoly n}
    {x : Tup (ExactMatrix q n 1 m) 6} {B : Nat}
    (h : EncRel sm x (∑ o : Fin M, fun j ↦ reducePoly q n (Xe (n := n) (o : Nat)) •
      (reducePoly q n (Δ' * ν o * unitPair sp j) • G)) B) :
    ∃ e : ErrorMatrix n 1 m', CoeffBound e (m * n * B * (base.toNat / 2)) ∧
      x.get 5 * D = maskPart (reducePoly q n sm) (fun k ↦ Enc.A x k 1 * D) -
        reducePoly q n (Δ' * ∑ o : Fin M, Xe (n := n) (o : Nat) * ν o) • X + reduceMatrix q n 1 m' e := by
  have ht : (∑ o : Fin M, fun j ↦ reducePoly q n (Xe (n := n) (o : Nat)) •
      (reducePoly q n (Δ' * ν o * unitPair sp j) • G)) =
      fun j ↦ reducePoly q n (Δ' * (∑ o : Fin M, Xe (n := n) (o : Nat) * ν o) * unitPair sp j) • G := by
    funext j
    rw [Finset.sum_apply]
    simp only [smul_smul, ← map_mul]
    rw [← Finset.sum_smul, ← map_sum]
    congr 2
    rw [Finset.mul_sum, Finset.sum_mul]
    exact Finset.sum_congr rfl fun o _ ↦ by ring
  unfold EncRel at h
  rw [ht] at h
  have hg := Rel.gamma hn hq layout hlayout hexact hG hD h
  obtain ⟨e, he, hc⟩ := Rel.comp hg 1
  refine ⟨e, he, ?_⟩
  simp only [Enc.c, Fin.isValue, Matrix.cons_val_one, Matrix.cons_val_zero] at hc
  rw [hc]
  simp [unitPair]

open W_Stage_online_0 in
/-- The gadget re-encoded at its own modulus and times one is the gadget. -/
theorem gadget_rebase_one {q m : Nat} (hq : 1 < q) {G G' : ExactMatrix q 8 1 m}
    (h : centeredRebaseRuns G G') : matrixMulScalarRight G' (matrixPolynomial [(1 : Int)]) = G := by
  have e : (fun row column ↦ reducePoly q 8 (centeredIntLift (G row column))) = G :=
    reduceMatrix_centeredIntLift hq (by norm_num) G
  rw [centeredRebaseRuns_reduce h]
  simp only [reduceMatrix] at e ⊢
  rw [e]
  funext r c
  simp [matrixMulScalarRight_eq, matrixPolynomial_single]

open W_Stage_online_0 in
set_option maxHeartbeats 16000000 in
/-- **The σ carrier of step 0** (`w_63`): component `1` of the σ sampler total (`hσ`, from
`online0_sigma_total` with `g_q` the gadget `w_57`) re-targeted to the gadget. -/
theorem online0_g {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp Δ' : ErrorPoly 8} {ν : Fin 8 → ErrorPoly 8} {B : Nat}
    (hσ : EncRel sm (generatedRoot.w_56_0 ctx, (generatedRoot.w_56_1 ctx, (generatedRoot.w_56_2 ctx,
        (generatedRoot.w_56_3 ctx, (generatedRoot.w_56_4 ctx, (generatedRoot.w_56_5 ctx, ()))))))
      (∑ o : Fin 8, fun j ↦ reducePoly modulus_0 8 (Xe (n := 8) (o : Nat)) •
        (reducePoly modulus_0 8 (Δ' * ν o * unitPair sp j) • generatedRoot.w_57_0 ctx)) B) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e (25 * 8 * B * 2048) ∧
      generatedRoot.w_63_0 ctx = maskPart (reducePoly modulus_0 8 sm)
        (fun k ↦ ![![generatedRoot.w_56_0 ctx, generatedRoot.w_56_1 ctx],
          ![generatedRoot.w_56_2 ctx, generatedRoot.w_56_3 ctx]] k 1 * generatedRoot.w_62_0 ctx) -
        reducePoly modulus_0 8 (Δ' * ∑ o : Fin 8, Xe (n := 8) (o : Nat) * ν o) •
          generatedRoot.w_57_0 ctx + reduceMatrix modulus_0 8 1 25 e := by
  have hX : generatedRoot.w_60_0 ctx = generatedRoot.w_57_0 ctx :=
    gadget_rebase_one hq1 F.c1.h58_0
  have hD : generatedRoot.w_62_0 ctx = generatedRoot.w_61_0 ctx :=
    compactRebase_self_gen (by norm_num) hq1 F.c1.h62_0
  have h61 := F.c1.h61_0
  rw [hX] at h61
  obtain ⟨e, he, hc⟩ := sampler_gamma_one (by norm_num) hq1 layout hlayout hexact F.c1.h57_0 h61 hσ
  refine ⟨e, he, ?_⟩
  show generatedRoot.w_56_5 ctx * generatedRoot.w_62_0 ctx = _
  rw [hD]
  exact hc

open W_Stage_online_0 in
set_option maxHeartbeats 16000000 in
/-- **The one-public row of step 0** (`w_70`): component `1` of the one-public root row (label
member `1`, `hone`) re-targeted to the gadget. -/
theorem online0_c1 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm : ErrorPoly 8} {K : Fin 2 → Fin 2 → ExactMatrix modulus_0 8 1 25} {c0 : ExactMatrix modulus_0 8 1 25}
    {B : Nat}
    (hone : Rel sm K (fun j ↦ reducePoly modulus_0 8 (1 * unitPair sm j) • generatedRoot.w_57_0 ctx)
      ![c0, generatedRoot.w_17_0 ctx 1] B) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e (25 * 8 * B * 2048) ∧
      generatedRoot.w_70_0 ctx = maskPart (reducePoly modulus_0 8 sm)
        (fun k ↦ K k 1 * generatedRoot.w_69_0 ctx) - generatedRoot.w_57_0 ctx +
          reduceMatrix modulus_0 8 1 25 e := by
  have hX : generatedRoot.w_67_0 ctx = generatedRoot.w_57_0 ctx :=
    gadget_rebase_one hq1 F.c1.h65_0
  have hD : generatedRoot.w_69_0 ctx = generatedRoot.w_68_0 ctx :=
    compactRebase_self_gen (by norm_num) hq1 F.c1.h69_0
  have h68 := F.c1.h68_0
  rw [hX] at h68
  have hg := Rel.gamma (by norm_num) hq1 layout hlayout hexact F.c1.h57_0 h68 hone
  obtain ⟨e, he, hc⟩ := Rel.comp hg 1
  refine ⟨e, he, ?_⟩
  have e64 : generatedRoot.w_64_0 ctx = generatedRoot.w_17_0 ctx 1 := by
    obtain ⟨pos, hpos, h⟩ := F.c1.h64_0
    rw [h]; congr 1; exact Fin.ext (by omega)
  show generatedRoot.w_64_0 ctx * generatedRoot.w_69_0 ctx = _
  rw [e64, hD]
  simp only [Fin.isValue, Matrix.cons_val_one, Matrix.cons_val_zero] at hc
  rw [hc]
  simp [unitPair]

/-- A matrix re-encoded at its own modulus and scaled by one on the left is itself. -/
theorem rebase_left_one {q m : Nat} (hq : 1 < q) {G G' : ExactMatrix q 8 1 m}
    (h : centeredRebaseRuns G G') : matrixMulScalarLeft (matrixPolynomial [(1 : Int)]) G' = G := by
  have e : (fun row column ↦ reducePoly q 8 (centeredIntLift (G row column))) = G :=
    reduceMatrix_centeredIntLift hq (by norm_num) G
  rw [centeredRebaseRuns_reduce h]
  simp only [reduceMatrix] at e ⊢
  rw [e]
  funext r c
  obtain rfl : r = 0 := Subsingleton.elim _ _
  simp [matrixMulScalarLeft_eq, matrixPolynomial_single, Matrix.mul_apply]

open W_Stage_online_0 in
/-- The gadget form the refresh targets of step 0 are re-targeted to (`w_234`) is the gadget
`w_229`. -/
theorem online0_target_form {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) (hq1 : 1 < modulus_0) :
    generatedRoot.w_234_0 ctx = generatedRoot.w_229_0 ctx := by
  have h1 : generatedRoot.w_231_0 ctx = generatedRoot.w_229_0 ctx := rebase_left_one hq1 F.c6.h230_0
  have h2 := F.c6.h232_0
  rw [h1] at h2
  exact gadget_rebase_one hq1 h2

open W_Stage_online_0 in
set_option maxHeartbeats 16000000 in
/-- **The ONEPRIV targets of step 0** (`w_237`, `w_238`): the one-private root row (`hpriv`, target
`(t̄, 1)` against the gadget) re-targeted to the gadget, at both components. -/
theorem online0_target_priv {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm tp : ErrorPoly 8} {K : Fin 2 → Fin 2 → ExactMatrix modulus_0 8 1 25} {B : Nat}
    (hpriv : Rel sm K (fun j ↦ reducePoly modulus_0 8 (1 * unitPair tp j) • generatedRoot.w_229_0 ctx)
      ![generatedRoot.w_21_0 ctx, generatedRoot.w_25_0 ctx] B) (j : Fin 2) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e (25 * 8 * B * 2048) ∧
      ![generatedRoot.w_237_0 ctx, generatedRoot.w_238_0 ctx] j =
        maskPart (reducePoly modulus_0 8 sm) (fun k ↦ K k j * generatedRoot.w_236_0 ctx) -
          reducePoly modulus_0 8 (1 * unitPair tp j) • generatedRoot.w_229_0 ctx +
          reduceMatrix modulus_0 8 1 25 e := by
  have hX := online0_target_form ctx F hq1
  have hD : generatedRoot.w_236_0 ctx = generatedRoot.w_235_0 ctx :=
    compactRebase_self_gen (by norm_num) hq1 F.c6.h236_0
  have h235 := F.c6.h235_0
  rw [hX] at h235
  have hg := Rel.gamma (by norm_num) hq1 layout hlayout hexact F.c6.h229_0 h235 hpriv
  obtain ⟨e, he, hc⟩ := Rel.comp hg j
  refine ⟨e, he, ?_⟩
  show ![generatedRoot.w_21_0 ctx * generatedRoot.w_236_0 ctx,
    generatedRoot.w_25_0 ctx * generatedRoot.w_236_0 ctx] j = _
  rw [hD]
  fin_cases j
  · exact hc
  · exact hc

open W_Stage_online_0 in
set_option maxHeartbeats 16000000 in
/-- **The ONEPUB targets of step 0** (`E^σ = (w_244, w_245)`): the σ carrier and the one-public
row (their forms `hg`, `hc₁`, from `online0_g` and `online0_c1`) re-targeted to the gadget. -/
theorem online0_target_pub {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {s β : ExactPoly modulus_0 8} {Kg K₁ : Fin 2 → ExactMatrix modulus_0 8 1 25}
    {eg e₁ : ErrorMatrix 8 1 25} {Beg Be1 : Nat} (heg : CoeffBound eg Beg) (he1 : CoeffBound e₁ Be1)
    (hg : generatedRoot.w_63_0 ctx = maskPart s Kg - β • generatedRoot.w_229_0 ctx +
      reduceMatrix modulus_0 8 1 25 eg)
    (hc₁ : generatedRoot.w_70_0 ctx = maskPart s K₁ - generatedRoot.w_229_0 ctx +
      reduceMatrix modulus_0 8 1 25 e₁) :
    (∃ e : ErrorMatrix 8 1 25, CoeffBound e (25 * 8 * Beg * 2048) ∧
      generatedRoot.w_244_0 ctx = maskPart s (fun k ↦ Kg k * generatedRoot.w_243_0 ctx) -
        β • generatedRoot.w_229_0 ctx + reduceMatrix modulus_0 8 1 25 e) ∧
    (∃ e : ErrorMatrix 8 1 25, CoeffBound e (25 * 8 * Be1 * 2048) ∧
      generatedRoot.w_245_0 ctx = maskPart s (fun k ↦ K₁ k * generatedRoot.w_243_0 ctx) -
        generatedRoot.w_229_0 ctx + reduceMatrix modulus_0 8 1 25 e) := by
  have hX : generatedRoot.w_241_0 ctx = generatedRoot.w_229_0 ctx := by
    have h1 : generatedRoot.w_231_0 ctx = generatedRoot.w_229_0 ctx := rebase_left_one hq1 F.c6.h230_0
    have h2 := F.c6.h239_0
    rw [h1] at h2
    exact gadget_rebase_one hq1 h2
  have hD : generatedRoot.w_243_0 ctx = generatedRoot.w_242_0 ctx :=
    compactRebase_self_gen (by norm_num) hq1 F.c6.h243_0
  have h242 := F.c6.h242_0
  rw [hX] at h242
  have hGD := gadget_mul_decompose hq1 (by norm_num) layout hlayout hexact F.c6.h229_0 h242
  obtain ⟨d, hd, hdb⟩ := gadgetDecomposeRuns_small hq1 (by norm_num) h242
  rw [← hD] at hGD hd
  refine ⟨⟨eg * d, coeffBound_mul (by norm_num) heg hdb, ?_⟩, ⟨e₁ * d, coeffBound_mul (by norm_num) he1 hdb, ?_⟩⟩
  · show generatedRoot.w_63_0 ctx * generatedRoot.w_243_0 ctx = _
    rw [hg, Matrix.add_mul, Matrix.sub_mul, maskPart_mul, Matrix.smul_mul, hGD, hd, ← reduceMatrix_mul]
  · show generatedRoot.w_70_0 ctx * generatedRoot.w_243_0 ctx = _
    rw [hc₁, Matrix.add_mul, Matrix.sub_mul, maskPart_mul, hGD, hd, ← reduceMatrix_mul]

open W_Stage_online_0 in
set_option maxHeartbeats 16000000 in
/-- **The STATE targets of step 0** (`w_356`, `w_371`): the state LUT's components (`h0`, `h1`, from
`online0_state_lut` with `g_q` the gadget `w_229`) re-targeted to the gadget. -/
theorem online0_target_state {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {s : ExactPoly modulus_0 8} {K : Fin 2 → Fin 2 → ExactMatrix modulus_0 8 1 25}
    {x : Fin 2 → ExactPoly modulus_0 8} {e : Fin 2 → ErrorMatrix 8 1 25} {B : Nat}
    (he : ∀ j, CoeffBound (e j) B)
    (h : ∀ j, ![generatedRoot.w_350_0 ctx, generatedRoot.w_370_0 ctx] j =
      maskPart s (fun k ↦ K k j) - x j • generatedRoot.w_229_0 ctx +
        reduceMatrix modulus_0 8 1 25 (e j)) (j : Fin 2) :
    ∃ e' : ErrorMatrix 8 1 25, CoeffBound e' (25 * 8 * B * 2048) ∧
      ![generatedRoot.w_356_0 ctx, generatedRoot.w_371_0 ctx] j =
        maskPart s (fun k ↦ K k j * generatedRoot.w_355_0 ctx) - x j • generatedRoot.w_229_0 ctx +
          reduceMatrix modulus_0 8 1 25 e' := by
  have hX : generatedRoot.w_353_0 ctx = generatedRoot.w_229_0 ctx := by
    have h1 : generatedRoot.w_231_0 ctx = generatedRoot.w_229_0 ctx := rebase_left_one hq1 F.c6.h230_0
    have h2 := F.c12.h351_0
    rw [h1] at h2
    exact gadget_rebase_one hq1 h2
  have hD : generatedRoot.w_355_0 ctx = generatedRoot.w_354_0 ctx :=
    compactRebase_self_gen (by norm_num) hq1 F.c12.h355_0
  have h354 := F.c12.h354_0
  rw [hX] at h354
  have hGD := gadget_mul_decompose hq1 (by norm_num) layout hlayout hexact F.c6.h229_0 h354
  obtain ⟨d, hd, hdb⟩ := gadgetDecomposeRuns_small hq1 (by norm_num) h354
  rw [← hD] at hGD hd
  refine ⟨e j * d, coeffBound_mul (by norm_num) (he j) hdb, ?_⟩
  have hj := h j
  show ![generatedRoot.w_350_0 ctx * generatedRoot.w_355_0 ctx,
    generatedRoot.w_370_0 ctx * generatedRoot.w_355_0 ctx] j = _
  have hm : ![generatedRoot.w_350_0 ctx * generatedRoot.w_355_0 ctx,
      generatedRoot.w_370_0 ctx * generatedRoot.w_355_0 ctx] j =
      ![generatedRoot.w_350_0 ctx, generatedRoot.w_370_0 ctx] j * generatedRoot.w_355_0 ctx := by
    fin_cases j <;> rfl
  rw [hm, hj, Matrix.add_mul, Matrix.sub_mul, maskPart_mul, Matrix.smul_mul, hGD, hd,
    ← reduceMatrix_mul]

end Dio2Proof
