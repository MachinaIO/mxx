import LwrModel1
import LwrCertificate

/-!
# A run is its model (prototype)

Every key, row and error of a run satisfying the generated claim is the corresponding function of
the model at its tape.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {tape : SampleTape}

theorem muOf_zero (ext : GeneratedClaim.ExternalInputs) : muOf ext 0 = Xe (n := N) 0 := by
  simp [muOf, zOf]

/-- The rebased gadget of the model is the run's. -/
theorem gReb_eq {q q' m : Nat} {g' : ExactMatrix q' N 1 m} {g : ExactMatrix q N 1 m}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 m g') (hg : centeredRebaseRuns g' g) :
    gReb q q' m = g := by
  have h1 : gadgetOf q' m = g' :=
    theOutput_eq (R := fun (_ : Unit) (G : ExactMatrix q' N 1 m) ↦
      gadgetMatrixRuns Backend.backend 1048576 m G) (fun _ h ↦ gadgetMatrixRuns_unique h hg') hg'
  rw [gReb, h1]
  exact theOutput_eq (fun _ h ↦ centeredRebaseRuns_unique h hg) hg

theorem gadgetOf_eq {q m : Nat} {G : ExactMatrix q N 1 m}
    (hG : gadgetMatrixRuns Backend.backend 1048576 m G) : gadgetOf q m = G :=
  theOutput_eq (R := fun (_ : Unit) (G : ExactMatrix q N 1 m) ↦
    gadgetMatrixRuns Backend.backend 1048576 m G) (fun _ h ↦ gadgetMatrixRuns_unique h hG) hG

/-- A selector RHS bottom from its payload relation. -/
theorem bot_of_rhs {q m : Nat} {sp : ErrorPoly N} {top bot T : ExactMatrix q N 1 m}
    {e : ErrorMatrix N 1 m}
    (h : reducePoly q N sp • top + bot = T + reduceMatrix q N 1 m e) :
    bot = T + reduceMatrix q N 1 m e - reducePoly q N sp • top := by
  rw [← h]; abel

/-- The state plaintext of block `b + 1` is that of block `b` times the member selected by block
`b`, for one-hot selectors and nonnegative coordinates. -/
theorem muOf_step (ext : GeneratedClaim.ExternalInputs) (b : Fin 5) {sel : Fin 14}
    (hsel : ∀ i, selOf ext b i = if i = sel then 1 else 0)
    (hnn : ∀ b' i, 0 ≤ selOf ext b' i) (ha : ∀ k, 0 ≤ ext.input_3 k) :
    muOf ext (b.val + 1) = alphaOf ext ⟨coords b sel, coords_lt b sel⟩ * muOf ext b := by
  have hblock : ∀ b' : Fin 5, 0 ≤ ∑ i : Fin 14, selOf ext b' i * ext.input_3 ⟨coords b' i, coords_lt b' i⟩ :=
    fun b' ↦ Finset.sum_nonneg fun i _ ↦ mul_nonneg (hnn b' i) (ha _)
  have hfilter : (Finset.univ.filter fun b' : Fin 5 ↦ b'.val < b.val + 1) =
      insert b (Finset.univ.filter fun b' : Fin 5 ↦ b'.val < b.val) := by
    ext b'
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert]
    constructor
    · intro h
      rcases Nat.lt_succ_iff_lt_or_eq.mp h with h | h
      · exact Or.inr h
      · exact Or.inl (Fin.ext h)
    · rintro (rfl | h) <;> omega
  have hnotmem : b ∉ Finset.univ.filter fun b' : Fin 5 ↦ b'.val < b.val := by simp
  have hsel' : ∑ i : Fin 14, selOf ext b i * ext.input_3 ⟨coords b i, coords_lt b i⟩ =
      ext.input_3 ⟨coords b sel, coords_lt b sel⟩ := by
    rw [Finset.sum_eq_single sel]
    · simp [hsel]
    · intro i _ hi; simp [hsel, hi]
    · simp
  have hrest : 0 ≤ (Finset.univ.filter fun b' : Fin 5 ↦ b'.val < b.val).sum fun b' ↦
      ∑ i : Fin 14, selOf ext b' i * ext.input_3 ⟨coords b' i, coords_lt b' i⟩ :=
    Finset.sum_nonneg fun b' _ ↦ hblock b'
  rw [muOf, muOf, alphaOf, Xe_mul, zOf, zOf, hfilter, Finset.sum_insert hnotmem, hsel']
  congr 1
  have hak := ha ⟨coords b sel, coords_lt b sel⟩
  omega

set_option maxRecDepth 100000 in
set_option maxHeartbeats 64000000 in
/-- The keys and the error leaving block `0` are those of the model. -/
theorem chain0 {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (hruns : GeneratedClaim.Runs model external tape execution) :
    ∃ w, Stage_online_eval.generatedRoot.body Backend.backend model tape [4]
        GeneratedClaim.stage_4_params
        (execution.«stage_2».1, execution.«stage_2».2.1, execution.«stage_2».2.2.1,
          execution.«stage_2».2.2.2.1, execution.«stage_3».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          external.input_0, execution.«stage_3».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_3».2.2.1, execution.«stage_3».2.2.2.1, execution.«stage_3».1,
          execution.«stage_3».2.2.2.2.1, execution.«stage_3».2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, ()) execution.«stage_4» w ∧
      ![![w.w_42_0, w.w_42_1], ![w.w_42_2, w.w_42_3]] = key1 model external.input_0 external tape ∧
      RelE (maskM tape) ![![w.w_42_0, w.w_42_1], ![w.w_42_2, w.w_42_3]]
        (∑ i : Fin 14, fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) *
          (alphaOf external ⟨coords 0 i, coords_lt 0 i⟩ * muOf external 0 * nuOf external 0 i) *
            unitPair (payloadM tape) j) • gReb Q6 Q5 10)
        ![w.w_42_4, w.w_42_5] (err1 model external.input_0 external tape) := by
  obtain ⟨hvalid, hpay, hmask, hpre, henc, honlineEval, hdec, hreq, hideal, hreqtrue⟩ := hruns
  obtain ⟨sp, hsp, hspn⟩ := secret_ternary hpay
  obtain ⟨sm, hsm, hsmn⟩ := mask_ternary hmask
  obtain ⟨-, ⟨hc1, hc2⟩, ⟨he1, he2⟩, ha, hac, hae, hs0, hx0, hs1, hx1, hs2, hx2, hs3, hx3, hs4, hx4⟩ :=
    hvalid
  obtain ⟨hmem, hsum0, hsum1, hsum2, hsum3, hsum4⟩ := requirement_holds hreq hreqtrue
  have hid := ideal_value hideal
  set a := external.input_3
  set s0 := external.input_6
  set s1 := external.input_8
  set s2 := external.input_10
  set s3 := external.input_12
  set s4 := external.input_14
  set x0 := external.input_7
  set x1 := external.input_9
  set x2 := external.input_11
  set x3 := external.input_13
  set x4 := external.input_15
  -- The selected cells and the plaintexts.
  obtain ⟨sel0, hsel0⟩ := one_hot hs0 hsum0
  obtain ⟨sel1, hsel1⟩ := one_hot hs1 hsum1
  obtain ⟨sel2, hsel2⟩ := one_hot hs2 hsum2
  obtain ⟨sel3, hsel3⟩ := one_hot hs3 hsum3
  obtain ⟨sel4, hsel4⟩ := one_hot hs4 hsum4
  let α : Fin 17 → ErrorPoly N := fun k ↦ Xe (2 * (a k).toNat)
  have hα : ∀ k, message (n := N) (external.input_4 k) (external.input_5 k) = α k := fun k ↦
    message_member (hmem k).1 (hmem k).2 (hae k)
  have hν0 : ∀ i, message (n := N) (s0 i) (x0 i) = (s0 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx0 i; omega)
  have hν1 : ∀ i, message (n := N) (s1 i) (x1 i) = (s1 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx1 i; omega)
  have hν2 : ∀ i, message (n := N) (s2 i) (x2 i) = (s2 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx2 i; omega)
  have hν3 : ∀ i, message (n := N) (s3 i) (x3 i) = (s3 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx3 i; omega)
  have hν4 : ∀ i, message (n := N) (s4 i) (x4 i) = (s4 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx4 i; omega)
  have hμ0 : message (n := N) external.input_1 external.input_2 = Xe 0 :=
    message_one (k := 0) (by omega) (by simp; omega) (by simp only [N]; omega)
  have hspM := payload_model hpay
  have hsmM := mask_model hmask
  obtain ⟨G14, hG14⟩ := gadget_exists (m := 14) layout7 rfl rfl rfl
  obtain ⟨g12, hg12⟩ := gadget_exists (m := 12) layout6 rfl rfl rfl
  obtain ⟨g10, hg10⟩ := gadget_exists (m := 10) layout5 rfl rfl rfl
  have hg12u : centeredRebaseRuns g12 (centeredRebase g12 : ExactMatrix Q7 N 1 12) := ⟨hQ7, rfl⟩
  have hg10u : centeredRebaseRuns g10 (centeredRebase g10 : ExactMatrix Q6 N 1 10) := ⟨hQ6, rfl⟩
  -- Encryption and preprocessing.
  have hstate := encrypt_stateE henc hsmM hspM hg12 hg12u
  rw [hμ0] at hstate
  have hmembers := encrypt_memberE henc hsmM hG14
  have hmemα : ∀ k : Fin 17, RelE (maskM tape) (memberKey model external.input_0 k)
      (fun j ↦ reducePoly Q7 N (α k * unitPair (maskM tape) j) • G14)
      ![execution.«stage_3».2.2.1 k, execution.«stage_3».2.2.2.1 k] (memberError tape [3] k) :=
    fun k ↦ by rw [← hα k]; exact (hmembers k).1
  have hpayα : ∀ k : Fin 17, execution.«stage_3».2.2.2.2.1 k = fun _ _ ↦ reducePoly Q7 N (α k) :=
    fun k ↦ by rw [← hα k]; exact (hmembers k).2
  have hrhs0 := preprocess_selector0E hpre hspM hg10 hg10u
  -- Block 0.
  obtain ⟨w, hb⟩ := honlineEval
  obtain ⟨hk, -, hrel⟩ := block0M hb (μ := Xe 0) (α := α) (ν := fun i ↦ message (n := N) (s0 i) (x0 i))
    hG14 hg12 hg12u hmemα hpayα hstate (fun i ↦ (hrhs0 i).2)
  -- The run's bottoms, rows and constants are the model's.
  have hbot : ∀ i : Fin 14, ![execution.«stage_2».2.2.2.2.2.2.2.2.1 i,
      execution.«stage_2».2.2.2.2.2.2.2.2.2.1 i] = bot0 model external.input_0 external tape i := by
    intro i
    funext j
    rw [bot_of_rhs ((hrhs0 i).2 j), bot0, gReb_eq hg10 hg10u, hν0 i]
    rfl
  have hmrow : ∀ k : Fin 17, ![execution.«stage_3».2.2.1 k, execution.«stage_3».2.2.2.1 k] =
      memberRow model external.input_0 external tape k := by
    intro k
    rw [(hmemα k).eq_encRow, memberRow]
    congr 1
    funext j
    rw [memberTarget, gadgetOf_eq hG14]
    rfl
  have hsrow : ![execution.«stage_3».1, execution.«stage_3».2.1] =
      stateRow0 model external.input_0 external tape := by
    rw [hstate.eq_encRow, stateRow0]
    congr 1
    funext j
    rw [stateTarget0, gReb_eq hg12 hg12u, muOf_zero]
  refine ⟨w, hb, ?_, ?_⟩
  · rw [hk, key1]
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    rw [hbot i]
  · have h := hrel
    simp only [hbot, hmrow, hsrow, hpayα, hν0] at h
    convert h using 2 with i
    funext j
    rw [muOf_zero, gReb_eq hg10 hg10u]
    rfl

set_option maxRecDepth 100000 in
set_option maxHeartbeats 64000000 in
/-- The keys and the error leaving block `1` are those of the model, and the numerators of blocks
`0` and `1` are divisible by their dropped factors. -/
theorem chain1 {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (hruns : GeneratedClaim.Runs model external tape execution) :
    ∃ w, Stage_online_eval.generatedRoot.body Backend.backend model tape [4]
        GeneratedClaim.stage_4_params
        (execution.«stage_2».1, execution.«stage_2».2.1, execution.«stage_2».2.2.1,
          execution.«stage_2».2.2.2.1, execution.«stage_3».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          external.input_0, execution.«stage_3».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_3».2.2.1, execution.«stage_3».2.2.2.1, execution.«stage_3».1,
          execution.«stage_3».2.2.2.2.1, execution.«stage_3».2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, ()) execution.«stage_4» w ∧
      div0 model external.input_0 external tape ∧ div1 model external.input_0 external tape ∧
      ![![w.w_64_0, w.w_64_1], ![w.w_64_2, w.w_64_3]] = key2 model external.input_0 external tape ∧
      RelE (maskM tape) ![![w.w_64_0, w.w_64_1], ![w.w_64_2, w.w_64_3]]
        (∑ i : Fin 14, fun j ↦ reducePoly Q5 N (((P5 : Nat) : Int) *
          (alphaOf external ⟨coords 1 i, coords_lt 1 i⟩ * muOf external 1 * nuOf external 1 i) *
            unitPair (payloadM tape) j) • gReb Q5 Q4 8)
        ![w.w_64_4, w.w_64_5] (err2 model external.input_0 external tape) := by
  obtain ⟨hvalid, hpay, hmask, hpre, henc, honlineEval, hdec, hreq, hideal, hreqtrue⟩ := hruns
  obtain ⟨sp, hsp, hspn⟩ := secret_ternary hpay
  obtain ⟨sm, hsm, hsmn⟩ := mask_ternary hmask
  obtain ⟨-, ⟨hc1, hc2⟩, ⟨he1, he2⟩, ha, hac, hae, hs0, hx0, hs1, hx1, hs2, hx2, hs3, hx3, hs4, hx4⟩ :=
    hvalid
  obtain ⟨hmem, hsum0, hsum1, hsum2, hsum3, hsum4⟩ := requirement_holds hreq hreqtrue
  have hid := ideal_value hideal
  set a := external.input_3
  set s0 := external.input_6
  set s1 := external.input_8
  set s2 := external.input_10
  set s3 := external.input_12
  set s4 := external.input_14
  set x0 := external.input_7
  set x1 := external.input_9
  set x2 := external.input_11
  set x3 := external.input_13
  set x4 := external.input_15
  -- The selected cells and the plaintexts.
  obtain ⟨sel0, hsel0⟩ := one_hot hs0 hsum0
  obtain ⟨sel1, hsel1⟩ := one_hot hs1 hsum1
  obtain ⟨sel2, hsel2⟩ := one_hot hs2 hsum2
  obtain ⟨sel3, hsel3⟩ := one_hot hs3 hsum3
  obtain ⟨sel4, hsel4⟩ := one_hot hs4 hsum4
  let α : Fin 17 → ErrorPoly N := fun k ↦ Xe (2 * (a k).toNat)
  have hα : ∀ k, message (n := N) (external.input_4 k) (external.input_5 k) = α k := fun k ↦
    message_member (hmem k).1 (hmem k).2 (hae k)
  have hν0 : ∀ i, message (n := N) (s0 i) (x0 i) = (s0 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx0 i; omega)
  have hν1 : ∀ i, message (n := N) (s1 i) (x1 i) = (s1 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx1 i; omega)
  have hν2 : ∀ i, message (n := N) (s2 i) (x2 i) = (s2 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx2 i; omega)
  have hν3 : ∀ i, message (n := N) (s3 i) (x3 i) = (s3 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx3 i; omega)
  have hν4 : ∀ i, message (n := N) (s4 i) (x4 i) = (s4 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx4 i; omega)
  have hμ0 : message (n := N) external.input_1 external.input_2 = Xe 0 :=
    message_one (k := 0) (by omega) (by simp; omega) (by simp only [N]; omega)
  have hspM := payload_model hpay
  have hsmM := mask_model hmask
  obtain ⟨G14, hG14⟩ := gadget_exists (m := 14) layout7 rfl rfl rfl
  obtain ⟨g12, hg12⟩ := gadget_exists (m := 12) layout6 rfl rfl rfl
  obtain ⟨g10, hg10⟩ := gadget_exists (m := 10) layout5 rfl rfl rfl
  have hg12u : centeredRebaseRuns g12 (centeredRebase g12 : ExactMatrix Q7 N 1 12) := ⟨hQ7, rfl⟩
  have hg10u : centeredRebaseRuns g10 (centeredRebase g10 : ExactMatrix Q6 N 1 10) := ⟨hQ6, rfl⟩
  -- Encryption and preprocessing.
  have hstate := encrypt_stateE henc hsmM hspM hg12 hg12u
  rw [hμ0] at hstate
  have hmembers := encrypt_memberE henc hsmM hG14
  have hmemα : ∀ k : Fin 17, RelE (maskM tape) (memberKey model external.input_0 k)
      (fun j ↦ reducePoly Q7 N (α k * unitPair (maskM tape) j) • G14)
      ![execution.«stage_3».2.2.1 k, execution.«stage_3».2.2.2.1 k] (memberError tape [3] k) :=
    fun k ↦ by rw [← hα k]; exact (hmembers k).1
  have hpayα : ∀ k : Fin 17, execution.«stage_3».2.2.2.2.1 k = fun _ _ ↦ reducePoly Q7 N (α k) :=
    fun k ↦ by rw [← hα k]; exact (hmembers k).2
  have hrhs0 := preprocess_selector0E hpre hspM hg10 hg10u
  -- Block 0.
  obtain ⟨w, hb⟩ := honlineEval
  obtain ⟨hk, hdiv, hrel⟩ := block0M hb (μ := Xe 0) (α := α) (ν := fun i ↦ message (n := N) (s0 i) (x0 i))
    hG14 hg12 hg12u hmemα hpayα hstate (fun i ↦ (hrhs0 i).2)
  -- The run's bottoms, rows and constants are the model's.
  have hbot : ∀ i : Fin 14, ![execution.«stage_2».2.2.2.2.2.2.2.2.1 i,
      execution.«stage_2».2.2.2.2.2.2.2.2.2.1 i] = bot0 model external.input_0 external tape i := by
    intro i
    funext j
    rw [bot_of_rhs ((hrhs0 i).2 j), bot0, gReb_eq hg10 hg10u, hν0 i]
    rfl
  have hmrow : ∀ k : Fin 17, ![execution.«stage_3».2.2.1 k, execution.«stage_3».2.2.2.1 k] =
      memberRow model external.input_0 external tape k := by
    intro k
    rw [(hmemα k).eq_encRow, memberRow]
    congr 1
    funext j
    rw [memberTarget, gadgetOf_eq hG14]
    rfl
  have hsrow : ![execution.«stage_3».1, execution.«stage_3».2.1] =
      stateRow0 model external.input_0 external tape := by
    rw [hstate.eq_encRow, stateRow0]
    congr 1
    funext j
    rw [stateTarget0, gReb_eq hg12 hg12u, muOf_zero]
  have hk0M : ![![w.w_42_0, w.w_42_1], ![w.w_42_2, w.w_42_3]] = key1 model external.input_0 external tape := by
    rw [hk, key1]
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    rw [hbot i]
  have hrel0M : RelE (maskM tape) ![![w.w_42_0, w.w_42_1], ![w.w_42_2, w.w_42_3]]
      (∑ i : Fin 14, fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) *
        (alphaOf external ⟨coords 0 i, coords_lt 0 i⟩ * muOf external 0 * nuOf external 0 i) *
          unitPair (payloadM tape) j) • gReb Q6 Q5 10)
      ![w.w_42_4, w.w_42_5] (err1 model external.input_0 external tape) := by
    have h := hrel
    simp only [hbot, hmrow, hsrow, hpayα, hν0] at h
    convert h using 2 with i
    funext j
    rw [muOf_zero, gReb_eq hg10 hg10u]
    rfl
  have hdiv0M : div0 model external.input_0 external tape := by
    intro i j row column
    have h := hdiv i j row column
    simp only [hbot, hmrow, hsrow, hpayα] at h
    exact h
  -- Block 1: the block-0 target collapses to the state target of block `1`.
  have hnn : ∀ b' i, 0 ≤ selOf external b' i := by
    intro b' i
    fin_cases b'
    exacts [(hs0 i).1, (hs1 i).1, (hs2 i).1, (hs3 i).1, (hs4 i).1]
  have hstep0 : muOf external 1 = alphaOf external ⟨coords 0 sel0, coords_lt 0 sel0⟩ * muOf external 0 :=
    muOf_step external 0 (sel := sel0) hsel0 hnn (fun k ↦ (ha k).1)
  have hin1 := hrel0M.target_eq (collapse _ _ _ (fun i ↦ alphaOf external ⟨coords 0 i, coords_lt 0 i⟩)
    (nuOf external 0) _ sel0 hsel0 (fun i ↦ rfl))
  beta_reduce at hin1
  rw [← hstep0] at hin1
  rw [hk0M] at hin1
  obtain ⟨g8, hg8⟩ := gadget_exists (m := 8) layout4 rfl rfl rfl
  have hg8u : centeredRebaseRuns g8 (centeredRebase g8 : ExactMatrix Q5 N 1 8) := ⟨hQ5, rfl⟩
  have hrhs1 := preprocess_selector1E hpre hspM hg8 hg8u
  rw [gReb_eq hg10 hg10u] at hin1
  have hsrow1 : ![w.w_42_4, w.w_42_5] = stateRow1 model external.input_0 external tape := by
    rw [hin1.eq_encRow, stateRow1]
    congr 1
    funext j
    rw [stateTarget1, gReb_eq hg10 hg10u]
  rw [← hk0M] at hin1
  obtain ⟨hk1, hadiv1, hdiv1, hrel1⟩ := block1M hb (μ := muOf external 1) (α := α)
    (ν := fun i ↦ message (n := N) (s1 i) (x1 i)) hG14 hg10 hg10u hmemα hpayα hin1
    (fun i ↦ (hrhs1 i).2)
  have hbot1 : ∀ i : Fin 14, ![execution.«stage_2».2.2.2.2.2.2.2.2.2.2.1 i,
      execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.1 i] = bot1 model external.input_0 external tape i := by
    intro i
    funext j
    rw [bot_of_rhs ((hrhs1 i).2 j), bot1, gReb_eq hg8 hg8u, hν1 i]
    rfl
  have hk1M : ![![w.w_64_0, w.w_64_1], ![w.w_64_2, w.w_64_3]] = key2 model external.input_0 external tape := by
    rw [hk1, key2, ← hk0M]
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    rw [hbot1 i]
  have hrel1M : RelE (maskM tape) ![![w.w_64_0, w.w_64_1], ![w.w_64_2, w.w_64_3]]
      (∑ i : Fin 14, fun j ↦ reducePoly Q5 N (((P5 : Nat) : Int) *
        (alphaOf external ⟨coords 1 i, coords_lt 1 i⟩ * muOf external 1 * nuOf external 1 i) *
          unitPair (payloadM tape) j) • gReb Q5 Q4 8)
      ![w.w_64_4, w.w_64_5] (err2 model external.input_0 external tape) := by
    have h := hrel1
    simp only [hbot1, hmrow, hsrow1, hν1, hk0M] at h
    convert h using 2 with i
    funext j
    rw [gReb_eq hg8 hg8u]
    rfl
  have hdiv1M : div1 model external.input_0 external tape := by
    refine ⟨fun i j row column ↦ ?_, fun i j row column ↦ ?_⟩
    · have h := hadiv1 i j row column
      simp only [hmrow] at h
      exact h
    · have h := hdiv1 i j row column
      simp only [hbot1, hmrow, hsrow1, hk0M] at h
      exact h
  -- End of block 1.
  exact ⟨w, hb, hdiv0M, hdiv1M, hk1M, hrel1M⟩

end LwrProof
