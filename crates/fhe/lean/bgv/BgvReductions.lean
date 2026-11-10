import BgvPrograms
import BgvSetupForms
import BgvSecurity
import BgvBounded

set_option backward.isDefEq.respectTransparency false

/-! The reductions of `GeneratedScheme.BoundedSecurityClaim` are those of
`GeneratedScheme.SimulatedSecurity`: each program simulator of `Simulators.lean`, run through
`MxxFhe.reduceProgram`, is the same computation as the generated simulator run through
`MxxFhe.reduce`. With the costs of `BgvBounded`, the bounded claim follows from `security`. -/

namespace BgvSecurity
open MxxSecurity MxxSecurity.IR Mxx.Primitives MxxRuntime

variable {ctx : Context}

section Programs
open GeneratedSimulators

/-! ## The setup programs and the ciphertext simulator's encryption program -/

theorem prodQP : (18014398507892737 * (18014398508138497 * (18014398508400641 *
    (72057594037616641 * 1))) : Nat) = QP := rfl

set_option maxRecDepth 100000 in
theorem setup0_eval (ka kb : ExactMatrix QP n 1 3) (pa pb : ExactMatrix Q n 1 1) :
    evalScope ctx simulator_0.setup (simulator_0.setup.scopes.length + 1) 0 (fun _ ↦ 0)
        [ToValue.encode ka, ToValue.encode kb, ToValue.encode pa, ToValue.encode pb] =
      pure [@ToValue.encode (ExactMatrix QP n 2 3) _ (rowsOf ka kb),
        @ToValue.encode (ExactMatrix Q n 2 1) _ (rowsOf pa pb)] := by
  rw [evalScope.eq_2]
  conv_lhs =>
    simp only [simulator_0, stepNode, List.foldlM_cons, List.foldlM_nil, List.mapM_cons,
      List.mapM_nil, lookup, List.getElem?_cons_zero, List.getElem?_cons_succ, mk_some, pure_bind,
      bind_assoc, nodeResult_input', Option.bind_some, Option.bind_eq_bind, Option.pure_def,
      List.nil_append, List.cons_append, List.append_assoc, nodeResult_concat, ToValue.encode,
      applyPure_concatRows, mk_some, pure_bind, List.length_cons, List.length_nil, Nat.reduceAdd, conformsAll_cons, conformsAll_nil, conforms_Q21, conforms_Q11, conforms_Q12,
      conforms_QP13, conforms_QP23, Bool.and_self, if_true]
  refine congrArg pure ?_
  simp only [ToValue.encode, List.cons.injEq, and_true, conformsAll_cons, conformsAll_nil, conforms_Q21, conforms_Q11, conforms_Q12,
      conforms_QP13, conforms_QP23, Bool.and_self, if_true]
  constructor <;> (refine congrArg _ ?_; funext i j; fin_cases i <;> rfl)

set_option maxRecDepth 100000 in
theorem setup1_eval (a b : ExactMatrix Q n 1 1) :
    evalScope ctx simulator_1.setup (simulator_1.setup.scopes.length + 1) 0 (fun _ ↦ 0)
        [ToValue.encode a, ToValue.encode b] =
      (do let ek ← (liftM (sampleMatrix (.residue QP) QP n 2 3) : OptionT Comp _)
          pure [ToValue.encode ek,
            @ToValue.encode (ExactMatrix Q n 2 1) _
              (rowsOf (matrixScale 1032193 a) (matrixScale 1032193 b))]) := by
  rw [evalScope.eq_2]
  conv_lhs =>
    simp only [
      simulator_1, stepNode, List.foldlM_cons, List.foldlM_nil, List.mapM_cons,
      List.mapM_nil, lookup, List.getElem?_cons_zero, List.getElem?_cons_succ, mk_some,
      pure_bind, bind_assoc, nodeResult_input', Option.bind_some, Option.bind_eq_bind,
      Option.pure_def, List.nil_append, List.cons_append, List.append_assoc,
      nodeResult_slice, nodeResult_matrixBinary, nodeResult_matrixScale, nodeResult_concat,
      nodeResult_uniformResidueSample, conformsAll_cons, conformsAll_nil, conforms_Q21,
      conforms_Q11, conforms_Q12, conforms_QP13, conforms_QP23, Bool.and_self, if_true,
      applySample, MatrixTy.eval, RingE.eval, IExpr.evalNat, IExpr.eval, sampleValue,
      applyPure_scale, applyPure_concatRows, Int.reduceLT, Int.reduceToNat, reduceIte,
      applyPure_sliceColumn0, applyPure_sliceColumn1, liftM_bind_pure, ToValue.encode,
      List.prod_cons, List.prod_nil, prodQ, prodQP, dimN, List.length_cons, List.length_nil,
      Nat.reduceAdd]
    dsimp only [List.prod_cons, List.prod_nil, prodQ, prodQP, dimN]
    simp only [
      simulator_1, stepNode, List.foldlM_cons, List.foldlM_nil, List.mapM_cons,
      List.mapM_nil, lookup, List.getElem?_cons_zero, List.getElem?_cons_succ, mk_some,
      pure_bind, bind_assoc, nodeResult_input', Option.bind_some, Option.bind_eq_bind,
      Option.pure_def, List.nil_append, List.cons_append, List.append_assoc,
      nodeResult_slice, nodeResult_matrixBinary, nodeResult_matrixScale, nodeResult_concat,
      nodeResult_uniformResidueSample, conformsAll_cons, conformsAll_nil, conforms_Q21,
      conforms_Q11, conforms_Q12, conforms_QP13, conforms_QP23, Bool.and_self, if_true,
      applySample, MatrixTy.eval, RingE.eval, IExpr.evalNat, IExpr.eval, sampleValue,
      applyPure_scale, applyPure_concatRows, Int.reduceLT, Int.reduceToNat, reduceIte,
      applyPure_sliceColumn0, applyPure_sliceColumn1, liftM_bind_pure, ToValue.encode,
      List.prod_cons, List.prod_nil, prodQ, prodQP, dimN, List.length_cons, List.length_nil,
      Nat.reduceAdd]
  refine bind_congr fun ek ↦ congrArg pure ?_
  simp only [ToValue.encode, List.cons.injEq, and_true, true_and, conformsAll_cons, conformsAll_nil, conforms_Q21, conforms_Q11, conforms_Q12,
      conforms_QP13, conforms_QP23, Bool.and_self, if_true]
  refine congrArg _ ?_
  funext i j
  fin_cases i <;> rfl

set_option maxRecDepth 100000 in
theorem setup2_eval (a b : ExactMatrix Q n 1 2) :
    evalScope ctx simulator_2.setup (simulator_2.setup.scopes.length + 1) 0 (fun _ ↦ 0)
        [ToValue.encode a, ToValue.encode b] =
      (do let ek ← (liftM (sampleMatrix (.residue QP) QP n 2 3) : OptionT Comp _)
          pure [ToValue.encode b, ToValue.encode ek, ToValue.encode (pkOfRow a)]) := by
  rw [evalScope.eq_2]
  conv_lhs =>
    simp only [
      simulator_2, stepNode, List.foldlM_cons, List.foldlM_nil, List.mapM_cons,
      List.mapM_nil, lookup, List.getElem?_cons_zero, List.getElem?_cons_succ, mk_some,
      pure_bind, bind_assoc, nodeResult_input', Option.bind_some, Option.bind_eq_bind,
      Option.pure_def, List.nil_append, List.cons_append, List.append_assoc,
      nodeResult_slice, nodeResult_matrixBinary, nodeResult_matrixScale, nodeResult_concat,
      nodeResult_uniformResidueSample, conformsAll_cons, conformsAll_nil, conforms_Q21,
      conforms_Q11, conforms_Q12, conforms_QP13, conforms_QP23, Bool.and_self, if_true,
      applySample, MatrixTy.eval, RingE.eval, IExpr.evalNat, IExpr.eval, sampleValue,
      applyPure_scale, applyPure_concatRows, Int.reduceLT, Int.reduceToNat, reduceIte,
      applyPure_sliceColumn0, applyPure_sliceColumn1, liftM_bind_pure, ToValue.encode,
      List.prod_cons, List.prod_nil, prodQ, prodQP, dimN, List.length_cons, List.length_nil,
      Nat.reduceAdd]
    dsimp only [List.prod_cons, List.prod_nil, prodQ, prodQP, dimN]
    simp only [
      simulator_2, stepNode, List.foldlM_cons, List.foldlM_nil, List.mapM_cons,
      List.mapM_nil, lookup, List.getElem?_cons_zero, List.getElem?_cons_succ, mk_some,
      pure_bind, bind_assoc, nodeResult_input', Option.bind_some, Option.bind_eq_bind,
      Option.pure_def, List.nil_append, List.cons_append, List.append_assoc,
      nodeResult_slice, nodeResult_matrixBinary, nodeResult_matrixScale, nodeResult_concat,
      nodeResult_uniformResidueSample, conformsAll_cons, conformsAll_nil, conforms_Q21,
      conforms_Q11, conforms_Q12, conforms_QP13, conforms_QP23, Bool.and_self, if_true,
      applySample, MatrixTy.eval, RingE.eval, IExpr.evalNat, IExpr.eval, sampleValue,
      applyPure_scale, applyPure_concatRows, Int.reduceLT, Int.reduceToNat, reduceIte,
      applyPure_sliceColumn0, applyPure_sliceColumn1, liftM_bind_pure, ToValue.encode,
      List.prod_cons, List.prod_nil, prodQ, prodQP, dimN, List.length_cons, List.length_nil,
      Nat.reduceAdd]
  refine bind_congr fun ek ↦ congrArg pure ?_
  simp only [ToValue.encode, List.cons.injEq, and_true, true_and, conformsAll_cons, conformsAll_nil, conforms_Q21, conforms_Q11, conforms_Q12,
      conforms_QP13, conforms_QP23, Bool.and_self, if_true]
  refine congrArg _ ?_
  funext i j
  fin_cases i <;> fin_cases j <;> simp [matrixScale, pkOfRow, tPoly]
set_option maxRecDepth 100000000 in
set_option maxHeartbeats 4000000 in
theorem encrypt2_eval (b : ExactMatrix Q n 1 2) (m : Fin n → Int) :
    evalScope ctx simulator_2.encrypt (simulator_2.encrypt.scopes.length + 1) 0 (fun _ ↦ 0)
        [ToValue.encode b, ToValue.encode m] =
      pure [ToValue.encode (ctOfRow b (encodeSlots m))] := by
  rw [evalScope.eq_2]
  conv_lhs =>
    simp only [
      simulator_2, stepNode, List.foldlM_cons, List.foldlM_nil, List.mapM_cons,
      List.mapM_nil, lookup, List.getElem?_cons_zero, List.getElem?_cons_succ, mk_some,
      pure_bind, bind_assoc, nodeResult_input', Option.bind_some, Option.bind_eq_bind,
      Option.pure_def, List.nil_append, List.cons_append, List.append_assoc,
      nodeResult_slice, nodeResult_matrixBinary, nodeResult_matrixScale, nodeResult_concat,
      nodeResult_constMatrix, nodeResult_polynomialValues, nodeResult_polynomialFromValues,
      nodeResult_centeredRebase, nodeResult_familyGetDynamic,
      nodeResult_uniformIntervalSample, nodeResult_gaussianSample,
      nodeResult_uniformResidueSample, conformsAll_cons, conformsAll_nil, conforms_Q21,
      conforms_Q11, conforms_t11, conforms_slots, conforms_int, Bool.and_self, if_true,
      applySample, MatrixTy.eval, RingE.eval, IExpr.evalNat, IExpr.eval, RExpr.evalRat,
      sampleValue, applyPure_add, applyPure_mul11, applyPure_scale, applyPure_concatRows,
      applyPure_polynomialValues, applyPure_constPacked, applyPure_centeredRebase,
      applyPure_fromValuesNtt, Int.reduceLT, Int.reduceToNat, reduceIte,
      applyPure_sliceRow0, applyPure_sliceRow1, applyPure_sliceColumn0, applyPure_sliceColumn1, conforms_Q12, liftM_bind_pure, ToValue.encode,
      List.prod_cons, List.prod_nil, prodQ, prodT, dimN, le_refl, List.length_cons,
      List.length_nil, Nat.reduceAdd]
    dsimp only [List.prod_cons, List.prod_nil, prodQ, prodT, dimN]
    simp only [
      simulator_2, stepNode, List.foldlM_cons, List.foldlM_nil, List.mapM_cons,
      List.mapM_nil, lookup, List.getElem?_cons_zero, List.getElem?_cons_succ, mk_some,
      pure_bind, bind_assoc, nodeResult_input', Option.bind_some, Option.bind_eq_bind,
      Option.pure_def, List.nil_append, List.cons_append, List.append_assoc,
      nodeResult_slice, nodeResult_matrixBinary, nodeResult_matrixScale, nodeResult_concat,
      nodeResult_constMatrix, nodeResult_polynomialValues, nodeResult_polynomialFromValues,
      nodeResult_centeredRebase, nodeResult_familyGetDynamic,
      nodeResult_uniformIntervalSample, nodeResult_gaussianSample,
      nodeResult_uniformResidueSample, conformsAll_cons, conformsAll_nil, conforms_Q21,
      conforms_Q11, conforms_t11, conforms_slots, conforms_int, Bool.and_self, if_true,
      applySample, MatrixTy.eval, RingE.eval, IExpr.evalNat, IExpr.eval, RExpr.evalRat,
      sampleValue, applyPure_add, applyPure_mul11, applyPure_scale, applyPure_concatRows,
      applyPure_polynomialValues, applyPure_constPacked, applyPure_centeredRebase,
      applyPure_fromValuesNtt, Int.reduceLT, Int.reduceToNat, reduceIte,
      applyPure_sliceRow0, applyPure_sliceRow1, applyPure_sliceColumn0, applyPure_sliceColumn1, conforms_Q12, liftM_bind_pure, ToValue.encode,
      List.prod_cons, List.prod_nil, prodQ, prodT, dimN, le_refl, List.length_cons,
      List.length_nil, Nat.reduceAdd]
  rw [nodeResult_broadcastZip (f := fun i ↦ Value.int (nativeOf m i))]
  rotate_left
  · simp [IExpr.evalNat, IExpr.eval]
  · intro i
    have hentry : (((Negacyclic.coeff (packedPolynomial (q := t) (n := n) 13 n encodeTable 0 0) i).val :
        Nat) : Int) = (packedEntry 13 encodeTable i.val : Nat) := tableValues_eq i
    simp only [encodeTable, Sim_2_encrypt.generatedRoot.table_6, Nat.cast_inj] at hentry
    rw [evalScope.eq_2]
    simp only [List.getElem?_cons_succ, List.getElem?_cons_zero, mk_some, pure_bind, bind_assoc,
      stepNode, List.foldlM_cons, List.foldlM_nil, List.mapM_cons, List.mapM_nil, lookup,
      nodeResult_input', Option.bind_some, nodeResult_familyGetDynamic, List.nil_append,
      List.cons_append, List.length_cons, List.length_nil, Nat.reduceAdd, Option.pure_def,
      Option.bind_eq_bind, applyPure_familyGetDynamic', conformsAll_cons, conformsAll_nil, conforms_Q21, conforms_Q11,
      conforms_t11, conforms_slots, conforms_int, Bool.and_self, if_true]
    rw [hentry]
    have hlt := encodeEntry_lt i
    simp only [encodeTable, Sim_2_encrypt.generatedRoot.table_6] at hlt
    simp only [List.getElem?_ofFn, hlt, dif_pos, Option.map_some, mk_some, pure_bind,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.bind_some, nativeOf, encodeTable,
      Sim_2_encrypt.generatedRoot.table_6, conformsAll_cons, conformsAll_nil, conforms_Q21, conforms_Q11,
      conforms_t11, conforms_slots, conforms_int, Bool.and_self, if_true]
  simp only [pure_bind, mk_some, List.getElem?_cons_zero, List.getElem?_cons_succ,
    List.cons_append, List.nil_append, List.append_assoc,
    Option.bind_some, Option.bind_eq_bind, Option.pure_def, bind_assoc, applyPure_slots,
    applyPure_rebaseQ, applyPure_add, applyPure_concatRows, conformsAll_cons, conformsAll_nil,
    conforms_Q21, conforms_Q21', conforms_Q11, conforms_Q12, conforms_t11, conforms_slots,
    conforms_int, Bool.and_self, Bool.and_true, if_true]
  refine congrArg pure ?_
  simp only [ToValue.encode, List.cons.injEq, and_true]
  refine congrArg (Value.matrix Q n 2 1) ?_
  funext i j
  fin_cases i <;> fin_cases j
  · simp [matrixScale, ctOfRow, tPoly]
  · simp [matrixScale, ctOfRow, tPoly, encodeSlots, ← irPlain_nativeOf]

/-- The two key simulators share their encryption program. -/
theorem sim1_encrypt_program : simulator_1.encrypt = simulator_0.encrypt := rfl

end Programs

/-! ## The reductions -/

/-- The public-key reduction, run as IR programs, is the generated one. -/
theorem reduce1 (A : MxxFhe.Adversary GeneratedScheme.scheme) (right : Bool)
    (a b : ExactMatrix Q n 1 1) :
    MxxFhe.reduceProgram ctx GeneratedScheme.codec
        (AssumptionId.challengeValues .decisionRlwe GeneratedScheme.params_1)
        GeneratedSimulators.simulator_1 A right (a, b) =
      MxxFhe.reduce GeneratedScheme.simulator_1 A right (a, b) := by
  unfold MxxFhe.reduceProgram MxxFhe.reduce
  have hin : GeneratedSimulators.simulator_1.setup.inputs = ["a", "b"] := rfl
  have hout : GeneratedSimulators.simulator_1.setup.outputs = ["evaluation_key", "public_key"] := rfl
  have ein : GeneratedSimulators.simulator_1.encrypt.inputs = ["message", "public_key"] := rfl
  have eout : GeneratedSimulators.simulator_1.encrypt.outputs = ["ciphertext"] := rfl
  simp only [Program.runNamed, hin, hout, ein, eout, AssumptionId.challengeValues, List.mapM_cons,
    List.mapM_nil, lookupNamed, List.find?, Program.run, Program.fuel, String.reduceEq,
    decide_true, decide_false, Option.map_some, Option.bind_eq_bind, Option.bind_some,
    Option.pure_def]
  rw [setup1_eval]
  simp only [run_liftM_bind, run_pure', bind_assoc, pure_bind, Option.map_some, List.zip_cons_cons,
    List.zip_nil_right, List.find?, String.reduceEq, decide_true, decide_false, Option.map_some,
    Option.bind_some, GeneratedScheme.codec, decode_encode_matrix]
  rw [sim1_encrypt_program]
  simp only [encrypt_eval, run_liftM_bind, run_pure', encryptRows_eq, bind_assoc, pure_bind,
    Option.map_some, List.zip_cons_cons, List.zip_nil_right, List.find?, decide_true,
    Option.bind_some, decode_encode_matrix]
  simp only [GeneratedScheme.simulator_1, sim1_setup_eq, sim1_encrypt_eq, bind_assoc, pure_bind,
    sampleResidue, sampleMatrix_clamp, mat11]

/-- The circular key-switching reduction, run as IR programs, is the generated one. -/
theorem reduce0 (A : MxxFhe.Adversary GeneratedScheme.scheme) (right : Bool)
    (pa pb : ExactMatrix Q n 1 1) (ka kb : ExactMatrix QP n 1 3) :
    MxxFhe.reduceProgram ctx GeneratedScheme.codec
        (AssumptionId.challengeValues .rlweKeySwitchCircular GeneratedScheme.params_0)
        GeneratedSimulators.simulator_0 A right ((pa, pb), (ka, kb)) =
      MxxFhe.reduce GeneratedScheme.simulator_0 A right ((pa, pb), (ka, kb)) := by
  unfold MxxFhe.reduceProgram MxxFhe.reduce
  have hin : GeneratedSimulators.simulator_0.setup.inputs = ["key_a", "key_b", "pk_a", "pk_b"] :=
    rfl
  have hout : GeneratedSimulators.simulator_0.setup.outputs = ["evaluation_key", "public_key"] := rfl
  have ein : GeneratedSimulators.simulator_0.encrypt.inputs = ["message", "public_key"] := rfl
  have eout : GeneratedSimulators.simulator_0.encrypt.outputs = ["ciphertext"] := rfl
  simp only [Program.runNamed, hin, hout, ein, eout, AssumptionId.challengeValues, List.mapM_cons,
    List.mapM_nil, lookupNamed, List.find?, Program.run, Program.fuel, String.reduceEq,
    decide_true, decide_false, Option.map_some, Option.bind_eq_bind, Option.bind_some,
    Option.pure_def]
  rw [setup0_eval]
  simp only [run_pure', bind_assoc, pure_bind, Option.map_some, List.zip_cons_cons,
    List.zip_nil_right, List.find?, String.reduceEq, decide_true, decide_false, Option.map_some,
    Option.bind_some, GeneratedScheme.codec, decode_encode_matrix]
  simp only [encrypt_eval, run_liftM_bind, run_pure', encryptRows_eq, bind_assoc, pure_bind,
    Option.map_some, List.zip_cons_cons, List.zip_nil_right, List.find?, decide_true,
    Option.bind_some, decode_encode_matrix]
  simp only [GeneratedScheme.simulator_0, sim0_setup_eq, sim0_encrypt_eq, bind_assoc, pure_bind,
    sampleMatrix_clamp, mat11]

/-- The ciphertext reduction, run as IR programs, is the generated one. -/
theorem reduce2 (A : MxxFhe.Adversary GeneratedScheme.scheme) (right : Bool)
    (a b : ExactMatrix Q n 1 2) :
    MxxFhe.reduceProgram ctx GeneratedScheme.codec
        (AssumptionId.challengeValues .decisionRlwe GeneratedScheme.params_2)
        GeneratedSimulators.simulator_2 A right (a, b) =
      MxxFhe.reduce GeneratedScheme.simulator_2 A right (a, b) := by
  unfold MxxFhe.reduceProgram MxxFhe.reduce
  have hin : GeneratedSimulators.simulator_2.setup.inputs = ["a", "b"] := rfl
  have hout : GeneratedSimulators.simulator_2.setup.outputs =
    ["b", "evaluation_key", "public_key"] := rfl
  have ein : GeneratedSimulators.simulator_2.encrypt.inputs = ["b", "message"] := rfl
  have eout : GeneratedSimulators.simulator_2.encrypt.outputs = ["ciphertext"] := rfl
  simp only [Program.runNamed, hin, hout, ein, eout, AssumptionId.challengeValues, List.mapM_cons,
    List.mapM_nil, lookupNamed, List.find?, Program.run, Program.fuel, String.reduceEq,
    decide_true, decide_false, Option.map_some, Option.bind_eq_bind, Option.bind_some,
    Option.pure_def]
  rw [setup2_eval]
  simp only [run_liftM_bind, run_pure', bind_assoc, pure_bind, Option.map_some, List.zip_cons_cons,
    List.zip_nil_right, List.find?, String.reduceEq, decide_true, decide_false, Option.map_some,
    Option.bind_some, GeneratedScheme.codec, decode_encode_matrix]
  simp only [encrypt2_eval, run_pure', pure_bind,
    Option.map_some, List.zip_cons_cons, List.zip_nil_right, List.find?, decide_true,
    Option.bind_some, decode_encode_matrix]
  simp only [GeneratedScheme.simulator_2, sim2_setup_eq, sim2_encrypt_eq, bind_assoc, pure_bind,
    sampleResidue]

theorem reduceProgram_eq0 (A : MxxFhe.Adversary GeneratedScheme.scheme) (right : Bool) :
    MxxFhe.reduceProgram ctx GeneratedScheme.codec
        (AssumptionId.challengeValues .rlweKeySwitchCircular GeneratedScheme.params_0)
        GeneratedSimulators.simulator_0 A right =
      MxxFhe.reduce GeneratedScheme.simulator_0 A right := by
  funext ch
  obtain ⟨⟨pa, pb⟩, ⟨ka, kb⟩⟩ := ch
  exact reduce0 A right pa pb ka kb

theorem reduceProgram_eq1 (A : MxxFhe.Adversary GeneratedScheme.scheme) (right : Bool) :
    MxxFhe.reduceProgram ctx GeneratedScheme.codec
        (AssumptionId.challengeValues .decisionRlwe GeneratedScheme.params_1)
        GeneratedSimulators.simulator_1 A right =
      MxxFhe.reduce GeneratedScheme.simulator_1 A right := by
  funext ch
  obtain ⟨a, b⟩ := ch
  exact reduce1 A right a b

theorem reduceProgram_eq2 (A : MxxFhe.Adversary GeneratedScheme.scheme) (right : Bool) :
    MxxFhe.reduceProgram ctx GeneratedScheme.codec
        (AssumptionId.challengeValues .decisionRlwe GeneratedScheme.params_2)
        GeneratedSimulators.simulator_2 A right =
      MxxFhe.reduce GeneratedScheme.simulator_2 A right := by
  funext ch
  obtain ⟨a, b⟩ := ch
  exact reduce2 A right a b

/-- IND-CPA security with reductions whose simulators are the IR programs of
`Simulators.lean`, each of cost at most `2^30`. -/
theorem bounded_security : GeneratedScheme.BoundedSecurityClaim := by
  intro backend
  refine ⟨![GeneratedSimulators.simulator_0, GeneratedSimulators.simulator_1,
    GeneratedSimulators.simulator_2], fun i ↦ ?_, fun hash A ε h ↦ security A ε fun i right ↦ ?_⟩
  · fin_cases i
    · exact ⟨_, BgvBounded.simulator_0_cost, by decide⟩
    · exact ⟨_, BgvBounded.simulator_1_cost, by decide⟩
    · exact ⟨_, BgvBounded.simulator_2_cost, by decide⟩
  · fin_cases i
    · have h0 := h ⟨0, by decide⟩ right
      change AssumptionId.advantage .rlweKeySwitchCircular GeneratedScheme.params_0
          (MxxFhe.reduceProgram ⟨hash, backend⟩ GeneratedScheme.codec
            (AssumptionId.challengeValues .rlweKeySwitchCircular GeneratedScheme.params_0)
            GeneratedSimulators.simulator_0 A right) ≤ _ at h0
      rw [reduceProgram_eq0] at h0
      exact h0
    · have h1 := h ⟨1, by decide⟩ right
      change AssumptionId.advantage .decisionRlwe GeneratedScheme.params_1
          (MxxFhe.reduceProgram ⟨hash, backend⟩ GeneratedScheme.codec
            (AssumptionId.challengeValues .decisionRlwe GeneratedScheme.params_1)
            GeneratedSimulators.simulator_1 A right) ≤ _ at h1
      rw [reduceProgram_eq1] at h1
      exact h1
    · have h2 := h ⟨2, by decide⟩ right
      change AssumptionId.advantage .decisionRlwe GeneratedScheme.params_2
          (MxxFhe.reduceProgram ⟨hash, backend⟩ GeneratedScheme.codec
            (AssumptionId.challengeValues .decisionRlwe GeneratedScheme.params_2)
            GeneratedSimulators.simulator_2 A right) ≤ _ at h2
      rw [reduceProgram_eq2] at h2
      exact h2

end BgvSecurity
