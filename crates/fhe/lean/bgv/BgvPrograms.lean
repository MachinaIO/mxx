import BgvNtt
import BgvForms
import BgvEncryptForms
import Simulators

set_option backward.isDefEq.respectTransparency false

/-! The simulators of `Simulators.lean`, run as IR programs, compute what the generated simulators
of `Scheme.lean` compute. The slot encoding of the encryption simulators is the inverse transform
`MxxSecurity.IR.inverseNtt`; `inverseNtt_spec` shows that it is the runtime's
`polynomialFromValues`, so it is the plaintext `plainOf` the closed forms use. -/

namespace BgvSecurity
open MxxSecurity MxxSecurity.IR Mxx.Primitives MxxRuntime GeneratedSimulators

variable {ctx : Context}

/-- The native root of the plaintext ring in `ZMod t`. -/
noncomputable abbrev ω : ZMod t := ((nativeNttRoot t n : Nat) : ZMod t)

theorem ω_ne_zero : ω ≠ 0 := native_root_primitive.ne_zero (by decide)

theorem ω_pow_n : ω ^ n = -1 := by
  have h2 : IsPrimitiveRoot (ω ^ n) 2 := by
    have := native_root_primitive.pow_of_dvd (p := n) (by decide) (dvd_mul_left n 2)
    simpa using this
  exact h2.eq_neg_one_of_two_right

/-- The ratio of two odd powers sums to `n` over a period when they agree, and to `0`
otherwise. -/
theorem odd_ratio_sum (i j : Fin n) :
    ∑ k : Fin n, ((ω⁻¹) ^ (2 * i.val + 1) * ω ^ (2 * j.val + 1)) ^ k.val =
      if i = j then (n : ZMod t) else 0 := by
  set μ := (ω⁻¹) ^ (2 * i.val + 1) * ω ^ (2 * j.val + 1)
  rw [Fin.sum_univ_eq_sum_range (fun k ↦ μ ^ k)]
  by_cases hij : i = j
  · subst hij
    have : μ = 1 := by
      simp only [μ, inv_pow, inv_mul_cancel₀ (pow_ne_zero _ ω_ne_zero)]
    simp [this]
  · rw [ite_eq_right_iff.mpr (fun h ↦ absurd h hij)]
    have hμn : μ ^ n = 1 := by
      have hodd (a : Nat) : ω ^ ((2 * a + 1) * n) = -1 := by
        rw [mul_comm, pow_mul, ω_pow_n, pow_succ, pow_mul]
        norm_num
      simp only [μ, mul_pow, inv_pow, ← pow_mul, hodd]
      norm_num
    have hμ : μ ≠ 1 := by
      intro h
      have heq : ω ^ (2 * j.val + 1) = ω ^ (2 * i.val + 1) := by
        simp only [μ, inv_pow] at h
        exact ((inv_mul_eq_one₀ (pow_ne_zero _ ω_ne_zero)).mp h).symm
      have := native_root_primitive.pow_inj (by omega) (by omega) heq
      exact hij (Fin.ext (by omega))
    rw [geom_sum_eq hμ, hμn, sub_self, zero_div]

/-- The root of the transform over the prime `t` is its native root. -/
theorem nttRoot_t : nttRoot t n = nativeNttRoot t n := by
  have hp : Nat.primeFactors t = {t} := Nat.Prime.primeFactors Fact.out
  unfold nttRoot
  rw [hp, Finset.sum_singleton, Nat.div_self (by decide), Nat.cast_one, inv_one, ZMod.val_one,
    mul_one, mul_one, Nat.mod_eq_of_lt native_lt]

/-- `inverseNtt` inverts the odd-power evaluation: evaluation `j` of the coefficients it returns
is the value at slot `nttSlot 13 j`. -/
theorem oddEval_inverseNtt (v : Fin n → ZMod t) :
    oddEval ω (inverseNtt t 13 v) = fun j ↦ v (nttSlot 13 j) := by
  funext j
  simp only [oddEval, inverseNtt, nttRoot_t]
  have hn : (n : ZMod t) ≠ 0 := by
    rw [Ne, ZMod.natCast_eq_zero_iff]
    exact fun h ↦ absurd (Nat.le_of_dvd (by decide) h) (by decide)
  calc ∑ k : Fin n, ((n : ZMod t)⁻¹ * ∑ i : Fin n, v (nttSlot 13 i) *
          (ω⁻¹) ^ ((2 * i.val + 1) * k.val)) * ω ^ ((2 * j.val + 1) * k.val)
      = (n : ZMod t)⁻¹ * ∑ i : Fin n, v (nttSlot 13 i) *
          ∑ k : Fin n, ((ω⁻¹) ^ (2 * i.val + 1) * ω ^ (2 * j.val + 1)) ^ k.val := by
        rw [Finset.mul_sum]
        simp only [Finset.mul_sum, Finset.sum_mul]
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun i _ ↦ Finset.sum_congr rfl fun k _ ↦ ?_
        rw [mul_pow, ← pow_mul, ← pow_mul, mul_comm (2 * j.val + 1)]
        ring
    _ = v (nttSlot 13 j) := by
        simp only [odd_ratio_sum, mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
        field_simp

theorem nttSlot_eq (j : Fin n) : nttSlot 13 j = ⟨nttBitReverse 13 j.val, bitReverse_lt _ j.isLt⟩ := by
  simp only [nttSlot, dif_pos (bitReverse_lt _ j.isLt)]

/-- The plaintext the IR slot encoding returns for native evaluations `input`. -/
noncomputable def irPlain (input : Fin n → Int) : ExactMatrix t n 1 1 :=
  fun _ _ ↦ polynomialOfCoefficients fun k ↦
    ((inverseNtt t 13 fun s ↦ (input s : ZMod t)) k).val

set_option maxRecDepth 100000 in
/-- The IR slot encoding is the runtime's inverse transform. -/
theorem irPlain_spec (input : Fin n → Int) : polynomialFromValues true input (irPlain input) := by
  set c := inverseNtt t 13 fun s ↦ (input s : ZMod t)
  have hc : oddEval ω c = fun index : Fin n ↦
      ((input ⟨nttBitReverse 13 index.val, bitReverse_lt _ index.isLt⟩ : Int) : ZMod t) := by
    rw [oddEval_inverseNtt]
    funext j
    rw [nttSlot_eq]
  have hcoeff (k : Fin n) : ((polynomialOfCoefficients (q := t) fun k ↦ ((c k).val : Int)).coeff k)
      = c k := by
    rw [polynomialOfCoefficients_coeff (by decide) (by decide), zmod_val_cast]
  show polynomialNttRuns t true input _
  refine ⟨by decide, fun prime hprime hdvd ↦ ?_, 13, nativeNttRoot t n, by norm_num, native_lt,
    fun prime hprime hdvd ↦ ?_, fun i ↦ ⟨by positivity,
      Int.ofNat_lt.mpr (ZMod.val_lt _)⟩, ?_⟩
  · have hp : prime ∣ t := dvd_trans (dvd_mul_right prime prime) hdvd
    have hpt := (Nat.prime_dvd_prime_iff_eq hprime (Fact.out : t.Prime)).mp hp
    subst hpt
    exact absurd (Nat.le_of_dvd (by decide) hdvd) (by decide)
  · have hpt := (Nat.prime_dvd_prime_iff_eq hprime (Fact.out : t.Prime)).mp hdvd
    subst hpt
    exact ⟨by decide, by rw [Nat.mod_eq_of_lt native_lt]; exact native_primitive,
      Nat.mod_eq_of_lt native_lt⟩
  · intro index
    refine ⟨⟨nttBitReverse 13 index.val, bitReverse_lt _ index.isLt⟩, rfl, ?_⟩
    rw [← ZMod.intCast_eq_intCast_iff']
    have hi := congrFun hc index
    simp only [oddEval] at hi
    rw [← hi]
    push_cast
    refine Finset.sum_congr rfl fun k _ ↦ ?_
    have hplain : irPlain input 0 0 = polynomialOfCoefficients (q := t) fun k ↦ ((c k).val : Int) :=
      rfl
    rw [hplain, hcoeff k]
    simp only [ZMod.natCast_val, ZMod.cast_id', id]

/-- The IR slot encoding of `m` is `plainOf m`. -/
theorem irPlain_nativeOf (m : Fin n → Int) : irPlain (nativeOf m) = plainOf m :=
  fromValues_unique (irPlain_spec _) (plainOf_spec m)

/-! ## The encryption program -/

theorem prodQ : (18014398507892737 * (18014398508138497 * (18014398508400641 * 1)) : Nat) = Q :=
  rfl
theorem prodT : (1032193 * 1 : Nat) = t := rfl
theorem dimN : (8192 : Nat) = n := rfl

theorem log2_n : Nat.log2 n = 13 := by
  rw [show n = 2 ^ 13 from rfl, Nat.log2_two_pow]

theorem nttBits_t : nttBits? t n = some 13 := by
  unfold nttBits?
  have hp : Nat.primeFactors t = {t} := Nat.Prime.primeFactors Fact.out
  rw [log2_n, if_pos ⟨by decide, by rw [Nat.primeFactorsList_prime Fact.out]; simp,
    by norm_num, by rw [hp]; simp; norm_num⟩]

theorem one_lt_Q : 1 < Q := by norm_num

/-! The values of the simulator programs have the types their nodes declare. -/

theorem conforms_Q21 (env : LoopEnv) (a : ExactMatrix Q n 2 1) :
    Value.conforms env (.matrix ⟨.explicit [.const 18014398507892737, .const 18014398508138497, .const 18014398508400641] n, .const 2, .const 1⟩)
      (.matrix Q n 2 1 a) = true := rfl

theorem conforms_Q11 (env : LoopEnv) (a : ExactMatrix Q n 1 1) :
    Value.conforms env (.matrix ⟨.explicit [.const 18014398507892737, .const 18014398508138497, .const 18014398508400641] n, .const 1, .const 1⟩)
      (.matrix Q n 1 1 a) = true := rfl

theorem conforms_Q12 (env : LoopEnv) (a : ExactMatrix Q n 1 2) :
    Value.conforms env (.matrix ⟨.explicit [.const 18014398507892737, .const 18014398508138497, .const 18014398508400641] n, .const 1, .const 2⟩)
      (.matrix Q n 1 2 a) = true := rfl

theorem conforms_QP13 (env : LoopEnv) (a : ExactMatrix QP n 1 3) :
    Value.conforms env (.matrix ⟨.explicit [.const 18014398507892737, .const 18014398508138497, .const 18014398508400641,
      .const 72057594037616641] n, .const 1, .const 3⟩)
      (.matrix QP n 1 3 a) = true := rfl

theorem conforms_QP23 (env : LoopEnv) (a : ExactMatrix QP n 2 3) :
    Value.conforms env (.matrix ⟨.explicit [.const 18014398507892737, .const 18014398508138497, .const 18014398508400641,
      .const 72057594037616641] n, .const 2, .const 3⟩)
      (.matrix QP n 2 3 a) = true := rfl

theorem conforms_Q21' (env : LoopEnv) (a : ExactMatrix Q n (1 + 1) 1) :
    Value.conforms env (.matrix ⟨.explicit [.const 18014398507892737, .const 18014398508138497,
      .const 18014398508400641] n, .const 2, .const 1⟩) (.matrix Q n (1 + 1) 1 a) = true := rfl

theorem conforms_t11 (env : LoopEnv) (a : ExactMatrix t n 1 1) :
    Value.conforms env (.matrix ⟨.explicit [.const 1032193] n, .const 1, .const 1⟩)
      (.matrix t n 1 1 a) = true := rfl

theorem conforms_slots (env : LoopEnv) (f : Fin n → Int) :
    Value.conforms env (.family .int (.const 8192)) (.family (List.ofFn fun i ↦ .int (f i))) =
      true :=
  conforms_family_ofFn_int env _ f rfl

/-- The slot-encoding node: the inverse transform over the plaintext ring. -/
theorem applyPure_slots (env : LoopEnv) (v : Fin n → Int) :
    applyPure ctx env (.polynomialFromValues ⟨.explicit [.const 1032193] n, .const 1, .const 1⟩ true)
      [.family (List.ofFn fun i ↦ .int (v i))] = some [.matrix t n 1 1 (irPlain v)] := by
  have hring : RingE.eval env (.explicit [.const 1032193] n) = some ([t], n) := rfl
  have hbits : nttBits? [t].prod n = some 13 := by
    rw [show [t].prod = t from Nat.mul_one t]; exact nttBits_t
  rw [applyPure_fromValuesNtt_of env _ v hring hbits]
  rfl

/-- The lift of the plaintext to the ciphertext ring `R_Q`. -/
theorem applyPure_rebaseQ (env : LoopEnv) (a : ExactMatrix t n 1 1) :
    applyPure ctx env (.centeredRebase (.explicit [.const 18014398507892737, .const 18014398508138497,
      .const 18014398508400641] n)) [.matrix t n 1 1 a] =
      some [.matrix Q n 1 1 (centeredRebase a)] := by
  rw [applyPure_centeredRebase]
  simp only [RingE.eval, IExpr.evalNat, IExpr.eval, Int.reduceLT, Int.reduceToNat, reduceIte,
    List.mapM_cons, List.mapM_nil, Option.pure_def, Option.bind_eq_bind, Option.bind_some]
  dsimp only [List.prod_cons, List.prod_nil, prodQ]
  rw [if_pos ⟨one_lt_Q, trivial⟩]

/-- The rows the encryption program stacks: `pk₀ U + t e₁` and `pk₁ U + t e₂ + E`. -/
noncomputable def encryptRows (pk : ExactMatrix Q n 2 1) (U e1 e2 E : ExactMatrix Q n 1 1) :
    ExactMatrix Q n (1 + 1) 1 := fun i j ↦
  let top : ExactMatrix Q n 1 1 := fun i j ↦ pk 0 0 * U i j
  let bottom : ExactMatrix Q n 1 1 := fun i j ↦ pk 1 0 * U i j
  if h : i.val < 1 then (top + matrixScale 1032193 e1) ⟨i.val, h⟩ j
  else (bottom + matrixScale 1032193 e2 + E) ⟨i.val - 1, by omega⟩ j

set_option maxRecDepth 100000000 in
set_option maxHeartbeats 4000000 in
/-- The encryption program, node by node: three draws, then the rows. -/
theorem encrypt_eval (pk : ExactMatrix Q n 2 1) (m : Fin n → Int) :
    evalScope ctx simulator_0.encrypt (simulator_0.encrypt.scopes.length + 1) 0 (fun _ ↦ 0)
        [ToValue.encode m, ToValue.encode pk] =
      (do let U ← (liftM (sampleMatrix ternLaw Q n 1 1) : OptionT Comp _)
          let e1 ← (liftM (sampleMatrix errLaw Q n 1 1) : OptionT Comp _)
          let e2 ← (liftM (sampleMatrix errLaw Q n 1 1) : OptionT Comp _)
          pure [Value.matrix Q n (1 + 1) 1
            (encryptRows pk U e1 e2 (centeredRebase (irPlain (nativeOf m))))]) := by
  rw [evalScope.eq_2]
  conv_lhs =>
    simp only [
      simulator_0, stepNode, List.foldlM_cons, List.foldlM_nil, List.mapM_cons,
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
      applyPure_sliceRow0, applyPure_sliceRow1, liftM_bind_pure, ToValue.encode,
      List.prod_cons, List.prod_nil, prodQ, prodT, dimN, le_refl, List.length_cons,
      List.length_nil, Nat.reduceAdd]
    dsimp only [List.prod_cons, List.prod_nil, prodQ, prodT, dimN]
    simp only [
      simulator_0, stepNode, List.foldlM_cons, List.foldlM_nil, List.mapM_cons,
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
      applyPure_sliceRow0, applyPure_sliceRow1, liftM_bind_pure, ToValue.encode,
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
    conforms_Q21, conforms_Q11, conforms_t11, conforms_slots, conforms_int, Bool.and_self, if_true]
  rfl

theorem encryptRows_eq (pk : ExactMatrix Q n 2 1) (U e1 e2 : ExactMatrix Q n 1 1) (m : Fin n → Int) :
    Value.matrix Q n (1 + 1) 1 (encryptRows pk U e1 e2 (centeredRebase (irPlain (nativeOf m)))) =
      ToValue.encode (encOf pk U e1 e2 (encodeSlots m)) := by
  refine congrArg (Value.matrix Q n 2 1) ?_
  funext i j
  fin_cases i <;> fin_cases j
  · simp [encryptRows, encOf, matrixScale, tPoly]
  · simp [encryptRows, encOf, matrixScale, tPoly, encodeSlots, ← irPlain_nativeOf]

/-- The encryption program is BGV encryption of `m` under `pk`. -/
theorem encrypt_run (pk : ExactMatrix Q n 2 1) (m : Fin n → Int) :
    simulator_0.encrypt.run ctx [ToValue.encode m, ToValue.encode pk] =
      (do let U ← sampleMatrix ternLaw Q n 1 1
          let e1 ← sampleMatrix errLaw Q n 1 1
          let e2 ← sampleMatrix errLaw Q n 1 1
          pure (some [ToValue.encode (encOf pk U e1 e2 (encodeSlots m))])) := by
  unfold Program.run Program.fuel
  rw [encrypt_eval]
  simp only [run_liftM_bind, run_pure', encryptRows_eq]

end BgvSecurity
