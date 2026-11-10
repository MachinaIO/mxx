import MxxSecurity.IR.Codec

/-!
# Evaluation equations of IR programs

Equations that run a program node by node without unfolding the values it computes: each
operation's equation has arbitrary matrices or families as arguments and returns its result as
an expression of them. A proof about a concrete program unfolds the scope fold with
`evalScope_succ` and `stepNode_eq`, then rewrites each node with these equations, so large
payloads, such as packed tables or the coefficients of a polynomial, stay folded.
-/

namespace MxxSecurity.IR

open Mxx.Primitives MxxRuntime

variable {ctx : Context}

theorem mk_some {α : Type} (a : α) :
    (OptionT.mk (pure (some a) : Comp (Option α))) = (pure a : OptionT Comp α) := rfl

/-- A sampler's draw lifted into the evaluation monad, with the drawn value exposed. -/
theorem liftM_bind_pure {α β : Type} (c : Comp α) (f : α → β) :
    (liftM (c >>= fun a ↦ pure (f a)) : OptionT Comp β) =
      (do let a ← (liftM c : OptionT Comp α); pure (f a)) := by
  apply OptionT.ext
  simp [liftM, monadLift, MonadLift.monadLift, OptionT.lift, OptionT.mk]
  show ((fun a ↦ some (f a)) <$> c) = Option.map f <$> (some <$> c)
  rw [Functor.map_map]
  rfl

theorem run_liftM_bind {α β : Type} (c : Comp α) (f : α → OptionT Comp β) :
    ((liftM c : OptionT Comp α) >>= f).run = c >>= fun a ↦ (f a).run := by
  simp only [OptionT.run_bind, liftM, monadLift, MonadLift.monadLift, OptionT.lift, OptionT.mk,
    Option.elimM]
  show (c >>= fun a ↦ pure (some a)) >>= (fun o ↦ o.elim (pure none) fun x ↦ (f x).run) = _
  simp [bind_assoc]

theorem run_pure' {α : Type} (a : α) : (pure a : OptionT Comp α).run = pure (some a) := rfl

theorem mk_none {α : Type} : (OptionT.mk (pure none : Comp (Option α))) = OptionT.fail := rfl

/-- A node whose arguments resolve and whose operation returns `result`. -/
theorem stepNode_eq {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {outputs : List (List Value)} {node : Node}
    {args result : List Value} (hargs : node.args.mapM (lookup outputs) = some args)
    (hresult : nodeResult ctx scopeEval env inputs node.op node.outputs.length args = pure result)
    (htypes : Value.conformsAll env node.outputs result = true) :
    stepNode ctx scopeEval env inputs outputs node = pure (outputs ++ [result]) := by
  simp only [stepNode, hargs, mk_some, pure_bind, hresult, htypes, if_true]

theorem stepNode_sample {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {outputs : List (List Value)} {node : Node}
    {args : List Value} {draw : Comp Value} (hargs : node.args.mapM (lookup outputs) = some args)
    (hstructural : node.op.structural = false) (hdraw : applySample ctx env node.op = some draw) :
    stepNode ctx scopeEval env inputs outputs node =
      (do let v ← (draw : OptionT Comp Value)
          if Value.conformsAll env node.outputs [v] then pure (outputs ++ [[v]])
          else OptionT.fail) := by
  rcases node with ⟨op, nodeArgs, nodeOutputs⟩
  simp only [stepNode, hargs, mk_some, pure_bind] at hdraw ⊢
  cases op <;> simp_all [nodeResult, Op.structural]

/-! ## Output types -/

theorem conformsAll_nil (env : LoopEnv) : Value.conformsAll env [] [] = true := rfl

theorem conformsAll_cons (env : LoopEnv) (ty : Ty) (types : List Ty) (value : Value)
    (values : List Value) :
    Value.conformsAll env (ty :: types) (value :: values) =
      (value.conforms env ty && Value.conformsAll env types values) := by
  simp only [Value.conformsAll, List.length_cons, List.zip_cons_cons, List.all_cons,
    Nat.succ.injEq, beq_iff_eq]
  by_cases h : types.length = values.length <;> simp [h, Bool.and_comm, Bool.and_left_comm]

theorem conforms_int (env : LoopEnv) (x : Int) : Value.conforms env .int (.int x) = true := rfl

theorem conforms_matrix (env : LoopEnv) (type : MatrixTy) {q n r c : Nat}
    (a : ExactMatrix q n r c) :
    Value.conforms env (.matrix type) (.matrix q n r c a) =
      match type.eval env with
      | some (m, n', r', c') => decide (m.prod = q ∧ n' = n ∧ r' = r ∧ c' = c)
      | none => false := rfl

/-- A family of `k` integers has type `family int count` when `count` is `k`; the family stays
abstract, so large families are never unfolded. -/
theorem conforms_family_ofFn_int (env : LoopEnv) (count : IExpr) {k : Nat} (f : Fin k → Int)
    (h : count.evalNat env = some k) :
    Value.conforms env (.family .int count) (.family (List.ofFn fun i ↦ .int (f i))) = true := by
  simp [Value.conforms, h, List.all_eq_true, List.mem_ofFn]

/-- A node computing from its arguments. -/
theorem nodeResult_pure {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {op : Op} {count : Nat} {args result : List Value}
    (hstructural : op.structural = false) (hsample : applySample ctx env op = none)
    (hresult : applyPure ctx env op args = some result) :
    nodeResult ctx scopeEval env inputs op count args = pure result := by
  cases op <;> simp_all [nodeResult, Op.structural, mk_some]

theorem nodeResult_input {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {position count : Nat} {value : Value}
    (h : inputs[position]? = some value) :
    nodeResult ctx scopeEval env inputs (.input position) count [] = pure [value] := by
  simp [nodeResult, h, mk_some]

theorem nodeResult_input' {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {position count : Nat} {args : List Value} :
    nodeResult ctx scopeEval env inputs (.input position) count args =
      OptionT.mk (pure (inputs[position]?.bind fun value ↦ some [value])) := rfl

/-- A trapdoor draw returns the public matrix and the trapdoor. -/
theorem nodeResult_trapdoorSample
    {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {count : Nat} {args : List Value} {type : MatrixTy}
    {sigma : RExpr} {base digits bound : IExpr} {statisticalRows : Option Nat}
    {preimageSigma : Option RExpr} :
    nodeResult ctx scopeEval env inputs
        (.trapdoorSample type sigma base digits bound statisticalRows preimageSigma) count args =
      match trapdoorDraw ctx env type sigma base digits statisticalRows preimageSigma with
      | some draw => liftM draw
      | none => OptionT.fail := rfl

/-- A node that neither runs a scope nor reads an input: a sampler draws, any other operation
computes from its arguments. -/
theorem nodeResult_nonstructural
    {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {op : Op} {count : Nat} {args : List Value}
    (hstructural : op.structural = false) :
    nodeResult ctx scopeEval env inputs op count args =
      match applySample ctx env op with
      | some draw => do pure [← (draw : OptionT Comp Value)]
      | none => OptionT.mk (pure (applyPure ctx env op args)) := by
  cases op <;> first | rfl | (simp [Op.structural] at hstructural)

/-! ## Nodes by operation

`nodeResult` on each operation, without a side condition, so `simp` can rewrite a node of a
concrete program. -/

theorem nodeResult_slice {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {rows columns : Option (IExpr × IExpr)} {count : Nat} {args : List Value} :
    nodeResult ctx scopeEval env inputs (.slice rows columns) count args = OptionT.mk (pure (applyPure ctx env (.slice rows columns) args)) :=
  rfl

theorem nodeResult_matrixBinary {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {op : MatrixOp} {count : Nat} {args : List Value} :
    nodeResult ctx scopeEval env inputs (.matrixBinary op) count args = OptionT.mk (pure (applyPure ctx env (.matrixBinary op) args)) :=
  rfl

theorem nodeResult_matrixScale {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {scalar : IExpr} {count : Nat} {args : List Value} :
    nodeResult ctx scopeEval env inputs (.matrixScale scalar) count args = OptionT.mk (pure (applyPure ctx env (.matrixScale scalar) args)) :=
  rfl

theorem nodeResult_concat {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {axis : Axis} {count : Nat} {args : List Value} :
    nodeResult ctx scopeEval env inputs (.concat axis) count args = OptionT.mk (pure (applyPure ctx env (.concat axis) args)) :=
  rfl

theorem nodeResult_constMatrix {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {type : MatrixTy} {value : ConstMatrix} {count : Nat} {args : List Value} :
    nodeResult ctx scopeEval env inputs (.constMatrix type value) count args = OptionT.mk (pure (applyPure ctx env (.constMatrix type value) args)) :=
  rfl

theorem nodeResult_polynomialValues {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {evaluation : Bool} {count : Nat} {args : List Value} :
    nodeResult ctx scopeEval env inputs (.polynomialValues evaluation) count args = OptionT.mk (pure (applyPure ctx env (.polynomialValues evaluation) args)) :=
  rfl

theorem nodeResult_polynomialFromValues {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {type : MatrixTy} {evaluation : Bool} {count : Nat} {args : List Value} :
    nodeResult ctx scopeEval env inputs (.polynomialFromValues type evaluation) count args = OptionT.mk (pure (applyPure ctx env (.polynomialFromValues type evaluation) args)) :=
  rfl

theorem nodeResult_centeredRebase {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {ring : RingE} {count : Nat} {args : List Value} :
    nodeResult ctx scopeEval env inputs (.centeredRebase ring) count args = OptionT.mk (pure (applyPure ctx env (.centeredRebase ring) args)) :=
  rfl

theorem nodeResult_familyGetDynamic {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value}  {count : Nat} {args : List Value} :
    nodeResult ctx scopeEval env inputs .familyGetDynamic count args = OptionT.mk (pure (applyPure ctx env .familyGetDynamic args)) :=
  rfl

theorem nodeResult_uniformIntervalSample {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {type : MatrixTy} {minimum maximum : IExpr} {count : Nat} {args : List Value} :
    nodeResult ctx scopeEval env inputs (.uniformIntervalSample type minimum maximum) count args =
      match applySample ctx env (.uniformIntervalSample type minimum maximum) with
      | some draw => do pure [← (draw : OptionT Comp Value)]
      | none => OptionT.mk (pure none) :=
  rfl

theorem nodeResult_gaussianSample {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {type : MatrixTy} {sigma : RExpr} {bound : IExpr} {compact : Bool} {count : Nat} {args : List Value} :
    nodeResult ctx scopeEval env inputs (.gaussianSample type sigma bound compact) count args =
      match applySample ctx env (.gaussianSample type sigma bound compact) with
      | some draw => do pure [← (draw : OptionT Comp Value)]
      | none => OptionT.mk (pure none) :=
  rfl

theorem nodeResult_uniformResidueSample {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {type : MatrixTy} {count : Nat} {args : List Value} :
    nodeResult ctx scopeEval env inputs (.uniformResidueSample type) count args =
      match applySample ctx env (.uniformResidueSample type) with
      | some draw => do pure [← (draw : OptionT Comp Value)]
      | none => OptionT.mk (pure none) :=
  rfl

/-! ## Operations on arbitrary payloads -/

/-- A packed constant polynomial, as the packed table it is. -/
theorem applyPure_constPacked (env : LoopEnv) (ring : RingE) (width count table : Nat) :
    applyPure ctx env (.constMatrix ⟨ring, .const 1, .const 1⟩ (.packed width count table)) [] =
      (ring.eval env).bind fun (moduli, n) ↦
        if count ≤ n then some [.matrix moduli.prod n 1 1 (packedPolynomial width count table)]
        else none := by
  simp only [applyPure, constMatrixValue, MatrixTy.eval, IExpr.evalNat, IExpr.eval]
  cases ring.eval env with
  | none => rfl
  | some pair =>
    obtain ⟨moduli, n⟩ := pair
    by_cases h : count ≤ n
    · simp [h]
      rfl
    · simp [h]

theorem applyPure_centeredRebase (env : LoopEnv) (ring : RingE) {q n r c : Nat}
    (a : ExactMatrix q n r c) :
    applyPure ctx env (.centeredRebase ring) [.matrix q n r c a] =
      (ring.eval env).bind fun (moduli, n') ↦
        if 1 < moduli.prod ∧ n' = n then some [.matrix moduli.prod n r c (centeredRebase a)]
        else none := rfl

/-- The inverse transform of an encoded family over a single-prime ring. -/
theorem applyPure_fromValuesNtt (env : LoopEnv) (ring : RingE) {k : Nat} (v : Fin k → Int) :
    applyPure ctx env (.polynomialFromValues ⟨ring, .const 1, .const 1⟩ true)
        [.family (List.ofFn fun i ↦ .int (v i))] =
      (ring.eval env).bind fun (moduli, n) ↦
        (intValues? n (List.ofFn fun i ↦ Value.int (v i))).bind fun values ↦
          (nttBits? moduli.prod n).bind fun bits ↦
            some [.matrix moduli.prod n 1 1 fun _ _ ↦ polynomialOfCoefficients fun j ↦
              ((inverseNtt moduli.prod bits fun s ↦ (values s : ZMod moduli.prod)) j).val] := by
  simp only [applyPure, MatrixTy.eval, IExpr.evalNat, IExpr.eval]
  cases ring.eval env with
  | none => rfl
  | some pair => simp



theorem applyPure_add (env : LoopEnv) {q n r c : Nat} (a b : ExactMatrix q n r c) :
    applyPure ctx env (.matrixBinary .add) [.matrix q n r c a, .matrix q n r c b] =
      some [.matrix q n r c (a + b)] := by
  simp [applyPure, Value.zipMatrix]

/-- The product of two `1 × 1` matrices. -/
theorem applyPure_mul11 (env : LoopEnv) {q n : Nat} (a b : ExactMatrix q n 1 1) :
    applyPure ctx env (.matrixBinary .mul) [.matrix q n 1 1 a, .matrix q n 1 1 b] =
      some [.matrix q n 1 1 fun i j ↦ a 0 0 * b i j] := by
  simp [applyPure, Value.mulMatrix]

theorem applyPure_scale (env : LoopEnv) {q n r c : Nat} (s : Int) (a : ExactMatrix q n r c) :
    applyPure ctx env (.matrixScale (.const s)) [.matrix q n r c a] =
      some [.matrix q n r c (matrixScale s a)] := by
  simp [applyPure, Value.mapMatrix, IExpr.eval]

theorem applyPure_concatRows (env : LoopEnv) {q n r₁ r₂ c : Nat} (a : ExactMatrix q n r₁ c)
    (b : ExactMatrix q n r₂ c) :
    applyPure ctx env (.concat .rows) [.matrix q n r₁ c a, .matrix q n r₂ c b] =
      some [.matrix q n (r₁ + r₂) c fun i j ↦
        if h : i.val < r₁ then a ⟨i.val, h⟩ j else b ⟨i.val - r₁, by omega⟩ j] := by
  simp [applyPure, concatValue]

/-- Row `row` of a matrix of `rows` rows. -/
theorem applyPure_sliceRow (env : LoopEnv) {q n rows c : Nat} (row : Nat) (h : row < rows)
    (a : ExactMatrix q n rows c) :
    applyPure ctx env (.slice (some (.const row, .const (row + 1))) none) [.matrix q n rows c a] =
      some [.matrix q n (row + 1 - row) c fun i j ↦ a ⟨row + i.val, by omega⟩ j] := by
  simp only [applyPure, sliceValue, IExpr.evalNat, IExpr.eval]
  have hnn : ¬ (row : Int) < 0 := by omega
  have hnn' : ¬ ((row + 1 : Nat) : Int) < 0 := by omega
  simp only [Nat.cast_add, Nat.cast_one] at hnn'
  simp [hnn, hnn', h, Nat.succ_le_of_lt h, sliceIndex]

/-- Column `column` of a matrix of `columns` columns. -/
theorem applyPure_sliceColumn (env : LoopEnv) {q n r columns : Nat} (column : Nat)
    (h : column < columns) (a : ExactMatrix q n r columns) :
    applyPure ctx env (.slice none (some (.const column, .const (column + 1)))) [.matrix q n r columns a] =
      some [.matrix q n r (column + 1 - column) fun i j ↦ a i ⟨column + j.val, by omega⟩] := by
  simp only [applyPure, sliceValue, IExpr.evalNat, IExpr.eval]
  have hnn : ¬ (column : Int) < 0 := by omega
  have hnn' : ¬ ((column + 1 : Nat) : Int) < 0 := by omega
  simp only [Nat.cast_add, Nat.cast_one] at hnn'
  simp [hnn, hnn', h, Nat.succ_le_of_lt h, sliceIndex]

theorem applyPure_sliceRow0 (env : LoopEnv) {q n c : Nat} (a : ExactMatrix q n 2 c) :
    applyPure ctx env (.slice (some (.const 0, .const 1)) none) [.matrix q n 2 c a] =
      some [.matrix q n 1 c fun _ j ↦ a 0 j] := by
  simp [applyPure, sliceValue, IExpr.evalNat, IExpr.eval, sliceIndex]

theorem applyPure_sliceRow1 (env : LoopEnv) {q n c : Nat} (a : ExactMatrix q n 2 c) :
    applyPure ctx env (.slice (some (.const 1, .const 2)) none) [.matrix q n 2 c a] =
      some [.matrix q n 1 c fun _ j ↦ a 1 j] := by
  simp [applyPure, sliceValue, IExpr.evalNat, IExpr.eval, sliceIndex]

theorem applyPure_sliceColumn0 (env : LoopEnv) {q n r : Nat} (a : ExactMatrix q n r 2) :
    applyPure ctx env (.slice none (some (.const 0, .const 1))) [.matrix q n r 2 a] =
      some [.matrix q n r 1 fun i _ ↦ a i 0] := by
  simp [applyPure, sliceValue, IExpr.evalNat, IExpr.eval, sliceIndex]

theorem applyPure_sliceColumn1 (env : LoopEnv) {q n r : Nat} (a : ExactMatrix q n r 2) :
    applyPure ctx env (.slice none (some (.const 1, .const 2))) [.matrix q n r 2 a] =
      some [.matrix q n r 1 fun i _ ↦ a i 1] := by
  simp [applyPure, sliceValue, IExpr.evalNat, IExpr.eval, sliceIndex]

theorem applyPure_polynomialValues (env : LoopEnv) {q n : Nat} (a : ExactMatrix q n 1 1) :
    applyPure ctx env (.polynomialValues false) [.matrix q n 1 1 a] =
      some [ToValue.encode fun i : Fin n ↦ (((a 0 0).coeff i).val : Int)] := by
  simp [applyPure, ToValue.encode]

theorem applyPure_familyGetDynamic (env : LoopEnv) {members : List Value} {index : Nat}
    {value : Value} (h : members[index]? = some value) :
    applyPure ctx env .familyGetDynamic [.family members, .int index] = some [value] := by
  simp [applyPure, h]

theorem mapM_toInt_map (l : List Int) : (l.map Value.int).mapM Value.toInt? = some l := by
  induction l with
  | nil => rfl
  | cons x xs ih => simp [List.mapM_cons, Value.toInt?, ih]

theorem applyPure_familyGetDynamic' (env : LoopEnv) (members : List Value) (index : Nat) :
    applyPure ctx env .familyGetDynamic [.family members, .int index] =
      members[index]?.map fun value ↦ [value] := by
  simp only [applyPure]
  have : ¬ ((index : Int) < 0) := by omega
  simp only [this, if_false, Int.toNat_natCast]
  cases members[index]? <;> rfl

/-- The integers of an encoded family. -/
theorem intValues?_encode {k : Nat} (v : Fin k → Int) :
    intValues? k (List.ofFn fun i ↦ Value.int (v i)) = some v := by
  have hl : (List.ofFn fun i ↦ Value.int (v i)) = (List.ofFn v).map Value.int := by
    simp [List.map_ofFn, Function.comp_def]
  simp only [intValues?, hl, mapM_toInt_map]
  simp

/-- The inverse transform of an encoded family, given the evaluated ring and the transform size.
The family stays abstract here, so instances with large families never unfold it. -/
theorem applyPure_fromValuesNtt_of (env : LoopEnv) (ring : RingE) {k : Nat} (v : Fin k → Int)
    {moduli : List Nat} {bits : Nat} (hring : ring.eval env = some (moduli, k))
    (hbits : nttBits? moduli.prod k = some bits) :
    applyPure ctx env (.polynomialFromValues ⟨ring, .const 1, .const 1⟩ true)
        [.family (List.ofFn fun i ↦ .int (v i))] =
      some [.matrix moduli.prod k 1 1 fun _ _ ↦ polynomialOfCoefficients fun j ↦
        ((inverseNtt moduli.prod bits fun s ↦ (v s : ZMod moduli.prod)) j).val] := by
  rw [applyPure_fromValuesNtt, hring]
  simp only [Option.bind_some, intValues?_encode, hbits]

/-- A parallel loop that broadcasts a value and zips an integer family, whose iteration `i`
returns `f i` without sampling, returns the family of the `f i`. -/
theorem nodeResult_broadcastZip
    {scopeEval : Nat → LoopEnv → List Value → OptionT Comp (List Value)}
    {env : LoopEnv} {inputs : List Value} {body slot k : Nat} {count : IExpr} {shared : Value}
    (idx : Fin k → Int) (f : Fin k → Value)
    (hcount : count.evalNat env = some k)
    (hbody : ∀ i : Fin k, scopeEval body (Function.update env slot i) [shared, .int (idx i)] =
      pure [f i]) :
    nodeResult ctx scopeEval env inputs (.parallelLoop body count slot [.broadcast, .zip]) 1
      [shared, .family (List.ofFn fun i ↦ .int (idx i))] =
      pure [.family (List.ofFn f)] := by
  let g : Nat → List Value := fun j ↦ if h : j < k then [f ⟨j, h⟩] else []
  have hiter : parallelIterations (scopeEval body) env slot [.broadcast, .zip]
      [shared, .family (List.ofFn fun i ↦ .int (idx i))] 0 k =
        pure ((List.range k).map fun j ↦ g (0 + j)) := by
    refine parallelIterations_pure k 0 fun i _ hi ↦ ⟨[shared, .int (idx ⟨i, by omega⟩)], ?_, ?_⟩
    · have hi' : i < k := by omega
      simp [iterationArgs, Value.toFamily?, hi']
    · simp only [g, Nat.zero_add] at hi ⊢
      rw [dif_pos hi]
      exact hbody ⟨i, by omega⟩
  simp only [nodeResult, hcount, mk_some, pure_bind, hiter, List.range_one, List.map_cons,
    List.map_nil]
  congr 3
  apply List.ext_getElem
  · simp
  · intro j h₁ h₂
    simp only [List.length_map, List.length_range] at h₁
    simp [g, h₁]

end MxxSecurity.IR
