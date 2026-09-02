import MxxIrCore.Eval

namespace Mxx.IR

noncomputable section

/- The coordinates of a typed family index, in the same outer-to-inner order
   consumed by the evaluator's family indexing operations. -/
def FamilyIndex.coordinates : (shape : List Nat) → FamilyIndex shape → List Nat
  | [], _ => []
  | _ :: rest, (head, tail) => head.val :: FamilyIndex.coordinates rest tail

/- Converting the coordinates of a typed index back through the evaluator's
   bounds-checking parser recovers that exact index. -/
theorem familyIndexFromList_coordinates (shape : List Nat) (index : FamilyIndex shape) :
    familyIndexFromList shape (FamilyIndex.coordinates shape index) = some index := by
  induction shape with
  | nil => rfl
  | cons extent rest ih =>
      rcases index with ⟨head, tail⟩
      simp [FamilyIndex.coordinates, familyIndexFromList, head.isLt, ih tail]

/- Packing an array as a row-major family preserves the element at every
   typed family index. -/
theorem Family.pack_value
    {shape : List Nat} {element : Type} {values : Array element}
    (size_eq : values.size = Family.shapeProduct shape) {family : Family shape element}
    (packed : Family.pack shape values = some family) (index : FamilyIndex shape) :
    family index = values[Family.rowMajorOffset shape index]'(by
      simpa [size_eq] using Family.rowMajorOffset_lt shape index) := by
  rw [Family.pack, dif_pos size_eq] at packed
  exact Option.some.inj packed ▸ rfl

theorem FamilyIndex.coordinates_length (shape : List Nat) (index : FamilyIndex shape) :
    (FamilyIndex.coordinates shape index).length = shape.length := by
  induction shape with
  | nil => rfl
  | cons _ rest ih =>
      rcases index with ⟨head, tail⟩
      simp [FamilyIndex.coordinates, ih tail]

theorem shapeProductArray_eq_familyShapeProduct (shape : Array Nat) :
    shapeProductArray shape = Family.shapeProduct shape.toList := by
  generalize stored : shape.toList = values
  have shape_eq : shape = values.toArray := by
    rw [← stored, Array.toArray_toList]
  subst shape
  simp at stored ⊢
  clear stored
  simp [shapeProductArray]
  change values.foldl (fun result extent ↦ result * extent) 1 =
    Family.shapeProduct values
  have foldl_product : ∀ accumulator,
      values.foldl (fun result extent ↦ result * extent) accumulator =
        accumulator * Family.shapeProduct values := by
    induction values with
    | nil => intro accumulator; simp [Family.shapeProduct]
    | cons value rest ih =>
        intro accumulator
        simp only [List.foldl_cons, Family.shapeProduct]
        rw [ih]
        simp [Nat.mul_assoc]
  simpa using foldl_product 1

/- A successful array traversal computes each output slot from the input at
   the same row-major offset. -/
theorem Array.mapM_success_pointwise
    {error input output : Type} {function : input → Except error output}
    {inputs : Array input} {outputs : Array output}
    (success : inputs.mapM function = .ok outputs) (index : Fin inputs.size) :
    ∃ size_eq : outputs.size = inputs.size,
      function inputs[index] = .ok (outputs[index.val]'(by
        rw [size_eq]
        exact index.isLt)) := by
  have satisfies := Array.SatisfiesM_mapM'
    (m := Except error)
    (p := fun current value ↦ function inputs[current] = .ok value)
    (fun current ↦ SatisfiesM_Except_eq.mpr (fun value equation ↦ equation))
  obtain ⟨size_eq, pointwise⟩ := SatisfiesM_Except_eq.mp satisfies outputs success
  exact ⟨size_eq, pointwise index.val index.isLt⟩

theorem Array.mapM_some_pointwise
    {input output : Type} {function : input → Option output}
    {inputs : Array input} {outputs : Array output}
    (success : inputs.mapM function = some outputs) (index : Fin inputs.size) :
    ∃ size_eq : outputs.size = inputs.size,
      function inputs[index] = some (outputs[index.val]'(by
        rw [size_eq]
        exact index.isLt)) := by
  have satisfies := Array.SatisfiesM_mapM'
    (m := Option)
    (p := fun current value ↦ function inputs[current] = some value)
    (fun current ↦ SatisfiesM_Option_eq.mpr (fun value equation ↦ equation))
  obtain ⟨size_eq, pointwise⟩ := SatisfiesM_Option_eq.mp satisfies outputs success
  exact ⟨size_eq, pointwise index.val index.isLt⟩

/- Coercing a dynamic array to one wire type preserves every successfully
   typed payload at the same array offset. -/
theorem coerceValues_success_pointwise
    {backend : SemanticBackend} {element : WireType}
    {values : Array (DynamicValue backend)} {typed : Array (Value backend element)}
    (success : coerceValues element values = some typed) (index : Fin values.size) :
    ∃ size_eq : typed.size = values.size,
      coerceValue element values[index] = some (typed[index.val]'(by
        rw [size_eq]
        exact index.isLt)) := by
  exact Array.mapM_some_pointwise success index

/- A successful declared-family pack exposes the typed row-major array used
   by the evaluator; no separate family equality is supplied by the caller. -/
theorem packDeclaredFamily_success
    {backend : SemanticBackend} (stage scope node : Nat)
    {shape : List Nat} {element : WireType} (values : Array (DynamicValue backend))
    (family : Family shape (Value backend element))
    (success : packDeclaredFamily stage scope node (.family shape element) values =
      .ok #[⟨.family shape element, family⟩]) :
    ∃ typedValues : Array (Value backend element),
      values.size = Family.shapeProduct shape ∧
      coerceValues element values = some typedValues ∧
      Family.pack shape typedValues = some family := by
  unfold packDeclaredFamily packFamily at success
  dsimp only at success
  by_cases size_eq : values.size = Family.shapeProduct shape
  · rw [if_pos size_eq] at success
    cases typed_eq : coerceValues element values with
    | none => simp [typed_eq] at success
    | some typedValues =>
        rw [typed_eq] at success
        dsimp only at success
        cases packed_eq : Family.pack shape typedValues with
        | none => simp [packed_eq] at success
        | some packedFamily =>
            rw [packed_eq] at success
            have family_eq : packedFamily = family := by
              have dynamic_eq :
                  (⟨.family shape element, packedFamily⟩ : DynamicValue backend) =
                    ⟨.family shape element, family⟩ := by
                have selected := congrArg (fun output => output[0]?) (Except.ok.inj success)
                simpa using selected
              cases dynamic_eq
              rfl
            subst packedFamily
            exact ⟨typedValues, size_eq, rfl, packed_eq⟩
  · simp [size_eq] at success

/- Reading a packed output at a typed index is equivalent to coercing the
   dynamic intermediate at the corresponding row-major offset. -/
theorem packDeclaredFamily_success_value
    {backend : SemanticBackend} (stage scope node : Nat)
    {shape : List Nat} {element : WireType} {values : Array (DynamicValue backend)}
    {family : Family shape (Value backend element)}
    (success : packDeclaredFamily stage scope node (.family shape element) values =
      .ok #[⟨.family shape element, family⟩]) (index : FamilyIndex shape) :
    coerceValue element
      (values[Family.rowMajorOffset shape index]'(by
        rw [(packDeclaredFamily_success stage scope node values family success).choose_spec.1]
        exact Family.rowMajorOffset_lt shape index)) = some (family index) := by
  obtain ⟨typedValues, size_eq, coerced, packed⟩ :=
    packDeclaredFamily_success stage scope node values family success
  let offset : Fin values.size := ⟨Family.rowMajorOffset shape index, by
    rw [size_eq]
    exact Family.rowMajorOffset_lt shape index⟩
  obtain ⟨typed_size, pointwise⟩ := coerceValues_success_pointwise coerced offset
  have packed_value := Family.pack_value
    (values := typedValues) (family := family)
    (by rw [typed_size, size_eq]) packed index
  rw [packed_value]
  exact pointwise

private theorem arrayMapMExcept_of_list
    {error input output : Type} (function : input → Except error output)
    (values : Array input) (results : List output)
    (success : List.mapM function values.toList = .ok results) :
    values.mapM function = .ok results.toArray := by
  have mapped := Array.toList_mapM (m := Except error) (f := function) (xs := values)
  rw [success] at mapped
  cases run : values.mapM function with
  | error error => simp [run] at mapped
  | ok outputValues =>
      simp [run] at mapped
      rw [← mapped, Array.toArray_toList]

private theorem arrayMapMOption_of_list
    {input output : Type} (function : input → Option output)
    (values : Array input) (results : List output)
    (success : List.mapM function values.toList = some results) :
    values.mapM function = some results.toArray := by
  have mapped := Array.toList_mapM (m := Option) (f := function) (xs := values)
  rw [success] at mapped
  cases run : values.mapM function with
  | none => simp [run] at mapped
  | some outputValues =>
      simp [run] at mapped
      rw [← mapped, Array.toArray_toList]

private theorem evalLiteralCoordinates
    (structural : StructuralEnv) (stage scope node : Nat) (coordinates : List Nat) :
    ((coordinates.map
      (fun coordinate ↦ IndexMapExpr.literal (Int.ofNat coordinate))).toArray.mapM
        (evalIndexExpr structural stage scope node)) =
      .ok (coordinates.map Int.ofNat).toArray := by
  apply arrayMapMExcept_of_list
  simp only [List.mapM_map]
  induction coordinates with
  | nil => rfl
  | cons coordinate rest ih =>
      simp only [List.map_cons, List.mapM_cons]
      rw [show (evalIndexExpr structural stage scope node ∘
        fun coordinate ↦ IndexMapExpr.literal (Int.ofNat coordinate)) coordinate =
        .ok (Int.ofNat coordinate) by rfl]
      rw [ih]
      rfl

private theorem nonnegativeCoordinates (coordinates : List Nat) :
    (coordinates.map Int.ofNat).toArray.mapM (fun coordinate : Int ↦
      if 0 ≤ coordinate then some coordinate.toNat else none)
      = some coordinates.toArray := by
  apply arrayMapMOption_of_list
  simp only [List.mapM_map]
  induction coordinates with
  | nil => rfl
  | cons coordinate rest ih =>
      simp [ih]

/- A static get at literal coordinates returns the source element selected by
   the corresponding typed family index. -/
theorem familyStaticGet_literals
    {backend : SemanticBackend} (structural : StructuralEnv)
    (stage scope node : Nat) {shape : List Nat} {element : WireType}
    (source : Family shape (Value backend element)) (index : FamilyIndex shape) :
    familyStaticGet structural stage scope node
      ((FamilyIndex.coordinates shape index).map
        (fun coordinate ↦ IndexMapExpr.literal (Int.ofNat coordinate))).toArray
      #[⟨.family shape element, source⟩] = .ok #[⟨element, source index⟩] := by
  rw [familyStaticGet]
  change (do
    let evaluated : Array Int ← ((FamilyIndex.coordinates shape index).map
      (fun coordinate ↦ IndexMapExpr.literal (Int.ofNat coordinate))).toArray.mapM
        (evalIndexExpr structural stage scope node)
    match evaluated.mapM (fun (coordinate : Int) ↦
        if 0 ≤ coordinate then some coordinate.toNat else none) with
    | some coordinates =>
        match familyIndexFromList shape coordinates.toList with
        | some selected =>
            pure (#[⟨element, source selected⟩] : Array (DynamicValue backend))
        | none => throw (EvalError.invalidIndex stage scope node)
    | none => throw (EvalError.invalidIndex stage scope node)) = _
  rw [evalLiteralCoordinates]
  change (match
    (FamilyIndex.coordinates shape index).map Int.ofNat |>.toArray.mapM
      (fun (coordinate : Int) ↦
        if 0 ≤ coordinate then some coordinate.toNat else none) with
    | some coordinates =>
        match familyIndexFromList shape coordinates.toList with
        | some selected =>
            Except.ok (#[⟨element, source selected⟩] : Array (DynamicValue backend))
        | none => Except.error (EvalError.invalidIndex stage scope node)
    | none => Except.error (EvalError.invalidIndex stage scope node)) = _
  rw [nonnegativeCoordinates]
  simp [familyIndexFromList_coordinates]

/- Reindexing computes the output at a typed index by running the evaluator's
   index map at that index's row-major lane.  The theorem exposes that exact
   lane computation and the resulting typed payload, without accepting a
   caller-provided source-index equality. -/
theorem familyReindex_success_at
    {backend : SemanticBackend} (structural : StructuralEnv)
    (stage scope node : Nat) {sourceShape : List Nat} {element : WireType}
    (source : Family sourceShape (Value backend element))
    (outputShape : Array StructuralIntExpr) (map : IndexMap) (concreteShape : Array Nat)
    (shape_success : evalShape structural stage scope node outputShape = .ok concreteShape)
    (output : Family concreteShape.toList (Value backend element))
    (success : familyReindex structural stage scope node outputShape map
      (.family concreteShape.toList element) #[⟨.family sourceShape element, source⟩] =
        .ok #[⟨.family concreteShape.toList element, output⟩])
    (index : FamilyIndex concreteShape.toList) :
    let lane := fun offset ↦ do
      let coordinates := coordinatesFromOffset concreteShape.toList offset
      let laneStructural := {
        structural with axes := coordinates.map Int.ofNat |>.toArray }
      let sourceIndices ← map.inputIndices.mapM
        (evalIndexExpr laneStructural stage scope node)
      match sourceIndices.mapM (fun coordinate ↦
          if 0 ≤ coordinate then some coordinate.toNat else none) with
      | some sourceCoordinates =>
          match familyIndexFromList sourceShape sourceCoordinates.toList with
          | some sourceIndex =>
              pure (⟨element, source sourceIndex⟩ : DynamicValue backend)
          | none => throw (EvalError.invalidIndex stage scope node)
      | none => throw (EvalError.invalidIndex stage scope node)
    ∃ dynamicValues,
      (Array.range (shapeProductArray concreteShape)).mapM lane = .ok dynamicValues ∧
      lane (Family.rowMajorOffset concreteShape.toList index) =
        .ok (dynamicValues[Family.rowMajorOffset concreteShape.toList index]'(by
          rw [(Array.size_mapM lane _).run (by assumption)])) ∧
      coerceValue element
        (dynamicValues[Family.rowMajorOffset concreteShape.toList index]'(by
          rw [(Array.size_mapM lane _).run (by assumption)])) = some (output index) := by
  dsimp only
  unfold familyReindex at success
  rw [shape_success] at success
  dsimp only at success
  let lane := fun offset ↦ do
    let coordinates := coordinatesFromOffset concreteShape.toList offset
    let laneStructural := {
      structural with axes := coordinates.map Int.ofNat |>.toArray }
    let sourceIndices ← map.inputIndices.mapM
      (evalIndexExpr laneStructural stage scope node)
    match sourceIndices.mapM (fun coordinate ↦
        if 0 ≤ coordinate then some coordinate.toNat else none) with
    | some sourceCoordinates =>
        match familyIndexFromList sourceShape sourceCoordinates.toList with
        | some sourceIndex => pure (⟨element, source sourceIndex⟩ : DynamicValue backend)
        | none => throw (EvalError.invalidIndex stage scope node)
    | none => throw (EvalError.invalidIndex stage scope node)
  change (do
    let dynamicValues ← (Array.range (shapeProductArray concreteShape)).mapM lane
    packDeclaredFamily stage scope node (.family concreteShape.toList element) dynamicValues) = _
    at success
  cases dynamic_success : (Array.range (shapeProductArray concreteShape)).mapM lane with
  | error error => simp [dynamic_success] at success
  | ok dynamicValues =>
      rw [dynamic_success] at success
      dsimp only at success
      let offset : Fin (Array.range (shapeProductArray concreteShape)).size :=
        ⟨Family.rowMajorOffset concreteShape.toList index, by
          simp [shapeProductArray_eq_familyShapeProduct,
            Family.rowMajorOffset_lt concreteShape.toList index]⟩
      obtain ⟨dynamic_size, lane_success⟩ :=
        Array.mapM_success_pointwise dynamic_success offset
      have packed_value := packDeclaredFamily_success_value
        stage scope node success index
      refine ⟨dynamicValues, dynamic_success, ?_, ?_⟩
      · simpa [offset] using lane_success
      · simpa [offset] using packed_value

def scalarSelectArguments {backend : SemanticBackend} {branchCount : Nat}
    (selector : Fin branchCount) (branches : Fin branchCount → DynamicValue backend) :
    Array (DynamicValue backend) :=
  #[⟨.int, Int.ofNat selector.val⟩] ++ Array.ofFn branches

/- Dynamic family get is the same lookup as static get after its integer
   operands have been converted to literal index expressions. -/
/- Successful scalar selection returns the branch at `selector + 1`; slot
   zero is the selector itself in the primitive argument array. -/
theorem select_result_argument
    {backend : SemanticBackend} {branchCount : Nat} (selector : Fin branchCount)
    (branches : Fin branchCount → DynamicValue backend) :
    (scalarSelectArguments selector branches)[selector.val + 1]? =
      some (branches selector) := by
  unfold scalarSelectArguments
  rw [Array.getElem?_append_right]
  · simp
  · simp

end

end Mxx.IR
