import Mxx.Certificate.OperationalNoise.TallSecurity0ABI

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0ABI

open SchemaV1

def fixtureType : ValueType := .matrix "257" 1 1 1

def emptyExpressionInputs : ExpressionInputs := ⟨.empty, 0⟩

def repeatedInputLeaf : Array ExpressionRef :=
  Array.replicate orderedLeafSize ⟨2⟩

def longExpressionInputs : ExpressionInputs where
  leaves := .node 0 repeatedInputLeaf .empty
    (.node 1 #[⟨2⟩, ⟨2⟩] .empty .empty)
  size := 18

def twoFullExpressionInputs : ExpressionInputs where
  leaves := .node 0 repeatedInputLeaf .empty
    (.node 1 repeatedInputLeaf .empty .empty)
  size := 32

def singletonExpressionInputs (input : Nat) : ExpressionInputs where
  leaves := .node 0 #[⟨input⟩] .empty .empty
  size := 1

def fixtureExpression (position : Nat) : ExpressionRow where
  descriptor := .operation (.stable (.argument position fixtureType)) fixtureType
  inputs := emptyExpressionInputs
  program := none

def fixtureRootExpression : ExpressionRow where
  descriptor := .operation (.stable (.argument 0 fixtureType)) fixtureType
  inputs := longExpressionInputs
  program := none

def fixtureConsumerExpression (position input : Nat) : ExpressionRow where
  descriptor := .operation (.stable (.argument position fixtureType)) fixtureType
  inputs := singletonExpressionInputs input
  program := none

def fixturePreimageExpression : ExpressionRow where
  descriptor := .operation (.event (.sampler ⟨0⟩)) fixtureType
  inputs := emptyExpressionInputs
  program := none

def fixtureWire : ObservedWire where
  stage := "security0-fixture"
  definition := .root
  path := 0
  node := 0
  port := 0

def fixtureDocument : TallDocument where
  schemaId := "mxx-operational-noise-certificate"
  schemaVersion := 1
  plaintextModulus := "2"
  ciphertextModulus := "257"
  ringDimension := 1
  expressions := .node 1 (fixtureExpression 1)
    (.node 0 fixtureRootExpression .empty .empty)
    (.node 2 fixturePreimageExpression .empty
      (.node 3 (fixtureConsumerExpression 3 2) .empty .empty))
  programs := .node 0
    { signature := [], output := fixtureType, family := none, root := ⟨1⟩ } .empty .empty
  sources := .empty
  events := .node 0 (.sampler fixtureWire (.preimage fixtureType "8") none) .empty .empty
  indexUses := .empty
  sliceGroups := .empty
  residualRoot := .closed ⟨0⟩

def gappedExpressionInputs : ExpressionInputs where
  leaves := .node 0 repeatedInputLeaf .empty
    (.node 2 #[⟨2⟩] .empty .empty)
  size := 17

def shortNonfinalExpressionInputs : ExpressionInputs where
  leaves := .node 0 (Array.replicate 15 ⟨2⟩) .empty
    (.node 1 #[⟨2⟩] .empty .empty)
  size := 17

def wrongFinalExpressionInputs : ExpressionInputs where
  leaves := .node 0 repeatedInputLeaf .empty
    (.node 1 #[⟨2⟩, ⟨2⟩, ⟨2⟩] .empty .empty)
  size := 17

theorem expression_inputs_fixture :
    emptyExpressionInputs.Valid ∧ longExpressionInputs.Valid ∧
      twoFullExpressionInputs.Valid ∧
      longExpressionInputs.get? 0 = some ⟨2⟩ ∧
      longExpressionInputs.get? 8 = some ⟨2⟩ ∧
      longExpressionInputs.get? 15 = some ⟨2⟩ ∧
      longExpressionInputs.get? 16 = some ⟨2⟩ ∧
      longExpressionInputs.get? 17 = some ⟨2⟩ ∧
      longExpressionInputs.get? 18 = none ∧
      longExpressionInputs.get? 0 = longExpressionInputs.get? 15 := by
  simp [OrderedLeaves.Valid, OrderedLeaves.wellFormed, OrderedLeaves.leafCount,
    OrderedLeaves.expectedLeafSize, OrderedLeaves.get?, emptyExpressionInputs,
    longExpressionInputs, twoFullExpressionInputs, repeatedInputLeaf, orderedLeafSize,
    rowTableNodeCount, RowTable.wellFormed, RowTable.orderedFrom, RowTable.balanced,
    RowTable.height, RowTable.allBool, RowTable.lookup]

theorem malformed_expression_inputs_rejected :
    gappedExpressionInputs.wellFormed = false ∧
      shortNonfinalExpressionInputs.wellFormed = false ∧
      wrongFinalExpressionInputs.wellFormed = false := by
  decide

def genericSeventeen : OrderedLeaves Nat where
  leaves := .node 0 (Array.replicate orderedLeafSize 7) .empty
    (.node 1 #[7] .empty .empty)
  size := 17

theorem generic_ordered_leaves_fixture :
    genericSeventeen.Valid ∧
      genericSeventeen.get? 0 = some 7 ∧
      genericSeventeen.get? 15 = some 7 ∧
      genericSeventeen.get? 16 = some 7 ∧
      genericSeventeen.get? 17 = none ∧
      genericSeventeen.get? 0 = genericSeventeen.get? 15 := by
  simp [OrderedLeaves.Valid, OrderedLeaves.wellFormed, OrderedLeaves.leafCount,
    OrderedLeaves.expectedLeafSize, OrderedLeaves.get?, genericSeventeen, orderedLeafSize,
    rowTableNodeCount, RowTable.wellFormed, RowTable.orderedFrom, RowTable.balanced,
    RowTable.height, RowTable.allBool, RowTable.lookup]

def balancedLeafTable {α : Type} (values : Nat → Array α) (start count : Nat) :
    RowTable (Array α) :=
  if count = 0 then .empty else
    let leftCount := count / 2
    let root := start + leftCount
    .node root (values root)
      (balancedLeafTable values start leftCount)
      (balancedLeafTable values (root + 1) (count - leftCount - 1))
termination_by count
decreasing_by all_goals omega

def twoAxisIndexFrontier : List SchemaV1.FrontierAxis :=
  [.argument ⟨.root, 0⟩ (.expression ⟨10⟩) 0 (3, 5),
    .argument ⟨.root, 0⟩ (.expression ⟨11⟩) 1 (7, 10)]

def subtractIndexLutRows : IndexLutRows :=
  ⟨.add ⟨15⟩
    (.subtract ⟨12⟩ (.binding ⟨10⟩) (.binding ⟨11⟩))
    (.constant ⟨14⟩ (-2))⟩

theorem index_lut_rows_fixture :
    subtractIndexLutRows.get? twoAxisIndexFrontier 0 = some ⟨["3", "7"], "-6"⟩ ∧
      subtractIndexLutRows.get? twoAxisIndexFrontier 1 = some ⟨["3", "8"], "-7"⟩ ∧
      subtractIndexLutRows.get? twoAxisIndexFrontier 2 = some ⟨["3", "9"], "-8"⟩ ∧
      subtractIndexLutRows.get? twoAxisIndexFrontier 3 = some ⟨["4", "7"], "-5"⟩ ∧
      subtractIndexLutRows.get? twoAxisIndexFrontier 5 = some ⟨["4", "9"], "-7"⟩ ∧
      subtractIndexLutRows.get? twoAxisIndexFrontier 6 = none := by
  decide

def negativeDivisionRows : IndexLutRows :=
  ⟨.divide ⟨20⟩ (.constant ⟨21⟩ (-7)) (.constant ⟨22⟩ (-3))⟩

def negativeRemainderRows : IndexLutRows :=
  ⟨.remainder ⟨23⟩ (.constant ⟨21⟩ (-7)) (.constant ⟨22⟩ (-3))⟩

def zeroDivisionRows : IndexLutRows :=
  ⟨.divide ⟨24⟩ (.constant ⟨21⟩ (-7)) (.constant ⟨25⟩ 0)⟩

theorem typed_index_euclidean_fixture :
    negativeDivisionRows.get? [] 0 = some ⟨[], "-3"⟩ ∧
      negativeRemainderRows.get? [] 0 = some ⟨[], "2"⟩ ∧
      zeroDivisionRows.get? [] 0 = none ∧
      negativeDivisionRows.get? [] 1 = none := by
  decide

def closedOwner (expression : Nat) : Owner := ⟨.closed ⟨0⟩, ⟨expression⟩⟩

def singletonMonomial (expression : Nat) : Monomial := ⟨[], [closedOwner expression]⟩

def singletonTerm (expression : Nat) : Term := ⟨singletonMonomial expression, 1⟩

def ownerProductLeft : Monomial :=
  ⟨[closedOwner 10], [closedOwner 11, closedOwner 12]⟩

def ownerProductRight : Monomial :=
  ⟨[closedOwner 20], [closedOwner 21]⟩

def ownerProductOutput : Monomial :=
  ⟨[closedOwner 20, closedOwner 10], [closedOwner 11, closedOwner 12, closedOwner 21]⟩

def wrongOwnerProductOutput : Monomial :=
  ⟨[closedOwner 30, closedOwner 10], [closedOwner 11, closedOwner 12, closedOwner 21]⟩

def reversedOwnerProductOutput : Monomial :=
  ⟨[closedOwner 20, closedOwner 10], [closedOwner 21, closedOwner 12, closedOwner 11]⟩

theorem owner_product_key_fixture :
    ProductKey ownerProductLeft.toKey ownerProductRight.toKey ownerProductOutput.toKey := by
  constructor
  · exact .swap _ _ []
  · rfl

theorem owner_product_key_rejections :
    ¬ ProductKey ownerProductLeft.toKey ownerProductRight.toKey wrongOwnerProductOutput.toKey ∧
      ¬ ProductKey ownerProductLeft.toKey ownerProductRight.toKey
        reversedOwnerProductOutput.toKey := by
  constructor
  · intro valid
    have present : closedOwner 30 ∈
        ownerProductLeft.toKey.centralFactors ++ ownerProductRight.toKey.centralFactors :=
      valid.1.mem_iff.mp (by simp [wrongOwnerProductOutput, Monomial.toKey])
    simp [ownerProductLeft, ownerProductRight, Monomial.toKey, closedOwner] at present
  · intro valid
    have ordered := valid.2
    simp [ownerProductLeft, ownerProductRight, reversedOwnerProductOutput,
      Monomial.toKey, closedOwner] at ordered

def universalSourceMonomial : Monomial :=
  ⟨[closedOwner 30], [closedOwner 10, closedOwner 11, closedOwner 12]⟩

def universalReplacementMonomial : Monomial :=
  ⟨[closedOwner 31], [closedOwner 20, closedOwner 21]⟩

def universalContextOutput : Monomial :=
  ⟨[closedOwner 31, closedOwner 30],
    [closedOwner 10, closedOwner 20, closedOwner 21, closedOwner 12]⟩

theorem universal_context_key_fixture :
    ContextKey universalSourceMonomial.toKey universalReplacementMonomial.toKey
      universalContextOutput.toKey 1 2 := by
  constructor
  · exact .swap _ _ []
  · rfl

theorem owner_term_bridge_fixture :
    (singletonTerm 7).toExact =
      { coefficient := 1, key := { centralFactors := [], orderedFactors := [closedOwner 7] } } := by
  rfl

def largeTerms : List Term :=
  List.replicate 39 (singletonTerm 1) ++ [singletonTerm 2, singletonTerm 1]

theorem term_exists_fixture :
    termExists [singletonTerm 1] 0 = true ∧
      termExists [singletonTerm 1] 1 = false ∧
      termExists largeTerms 39 = true ∧
      termExists largeTerms 40 = true ∧
      termExists largeTerms 41 = false ∧
      List.get?Internal largeTerms 39 = some (singletonTerm 2) ∧
      List.get?Internal largeTerms 40 = some (singletonTerm 1) := by
  decide

def fixtureDispatch : UniversalDispatch := ⟨⟨0⟩, ⟨2⟩, ⟨3⟩⟩

def fixtureEvents : List Event :=
  [ .invocationStart (closedOwner 0),
    .invocationStart (closedOwner 1),
    .boundTransfer (closedOwner 1) (.authority .operator),
    .resultExact (closedOwner 1) [singletonTerm 1] .exactZero 2 .exactZero none,
    .preFoldPolynomial 3 [singletonTerm 1] .exactZero none,
    .invocationEndExact (closedOwner 1) 4 [singletonTerm 1] .exactZero 2 .exactZero none,
    .specializationComputed (closedOwner 0) fixtureDispatch ⟨1, 6⟩,
    .appliedRelation (closedOwner 0) ⟨[], [closedOwner 2, closedOwner 3]⟩ 1 0 2
      (.universal 6 ⟨[], [closedOwner 2, closedOwner 3]⟩ none 5),
    .coefficientMerge
      ⟨closedOwner 0, .relation 7 0, singletonMonomial 1, 1⟩,
    .boundTransfer (closedOwner 2) (.authority .operator),
    .resultExact (closedOwner 2) [singletonTerm 2] (.finite 1) 9 (.finite 1) (some 9),
    .predecessor (closedOwner 0) 0 ⟨2⟩ 10,
    .boundTransfer (closedOwner 0) (.identity (.predecessor 0 11 .coefficient)),
    .predecessor (closedOwner 3) 0 ⟨2⟩ 10,
    .boundTransfer (closedOwner 3) (.identity (.predecessor 0 13 .coefficient)),
    .predecessor (closedOwner 0) 17 ⟨2⟩ 10,
    .boundTransfer (closedOwner 0) (.identity (.predecessor 17 15 .coefficient)),
    .coefficientMerge
      ⟨closedOwner 0, .operator (⟨10, 0⟩, ⟨10, 0⟩), singletonMonomial 2, 1⟩,
    .boundTransfer (closedOwner 0) (.authority (.relationPreimageSource ⟨2⟩)),
    .survivorFold 1 18,
    .resultExact (closedOwner 0) [] (.finite 1) 18 (.finite 1) (some 18),
    .preFoldPolynomial 20 [] (.finite 1) (some (.result 20 .summary)),
    .invocationEndExact (closedOwner 0) 21 [] (.finite 1) 18 (.finite 1) (some 18) ]

def annotateEvents (events : List Event) (frameStarts : List Nat) : Array AnnotatedEvent :=
  (List.zipWith (fun event frameStart => ⟨event, frameStart⟩) events frameStarts).toArray

def smallHistory (events : Array AnnotatedEvent) : EventHistory where
  leaves := if events.size ≤ eventLeafSize then
      .node 0 events .empty .empty
    else
      .node 0 (events.extract 0 eventLeafSize) .empty
        (.node 1 (events.extract eventLeafSize events.size) .empty .empty)
  size := events.size

def annotated (event : Event) (frameStart : Nat) : AnnotatedEvent :=
  ⟨event, frameStart⟩

def fixtureFrameStarts : List Nat := [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0]

def fixtureHistory : EventHistory :=
  smallHistory (annotateEvents fixtureEvents fixtureFrameStarts)

def fixtureFinalState : ReplayState := ⟨fixtureEvents.length, []⟩

def fixtureState1 : ReplayState := ⟨1, [⟨closedOwner 0, 0⟩]⟩

def fixtureState5 : ReplayState :=
  ⟨5, [⟨closedOwner 1, 1⟩, ⟨closedOwner 0, 0⟩]⟩

def fixtureState14 : ReplayState := ⟨16, [⟨closedOwner 0, 0⟩]⟩

def fixtureState18 : ReplayState := ⟨20, [⟨closedOwner 0, 0⟩]⟩

theorem fixture_four_mixed_lifecycle_and_prior_refs :
    replayRange fixtureDocument fixtureHistory 5 fixtureState1 = some fixtureState5 := by
  rfl

theorem fixture_four_crosses_leaf_boundary :
    replayRange fixtureDocument fixtureHistory 20 fixtureState14 = some fixtureState18 := by
  rfl

theorem fixture_replay :
    ReplayChain fixtureDocument fixtureHistory initialState fixtureFinalState :=
  .trans (.chunk 4 (by rfl))
    (.trans (.chunk 8 (by rfl))
      (.trans (.chunk 12 (by rfl))
        (.trans (.chunk 16 (by rfl)) (.trans (.chunk 20 (by rfl)) (.chunk 23 (by rfl))))))

theorem fixture_valid : Valid fixtureDocument fixtureHistory :=
  ⟨by decide, fixtureFinalState, fixture_replay, rfl, rfl⟩

theorem dangling_survivor_rejected :
    stepAt fixtureDocument (smallHistory #[⟨.survivorFold 1 0, 0⟩]) initialState = none := by
  decide

def finiteWithoutSummaryProducerHistory : EventHistory :=
  smallHistory #[
    annotated (.invocationStart (closedOwner 0)) 0,
    annotated (.boundTransfer (closedOwner 0) (.authority .operator)) 0,
    annotated (.resultExact (closedOwner 0) [] .exactZero 1 (.finite 1) none) 0]

theorem finite_without_summary_producer_structurally_accepted :
    replayRange fixtureDocument finiteWithoutSummaryProducerHistory 3 initialState =
      some ⟨3, [⟨closedOwner 0, 0⟩]⟩ := by
  rfl

theorem end_without_prefold_rejected :
    replay fixtureDocument (smallHistory
      [ ⟨.invocationStart (closedOwner 0), 0⟩,
        ⟨.resultExact (closedOwner 0) [] .exactZero 0 .exactZero none, 0⟩,
        ⟨.invocationEndExact (closedOwner 0) 1 [] .exactZero 0 .exactZero none, 0⟩ ].toArray) =
      none := by
  decide

def repeatedStart : List AnnotatedEvent :=
  (annotateEvents fixtureEvents fixtureFrameStarts).toList ++
    [⟨.invocationStart (closedOwner 0), 21⟩]

def repeatedState : ReplayState := ⟨22, [⟨closedOwner 0, 21⟩]⟩

theorem stale_relation_rhs_rejected :
    let history := smallHistory
      (repeatedStart ++ [annotated
        (.appliedRelation (closedOwner 0) ⟨[], [closedOwner 2, closedOwner 3]⟩ 1 0 2
          (.universal 5 ⟨[], [closedOwner 2, closedOwner 3]⟩ none 4)) 21]).toArray
    replayRange fixtureDocument history history.size repeatedState = none := by
  decide

theorem stale_relation_merge_rejected :
    let history := smallHistory (repeatedStart ++ [annotated
      (.coefficientMerge ⟨closedOwner 0, .relation 6 0, singletonMonomial 1, 1⟩) 21]).toArray
    replayRange fixtureDocument history history.size repeatedState = none := by
  decide

theorem stale_summary_evidence_rejected :
    let history := smallHistory (repeatedStart ++
      [annotated (.resultExact (closedOwner 0) [] (.finite 1) 0 (.finite 1) (some 0)) 21,
        annotated (.preFoldPolynomial 22 [] (.finite 1) (some (.result 18 .summary))) 21]).toArray
    replayRange fixtureDocument history history.size repeatedState = none := by
  decide

theorem stale_survivor_transfer_rejected :
    let history := smallHistory
      (repeatedStart ++ [annotated (.survivorFold 1 16) 21]).toArray
    replayRange fixtureDocument history history.size repeatedState = none := by
  decide

theorem repeated_invocation_rejects_stale_references :
    (let history := smallHistory
      (repeatedStart ++ [annotated
        (.appliedRelation (closedOwner 0) ⟨[], [closedOwner 2, closedOwner 3]⟩ 1 0 2
          (.universal 5 ⟨[], [closedOwner 2, closedOwner 3]⟩ none 4)) 21]).toArray
    replayRange fixtureDocument history history.size repeatedState = none) ∧
      (let history := smallHistory (repeatedStart ++ [annotated
        (.coefficientMerge ⟨closedOwner 0, .relation 6 0, singletonMonomial 1, 1⟩) 21]).toArray
      replayRange fixtureDocument history history.size repeatedState = none) ∧
      (let history := smallHistory (repeatedStart ++
        [annotated (.resultExact (closedOwner 0) [] (.finite 1) 0 (.finite 1) (some 0)) 21,
          annotated (.preFoldPolynomial 22 [] (.finite 1) (some (.result 18 .summary))) 21]).toArray
      replayRange fixtureDocument history history.size repeatedState = none) ∧
      (let history := smallHistory
        (repeatedStart ++ [annotated (.survivorFold 1 16) 21]).toArray
      replayRange fixtureDocument history history.size repeatedState = none) :=
  ⟨stale_relation_rhs_rejected, stale_relation_merge_rejected,
    stale_summary_evidence_rejected, stale_survivor_transfer_rejected⟩

def multipleRhsEvents : List Event :=
  [ .invocationStart (closedOwner 0),
    .invocationStart (closedOwner 1),
    .invocationStart (closedOwner 2),
    .boundTransfer (closedOwner 2) (.authority .operator),
    .resultExact (closedOwner 2) [singletonTerm 2] .exactZero 3 .exactZero none,
    .preFoldPolynomial 4 [singletonTerm 2] .exactZero none,
    .invocationEndExact (closedOwner 2) 5 [singletonTerm 2] .exactZero 3 .exactZero none,
    .boundTransfer (closedOwner 1) (.authority .operator),
    .resultExact (closedOwner 1) [singletonTerm 1] .exactZero 7 .exactZero none,
    .preFoldPolynomial 8 [singletonTerm 1] .exactZero none,
    .invocationEndExact (closedOwner 1) 9 [singletonTerm 1] .exactZero 7 .exactZero none,
    .specializationComputed (closedOwner 3) fixtureDispatch ⟨1, 11⟩,
    .appliedRelation (closedOwner 0) ⟨[], [closedOwner 2, closedOwner 3]⟩ 1 0 2
      (.universal 11 ⟨[], [closedOwner 2, closedOwner 3]⟩ none 6),
    .coefficientMerge
      ⟨closedOwner 0, .relation 12 0, singletonMonomial 1, 1⟩,
    .boundTransfer (closedOwner 0) (.authority .operator),
    .resultExact (closedOwner 0) [] .exactZero 14 .exactZero none,
    .preFoldPolynomial 15 [] .exactZero none,
    .invocationEndExact (closedOwner 0) 16 [] .exactZero 14 .exactZero none ]

def multipleRhsFrameStarts : List Nat := [0, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0,
  0, 0, 0, 0]

def multipleRhsHistory : EventHistory :=
  smallHistory (annotateEvents multipleRhsEvents multipleRhsFrameStarts)

def multipleRhsFinalState : ReplayState :=
  ⟨multipleRhsEvents.length, []⟩

theorem in_range_nonfinal_rhs_accepted :
    ReplayChain fixtureDocument multipleRhsHistory initialState multipleRhsFinalState :=
  .trans (.chunk 4 (by rfl))
    (.trans (.chunk 8 (by rfl)) (.trans (.chunk 12 (by rfl))
      (.trans (.chunk 16 (by rfl)) (.chunk 18 (by rfl)))))

def multipleRhsBeforeComputed : ReplayState :=
  ⟨11, [⟨closedOwner 0, 0⟩]⟩

theorem nested_invocation_range_fixture :
    specializationRangeValid multipleRhsHistory multipleRhsBeforeComputed ⟨1, 11⟩ = true ∧
      completedInvocationInRange multipleRhsHistory ⟨1, 11⟩ 6 = some (closedOwner 2) := by
  decide

def siblingEvents : List Event :=
  [ .invocationStart (closedOwner 0),
    .invocationStart (closedOwner 1),
    .boundTransfer (closedOwner 1) (.authority .operator),
    .resultExact (closedOwner 1) [singletonTerm 1] .exactZero 2 .exactZero none,
    .preFoldPolynomial 3 [singletonTerm 1] .exactZero none,
    .invocationEndExact (closedOwner 1) 4 [singletonTerm 1] .exactZero 2 .exactZero none,
    .invocationStart (closedOwner 2),
    .boundTransfer (closedOwner 2) (.authority .operator),
    .resultExact (closedOwner 2) [singletonTerm 2] .exactZero 7 .exactZero none,
    .preFoldPolynomial 8 [singletonTerm 2] .exactZero none,
    .invocationEndExact (closedOwner 2) 9 [singletonTerm 2] .exactZero 7 .exactZero none,
    .specializationComputed (closedOwner 3) fixtureDispatch ⟨1, 11⟩,
    .appliedRelation (closedOwner 0) ⟨[], [closedOwner 2, closedOwner 3]⟩ 1 0 2
      (.universal 11 ⟨[], [closedOwner 2, closedOwner 3]⟩ none 10),
    .boundTransfer (closedOwner 0) (.authority .operator),
    .resultExact (closedOwner 0) [] .exactZero 13 .exactZero none,
    .preFoldPolynomial 14 [] .exactZero none,
    .invocationEndExact (closedOwner 0) 15 [] .exactZero 13 .exactZero none ]

def siblingFrameStarts : List Nat := [0, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 0, 0, 0,
  0, 0, 0]

def siblingHistory : EventHistory :=
  smallHistory (annotateEvents siblingEvents siblingFrameStarts)

def siblingFinalState : ReplayState := ⟨siblingEvents.length, []⟩

theorem sibling_specialization_and_rhs_accepted :
    ReplayChain fixtureDocument siblingHistory initialState siblingFinalState :=
  .trans (.chunk 4 (by rfl))
    (.trans (.chunk 8 (by rfl)) (.trans (.chunk 12 (by rfl))
      (.trans (.chunk 16 (by rfl)) (.chunk 17 (by rfl)))))

def mismatchedInvocationHistory : EventHistory :=
  smallHistory (annotateEvents
    [ .invocationStart (closedOwner 0),
      .invocationStart (closedOwner 1),
      .resultExact (closedOwner 1) [] .exactZero 0 .exactZero none,
      .preFoldPolynomial 2 [] .exactZero none,
      .invocationEndExact (closedOwner 2) 3 [] .exactZero 0 .exactZero none ]
    [0, 1, 1, 1, 1])

def largeSparseRangeHistory : EventHistory where
  leaves := .node 0
    (#[annotated (.invocationStart (closedOwner 0)) 0,
      annotated (.invocationStart (closedOwner 1)) 1] ++
      Array.replicate 14 (annotated (.resultCoefficient (closedOwner 1) .exactZero) 1))
    .empty .empty
  size := 100000

theorem specialization_range_rejections :
    specializationRangeValid multipleRhsHistory multipleRhsBeforeComputed ⟨1, 1⟩ = false ∧
      specializationRangeValid multipleRhsHistory multipleRhsBeforeComputed ⟨5, 1⟩ = false ∧
      specializationRangeValid multipleRhsHistory multipleRhsBeforeComputed ⟨1, 8⟩ = false ∧
      specializationRangeValid multipleRhsHistory multipleRhsBeforeComputed ⟨2, 9⟩ = false := by
  decide

theorem completed_invocation_rejections :
    completedInvocationInRange mismatchedInvocationHistory ⟨1, 5⟩ 4 = none ∧
      completedInvocationInRange multipleRhsHistory ⟨1, 5⟩ 8 = none ∧
      completedInvocationInRange largeSparseRangeHistory ⟨1, 100000⟩ 999 = none := by
  decide

def largeSparseRangeState : ReplayState :=
  ⟨100000, [⟨closedOwner 0, 0⟩]⟩

theorem specialization_range_uses_bounded_lookups :
    specializationRangeValid largeSparseRangeHistory largeSparseRangeState
      ⟨1, 100000⟩ = true := by
  decide

theorem wrong_nested_annotation_rejected :
    replay fixtureDocument
      (smallHistory #[⟨.invocationStart (closedOwner 0), 0⟩,
        ⟨.invocationStart (closedOwner 1), 0⟩]) = none := by
  decide

def nestedLinkPrefix : List Event :=
  [ .invocationStart (closedOwner 0),
    .invocationStart (closedOwner 1),
    .resultExact (closedOwner 1) [singletonTerm 1] .exactZero 0 .exactZero none,
    .preFoldPolynomial 2 [singletonTerm 1] .exactZero none,
    .invocationEndExact (closedOwner 1) 3 [singletonTerm 1] .exactZero 0 .exactZero none ]

def nestedLinkState : ReplayState :=
  ⟨5, [⟨closedOwner 0, 0⟩]⟩

theorem nested_frame_links_are_isolated :
    let preFoldHistory := smallHistory (annotateEvents
      (nestedLinkPrefix ++ [.preFoldPolynomial 2 [] .exactZero none])
      [0, 1, 1, 1, 1, 0])
    let endHistory := smallHistory (annotateEvents
      (nestedLinkPrefix ++
        [.invocationEndExact (closedOwner 0) 3 [] .exactZero 0 .exactZero none])
      [0, 1, 1, 1, 1, 0])
    replayRange fixtureDocument preFoldHistory preFoldHistory.size nestedLinkState = none ∧
      replayRange fixtureDocument endHistory endHistory.size nestedLinkState = none := by
  decide

def repeatedCoefficient : AnnotatedEvent :=
  ⟨.resultCoefficient (closedOwner 0) .exactZero, 0⟩

def fullLeaf : Array AnnotatedEvent := Array.replicate eventLeafSize repeatedCoefficient

def crossBoundaryHistory : EventHistory where
  leaves := .node 2 fullLeaf
    (.node 1 fullLeaf (.node 0 fullLeaf .empty .empty) .empty)
    (.node 3 fullLeaf .empty (.node 4 #[repeatedCoefficient] .empty .empty))
  size := 65

def beforeBoundary : ReplayState :=
  ⟨64, [⟨closedOwner 0, 0⟩]⟩

def afterBoundary : ReplayState :=
  ⟨65, [⟨closedOwner 0, 0⟩]⟩

theorem replay_chunk_crosses_64_boundary :
    replayRange fixtureDocument crossBoundaryHistory 65 beforeBoundary = some afterBoundary := by
  decide

def malformedFinalLeaf : EventHistory := ⟨.node 0 #[] .empty .empty, 1⟩

def gappedLeafHistory : EventHistory :=
  ⟨.node 0 fullLeaf .empty (.node 2 #[repeatedCoefficient] .empty .empty), 17⟩

theorem malformed_history_rejected :
    malformedFinalLeaf.wellFormed = false ∧ gappedLeafHistory.wellFormed = false := by
  decide

theorem tall_security0_abi_fixture :
    Valid fixtureDocument fixtureHistory ∧
      stepAt fixtureDocument (smallHistory #[⟨.survivorFold 1 0, 0⟩]) initialState = none :=
  ⟨fixture_valid, dangling_survivor_rejected⟩

#print axioms tall_security0_abi_fixture
#print axioms term_exists_fixture
#print axioms owner_product_key_fixture
#print axioms owner_product_key_rejections
#print axioms universal_context_key_fixture
#print axioms owner_term_bridge_fixture
#print axioms fixture_replay
#print axioms fixture_four_mixed_lifecycle_and_prior_refs
#print axioms fixture_four_crosses_leaf_boundary
#print axioms finite_without_summary_producer_structurally_accepted
#print axioms repeated_invocation_rejects_stale_references
#print axioms in_range_nonfinal_rhs_accepted
#print axioms nested_invocation_range_fixture
#print axioms sibling_specialization_and_rhs_accepted
#print axioms specialization_range_rejections
#print axioms completed_invocation_rejections
#print axioms specialization_range_uses_bounded_lookups
#print axioms wrong_nested_annotation_rejected
#print axioms nested_frame_links_are_isolated
#print axioms replay_chunk_crosses_64_boundary
#print axioms malformed_history_rejected
#print axioms expression_inputs_fixture
#print axioms malformed_expression_inputs_rejected
#print axioms generic_ordered_leaves_fixture
#print axioms index_lut_rows_fixture
#print axioms typed_index_euclidean_fixture

end Mxx.Certificate.OperationalNoise.TallSecurity0ABI
