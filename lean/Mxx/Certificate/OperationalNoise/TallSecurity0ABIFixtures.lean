import Mxx.Certificate.OperationalNoise.TallSecurity0ABI

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0ABI

open SchemaV1

def fixtureType : ValueType := .matrix "257" 1 1 1

def fixtureExpression (position : Nat) : ExpressionRow where
  descriptor := .operation (.stable (.argument position fixtureType)) fixtureType
  inputs := []
  program := none

def fixtureRootExpression : ExpressionRow where
  descriptor := .operation (.stable (.argument 0 fixtureType)) fixtureType
  inputs := [⟨2⟩]
  program := none

def fixturePreimageExpression : ExpressionRow where
  descriptor := .operation (.event (.sampler ⟨0⟩)) fixtureType
  inputs := []
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
      (.node 3 (fixtureExpression 3) .empty .empty))
  programs := .node 0
    { signature := [], output := fixtureType, family := none, root := ⟨1⟩ } .empty .empty
  sources := .empty
  events := .node 0 (.sampler fixtureWire (.preimage fixtureType "8") none) .empty .empty
  indexUses := .empty
  sliceGroups := .empty
  residualRoot := .closed ⟨0⟩

def closedOwner (expression : Nat) : Owner := ⟨.closed ⟨0⟩, ⟨expression⟩⟩

def singletonMonomial (expression : Nat) : Monomial := ⟨[], [closedOwner expression]⟩

def singletonTerm (expression : Nat) : Term := ⟨singletonMonomial expression, 1⟩

def fixtureDispatch : UniversalDispatch := ⟨⟨0⟩, ⟨2⟩, ⟨3⟩⟩

def fixtureEvents : List Event :=
  [ .invocationStart (closedOwner 0),
    .invocationStart (closedOwner 1),
    .resultExact (closedOwner 1) [singletonTerm 1] .exactZero,
    .preFoldPolynomial [singletonTerm 1] .exactZero none,
    .invocationEndExact (closedOwner 1) [singletonTerm 1] .exactZero,
    .specializationComputed (closedOwner 0) fixtureDispatch ⟨1, 5⟩,
    .appliedRelation (closedOwner 0) ⟨[], [closedOwner 2, closedOwner 3]⟩ 1 0 2
      (.universal 5 ⟨[], [closedOwner 2, closedOwner 3]⟩ none 4),
    .coefficientMerge
      ⟨closedOwner 0, .relation 6 0, singletonMonomial 1, 1⟩,
    .resultExact (closedOwner 2) [singletonTerm 2] (.finite 1),
    .predecessor (closedOwner 0) 0 ⟨2⟩ 8,
    .boundTransfer (closedOwner 0) (.identity (.predecessor 0 .coefficient)),
    .coefficientMerge
      ⟨closedOwner 0, .operator (⟨8, 0⟩, ⟨8, 0⟩), singletonMonomial 2, 1⟩,
    .boundTransfer (closedOwner 0) (.authority (.relationPreimageSource ⟨2⟩)),
    .survivorFold 1 12,
    .resultExact (closedOwner 0) [] (.finite 1),
    .preFoldPolynomial [] (.finite 1) (some (.result 14 .summary)),
    .invocationEndExact (closedOwner 0) [] (.finite 1) ]

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

def fixtureFrameStarts : List Nat := [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0]

def fixtureHistory : EventHistory :=
  smallHistory (annotateEvents fixtureEvents fixtureFrameStarts)

def fixtureFinalState : ReplayState := ⟨fixtureEvents.length, []⟩

theorem fixture_replay :
    replayRange fixtureDocument fixtureHistory fixtureHistory.size initialState =
      some fixtureFinalState := by
  decide

theorem fixture_valid : Valid fixtureDocument fixtureHistory :=
  validOfReplay fixtureDocument fixtureHistory fixtureFinalState fixture_replay (by decide) rfl rfl

theorem dangling_survivor_rejected :
    stepAt fixtureDocument (smallHistory #[⟨.survivorFold 1 0, 0⟩]) initialState = none := by
  decide

theorem end_without_prefold_rejected :
    replay fixtureDocument (smallHistory
      [ ⟨.invocationStart (closedOwner 0), 0⟩,
        ⟨.resultExact (closedOwner 0) [] .exactZero, 0⟩,
        ⟨.invocationEndExact (closedOwner 0) [] .exactZero, 0⟩ ].toArray) = none := by
  decide

def repeatedStart : List AnnotatedEvent :=
  (annotateEvents fixtureEvents fixtureFrameStarts).toList ++
    [⟨.invocationStart (closedOwner 0), 17⟩]

theorem stale_relation_rhs_rejected :
    replay fixtureDocument (smallHistory
      (repeatedStart ++
        [annotated (.appliedRelation (closedOwner 0) ⟨[], [closedOwner 2, closedOwner 3]⟩
          1 0 2 (.universal 5 ⟨[], [closedOwner 2, closedOwner 3]⟩ none 4)) 17]).toArray) =
      none := by
  decide

theorem stale_relation_merge_rejected :
    replay fixtureDocument (smallHistory
      (repeatedStart ++
        [annotated (.coefficientMerge
          ⟨closedOwner 0, .relation 6 0, singletonMonomial 1, 1⟩) 17]).toArray) = none := by
  decide

theorem stale_summary_evidence_rejected :
    replay fixtureDocument (smallHistory
      (repeatedStart ++
        [annotated (.resultExact (closedOwner 0) [] (.finite 1)) 17,
          annotated (.preFoldPolynomial [] (.finite 1) (some (.result 14 .summary))) 17]).toArray) =
      none := by
  decide

theorem stale_survivor_transfer_rejected :
    replay fixtureDocument
      (smallHistory (repeatedStart ++
        [annotated (.survivorFold 1 12) 17]).toArray) = none := by
  decide

theorem repeated_invocation_rejects_stale_references :
    replay fixtureDocument (smallHistory
        (repeatedStart ++
          [annotated (.appliedRelation (closedOwner 0) ⟨[], [closedOwner 2, closedOwner 3]⟩
            1 0 2 (.universal 5 ⟨[], [closedOwner 2, closedOwner 3]⟩ none 4)) 17]).toArray) =
        none ∧
      replay fixtureDocument (smallHistory
        (repeatedStart ++
          [annotated (.coefficientMerge
            ⟨closedOwner 0, .relation 6 0, singletonMonomial 1, 1⟩) 17]).toArray) = none ∧
      replay fixtureDocument (smallHistory
        (repeatedStart ++
          [annotated (.resultExact (closedOwner 0) [] (.finite 1)) 17,
            annotated
              (.preFoldPolynomial [] (.finite 1) (some (.result 14 .summary))) 17]).toArray) =
        none ∧
      replay fixtureDocument
        (smallHistory (repeatedStart ++
          [annotated (.survivorFold 1 12) 17]).toArray) = none :=
  ⟨stale_relation_rhs_rejected, stale_relation_merge_rejected,
    stale_summary_evidence_rejected, stale_survivor_transfer_rejected⟩

def multipleRhsEvents : List Event :=
  [ .invocationStart (closedOwner 0),
    .invocationStart (closedOwner 1),
    .invocationStart (closedOwner 2),
    .resultExact (closedOwner 2) [singletonTerm 2] .exactZero,
    .preFoldPolynomial [singletonTerm 2] .exactZero none,
    .invocationEndExact (closedOwner 2) [singletonTerm 2] .exactZero,
    .resultExact (closedOwner 1) [singletonTerm 1] .exactZero,
    .preFoldPolynomial [singletonTerm 1] .exactZero none,
    .invocationEndExact (closedOwner 1) [singletonTerm 1] .exactZero,
    .specializationComputed (closedOwner 0) fixtureDispatch ⟨1, 9⟩,
    .appliedRelation (closedOwner 0) ⟨[], [closedOwner 2, closedOwner 3]⟩ 1 0 2
      (.universal 9 ⟨[], [closedOwner 2, closedOwner 3]⟩ none 5),
    .coefficientMerge
      ⟨closedOwner 0, .relation 10 0, singletonMonomial 1, 1⟩,
    .resultExact (closedOwner 0) [] .exactZero,
    .preFoldPolynomial [] .exactZero none,
    .invocationEndExact (closedOwner 0) [] .exactZero ]

def multipleRhsFrameStarts : List Nat := [0, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0,
  0]

def multipleRhsHistory : EventHistory :=
  smallHistory (annotateEvents multipleRhsEvents multipleRhsFrameStarts)

def multipleRhsFinalState : ReplayState :=
  ⟨multipleRhsEvents.length, []⟩

theorem in_range_nonfinal_rhs_accepted :
    replayRange fixtureDocument multipleRhsHistory multipleRhsHistory.size initialState =
      some multipleRhsFinalState := by
  decide

def multipleRhsBeforeComputed : ReplayState :=
  ⟨9, [⟨closedOwner 0, 0, #[none], none, false⟩]⟩

theorem nested_endpoint_equivalence :
    exactFrameRange multipleRhsHistory multipleRhsBeforeComputed ⟨1, 9⟩ = true := by
  decide

def duplicatePredecessorEvents : List Event :=
  [ .invocationStart (closedOwner 0),
    .resultExact (closedOwner 2) [singletonTerm 2] (.finite 1),
    .predecessor (closedOwner 0) 0 ⟨2⟩ 1,
    .predecessor (closedOwner 0) 0 ⟨2⟩ 1 ]

theorem duplicate_predecessor_rejected :
    replay fixtureDocument
      (smallHistory (annotateEvents duplicatePredecessorEvents [0, 0, 0, 0])) = none := by
  decide

theorem wrong_nested_annotation_rejected :
    replay fixtureDocument
      (smallHistory #[⟨.invocationStart (closedOwner 0), 0⟩,
        ⟨.invocationStart (closedOwner 1), 0⟩]) = none := by
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
  ⟨64, [⟨closedOwner 0, 0, #[none], none, false⟩]⟩

def afterBoundary : ReplayState :=
  ⟨65, [⟨closedOwner 0, 0, #[none], none, false⟩]⟩

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
#print axioms fixture_replay
#print axioms repeated_invocation_rejects_stale_references
#print axioms in_range_nonfinal_rhs_accepted
#print axioms duplicate_predecessor_rejected
#print axioms nested_endpoint_equivalence
#print axioms wrong_nested_annotation_rejected
#print axioms replay_chunk_crosses_64_boundary
#print axioms malformed_history_rejected

end Mxx.Certificate.OperationalNoise.TallSecurity0ABI
