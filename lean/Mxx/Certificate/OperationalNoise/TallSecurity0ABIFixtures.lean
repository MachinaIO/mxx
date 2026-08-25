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

def fixtureDocument : Document where
  schemaId := "mxx-operational-noise-certificate"
  schemaVersion := 1
  plaintextModulus := "2"
  ciphertextModulus := "257"
  ringDimension := 1
  expressions := [fixtureExpression 0, fixtureExpression 1, fixturePreimageExpression,
    fixtureExpression 3]
  programs := [{ signature := [], output := fixtureType, family := none, root := ⟨1⟩ }]
  sources := []
  events := [.sampler fixtureWire (.preimage fixtureType "8") none]
  indexUses := []
  sliceGroups := []
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

def fixtureFrameStarts : List Nat := [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def fixtureFinalState : ReplayState := ⟨fixtureEvents, fixtureFrameStarts, []⟩

theorem fixture_replay : replay fixtureDocument fixtureEvents = some fixtureFinalState := by
  decide

theorem fixture_valid : Valid fixtureDocument fixtureEvents := by
  exact ⟨fixtureFinalState, fixture_replay, rfl⟩

theorem dangling_survivor_rejected :
    step fixtureDocument initialState (.survivorFold 1 0) = none := by
  decide

theorem end_without_prefold_rejected :
    replay fixtureDocument
      [.invocationStart (closedOwner 0), .resultExact (closedOwner 0) [] .exactZero,
        .invocationEndExact (closedOwner 0) [] .exactZero] = none := by
  decide

def repeatedStart : List Event := fixtureEvents ++ [.invocationStart (closedOwner 0)]

theorem stale_relation_rhs_rejected :
    replay fixtureDocument
      (repeatedStart ++
        [.appliedRelation (closedOwner 0) ⟨[], [closedOwner 2, closedOwner 3]⟩ 1 0 2
          (.universal 5 ⟨[], [closedOwner 2, closedOwner 3]⟩ none 4)]) = none := by
  decide

theorem stale_relation_merge_rejected :
    replay fixtureDocument
      (repeatedStart ++
        [.coefficientMerge
          ⟨closedOwner 0, .relation 6 0, singletonMonomial 1, 1⟩]) = none := by
  decide

theorem stale_summary_evidence_rejected :
    replay fixtureDocument
      (repeatedStart ++
        [.resultExact (closedOwner 0) [] (.finite 1),
          .preFoldPolynomial [] (.finite 1) (some (.result 14 .summary))]) = none := by
  decide

theorem stale_survivor_transfer_rejected :
    replay fixtureDocument (repeatedStart ++ [.survivorFold 1 12]) = none := by
  decide

theorem repeated_invocation_rejects_stale_references :
    replay fixtureDocument
        (repeatedStart ++
          [.appliedRelation (closedOwner 0) ⟨[], [closedOwner 2, closedOwner 3]⟩ 1 0 2
            (.universal 5 ⟨[], [closedOwner 2, closedOwner 3]⟩ none 4)]) = none ∧
      replay fixtureDocument
        (repeatedStart ++
          [.coefficientMerge
            ⟨closedOwner 0, .relation 6 0, singletonMonomial 1, 1⟩]) = none ∧
      replay fixtureDocument
        (repeatedStart ++
          [.resultExact (closedOwner 0) [] (.finite 1),
            .preFoldPolynomial [] (.finite 1) (some (.result 14 .summary))]) = none ∧
      replay fixtureDocument (repeatedStart ++ [.survivorFold 1 12]) = none :=
  ⟨stale_relation_rhs_rejected, stale_relation_merge_rejected,
    stale_summary_evidence_rejected, stale_survivor_transfer_rejected⟩

theorem tall_security0_abi_fixture :
    Valid fixtureDocument fixtureEvents ∧
      step fixtureDocument initialState (.survivorFold 1 0) = none :=
  ⟨fixture_valid, dangling_survivor_rejected⟩

#print axioms tall_security0_abi_fixture
#print axioms fixture_replay
#print axioms repeated_invocation_rejects_stale_references

end Mxx.Certificate.OperationalNoise.TallSecurity0ABI
