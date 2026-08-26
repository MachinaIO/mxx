import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0ABIFixtures

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSemanticsFixtures

open Mxx.Certificate.OperationalNoise
open EventReplay
open SchemaV1
open TallSecurity0ABI
open TallSemantics

def atomType : ValueType := .matrix "257" 1 1 1

def atomContract : RawValueContract :=
  { signedRange := none
    coefficientClass := some (.finite "5")
    canonicalCoefficientExclusiveUpper := none
    polynomialSupportUpper := none }

def atomSourceIdentity : SourceIdentity :=
  { definition := "toy-source"
    sampleEvent := none
    outputRole := "value"
    artifact := none
    valueType := atomType
    coordinates := []
    matrixConstant := none }

def sourceAtomOwner : Owner := closedOwner 0

def sourceAtomDocument : TallDocument where
  schemaId := "mxx-operational-noise-certificate"
  schemaVersion := 1
  plaintextModulus := "2"
  ciphertextModulus := "257"
  ringDimension := 1
  expressions := .node 0
    { descriptor := .source (.direct ⟨0⟩)
      inputs := emptyExpressionInputs
      program := none } .empty .empty
  programs := .empty
  sources := .node 0 (.direct atomSourceIdentity none (some atomContract)) .empty .empty
  events := .empty
  indexUses := .empty
  sliceGroups := .empty
  residualRoot := .closed ⟨0⟩

def sourceAtomHistory : EventHistory :=
  smallHistory #[
    annotated (.boundTransfer sourceAtomOwner (.sum [])) 0,
    annotated (.resultExact sourceAtomOwner [canonicalSelfTerm sourceAtomOwner]
      .exactZero 0 .exactZero none) 0]

def samplerAtomOwner : Owner := closedOwner 0

def samplerAtomDocument : TallDocument where
  schemaId := "mxx-operational-noise-certificate"
  schemaVersion := 1
  plaintextModulus := "2"
  ciphertextModulus := "257"
  ringDimension := 1
  expressions := .node 0
    { descriptor := .event (.sampler ⟨0⟩)
      inputs := emptyExpressionInputs
      program := none } .empty .empty
  programs := .empty
  sources := .empty
  events := .node 0 (.sampler fixtureWire (.preimage atomType "5") (some atomContract)) .empty .empty
  indexUses := .empty
  sliceGroups := .empty
  residualRoot := .closed ⟨0⟩

def samplerAtomHistory : EventHistory :=
  smallHistory #[
    annotated (.boundTransfer samplerAtomOwner (.sum [])) 0,
    annotated (.resultExact samplerAtomOwner [canonicalSelfTerm samplerAtomOwner]
      .exactZero 0 .exactZero none) 0]

def terminalAtomHistory : EventHistory :=
  smallHistory #[annotated
    (.boundTransfer sourceAtomOwner (.authority .factStore)) 0,
    annotated
      (.resultExact sourceAtomOwner [canonicalSelfTerm sourceAtomOwner]
        .exactZero 0 .exactZero none) 0]

def boundFixtureOwner : Owner := closedOwner 0

def boundFixtureReference : ValueRef := .result 1 .coefficient

def productFixtureReference : ValueRef := .result 3 .coefficient

def productFixtureFacts : TallSecurity0ABI.ProductFacts :=
  ⟨false, false, none, none, none⟩

def boundFixtureHistory : EventHistory :=
    smallHistory #[
    annotated (.boundTransfer boundFixtureOwner (.sum [])) 7,
    annotated (.resultExact boundFixtureOwner [] .exactZero 0 .exactZero (some 0)) 7,
    annotated (.boundTransfer boundFixtureOwner (.identity boundFixtureReference)) 7,
    annotated (.resultCoefficient boundFixtureOwner .exactZero) 7,
    annotated (.boundTransfer boundFixtureOwner
      (.product productFixtureReference productFixtureReference productFixtureFacts)) 7,
    annotated (.boundTransfer boundFixtureOwner (.authority .operator)) 7,
    annotated (.resultCoefficient boundFixtureOwner .exactZero) 7,
    annotated (.boundTransfer boundFixtureOwner
      (.scale productFixtureReference (.value productFixtureReference))) 7]

def mergeLeftTerm : Term :=
  { monomial := { centralFactors := [], orderedFactors := [] }, coefficient := 2 }

def mergeRightTerm : Term :=
  { monomial := { centralFactors := [], orderedFactors := [] }, coefficient := 3 }

def operatorMergeHistory : EventHistory :=
  smallHistory #[
    annotated (.resultExact boundFixtureOwner [mergeLeftTerm] (.finite 0) 0 .exactZero none) 7,
    annotated (.resultExact boundFixtureOwner [mergeRightTerm] (.finite 0) 0 .exactZero none) 7,
    annotated (.boundTransfer boundFixtureOwner
      (.product (.result 0 .coefficient) (.result 1 .coefficient) productFixtureFacts)) 7,
    annotated (.coefficientMerge
      ⟨boundFixtureOwner, .operator (⟨0, 0⟩, ⟨1, 0⟩),
        { centralFactors := [], orderedFactors := [] }, 6⟩) 7]

theorem operator_merge_delta_fixture :
    MergeDeltaAt operatorMergeHistory 3 7 boundFixtureOwner (.operator 0 1)
      { coefficient := 6, key := { centralFactors := [], orderedFactors := [] } } := by
  apply MergeDeltaAt.operator
    (leftResult := 0) (leftOrdinal := 0) (rightResult := 1) (rightOrdinal := 0)
    (leftTerms := [mergeLeftTerm]) (rightTerms := [mergeRightTerm])
    (leftTerm := mergeLeftTerm) (rightTerm := mergeRightTerm)
    (output := { centralFactors := [], orderedFactors := [] }) (signedContribution := 6)
  all_goals rfl

def relationMergeHistory : EventHistory :=
  smallHistory #[
    annotated (.resultExact boundFixtureOwner [mergeLeftTerm] (.finite 0) 0 .exactZero none) 7,
    annotated (.appliedRelation boundFixtureOwner
      { centralFactors := [], orderedFactors := [] } 3 0 0
      (.gadget boundFixtureOwner boundFixtureOwner ⟨0⟩ 0)) 7,
    annotated (.coefficientMerge
      ⟨boundFixtureOwner, .relation 1 0,
        { centralFactors := [], orderedFactors := [] }, 6⟩) 7]

theorem relation_merge_delta_fixture :
    MergeDeltaAt relationMergeHistory 2 7 boundFixtureOwner (.relation 1)
      { coefficient := 6, key := { centralFactors := [], orderedFactors := [] } } := by
  apply MergeDeltaAt.relation (application := 1) (rhsResult := 0)
    (sourceTermOrdinal := 0) (source := { centralFactors := [], orderedFactors := [] })
    (outerCoefficient := 3) (orderedStart := 0) (orderedEndExclusive := 0)
    (rule := .gadget boundFixtureOwner boundFixtureOwner ⟨0⟩ 0)
    (rhsTerms := [mergeLeftTerm]) (rhsTerm := mergeLeftTerm)
    (output := { centralFactors := [], orderedFactors := [] }) (signedContribution := 6)
  all_goals rfl

theorem balanced_merge_deltas_fixture :
    MergeDeltasAt operatorMergeHistory 7 boundFixtureOwner (.operator 0 1)
      [{ coefficient := 6, key := { centralFactors := [], orderedFactors := [] } },
        { coefficient := 6, key := { centralFactors := [], orderedFactors := [] } }] := by
  exact .append (.leaf operator_merge_delta_fixture) (.leaf operator_merge_delta_fixture)

def operator_merge_reconstruction_fixture :
    MergeReconstructionAt operatorMergeHistory 7 boundFixtureOwner (.operator 0 1) []
      [{ coefficient := 6, key := { centralFactors := [], orderedFactors := [] } }] := by
  refine ⟨_, .leaf operator_merge_delta_fixture, ?_⟩
  decide +kernel

def tensorFixtureLayout : Layout := ⟨"row-major-1x1", 1, 1⟩

def tensorFixtureDocument : TallDocument where
  schemaId := "mxx-operational-noise-certificate"
  schemaVersion := 1
  plaintextModulus := "2"
  ciphertextModulus := "257"
  ringDimension := 1
  expressions := .node 0
    { descriptor := .operation (.stable (.matrix
        (.tensor atomType tensorFixtureLayout tensorFixtureLayout tensorFixtureLayout))) atomType
      inputs := emptyExpressionInputs
      program := none } .empty .empty
  programs := .empty
  sources := .empty
  events := .empty
  indexUses := .empty
  sliceGroups := .empty
  residualRoot := .closed ⟨0⟩

def tensorMergeHistory : EventHistory :=
  smallHistory #[
    annotated (.resultExact boundFixtureOwner [mergeLeftTerm] (.finite 0) 0 .exactZero none) 7,
    annotated (.resultExact boundFixtureOwner [mergeRightTerm] (.finite 0) 0 .exactZero none) 7,
    annotated (.boundTransfer boundFixtureOwner
      (.tensor (.result 0 .coefficient) (.result 1 .coefficient) true false)) 7,
    annotated (.coefficientMerge
      ⟨boundFixtureOwner, .operator (⟨0, 0⟩, ⟨1, 0⟩),
        { centralFactors := [], orderedFactors := [] }, 6⟩) 7]

theorem tensor_merge_delta_fixture :
    MergeDeltaAt tensorMergeHistory 3 7 boundFixtureOwner (.operator 0 1)
      { coefficient := 6, key := { centralFactors := [], orderedFactors := [] } } := by
  apply MergeDeltaAt.operator
    (leftResult := 0) (leftOrdinal := 0) (rightResult := 1) (rightOrdinal := 0)
    (leftTerms := [mergeLeftTerm]) (rightTerms := [mergeRightTerm])
    (leftTerm := mergeLeftTerm) (rightTerm := mergeRightTerm)
    (output := { centralFactors := [], orderedFactors := [] }) (signedContribution := 6)
  all_goals rfl

def tensor_merge_reconstruction_fixture :
    MergeReconstructionAt tensorMergeHistory 7 boundFixtureOwner (.operator 0 1) []
      [{ coefficient := 6, key := { centralFactors := [], orderedFactors := [] } }] := by
  refine ⟨_, .leaf tensor_merge_delta_fixture, ?_⟩
  decide +kernel

theorem tensor_left_claim_fixture :
    ExactClaimAt tensorMergeHistory 257 (fun _ => 1) 0 boundFixtureOwner 2
      [mergeLeftTerm] .exactZero := by
  refine ⟨⟨.finite 0, 0, none, rfl⟩, 0, ?_, ?_⟩
  · decide +kernel
  · simp [boundInterprets, centeredNorm, centeredCoefficient]

theorem tensor_right_claim_fixture :
    ExactClaimAt tensorMergeHistory 257 (fun _ => 1) 1 boundFixtureOwner 3
      [mergeRightTerm] .exactZero := by
  refine ⟨⟨.finite 0, 0, none, rfl⟩, 0, ?_, ?_⟩
  · decide +kernel
  · simp [boundInterprets, centeredNorm, centeredCoefficient]

theorem operator_tensor_merge_claim_fixture :
    ValueClaim.Interprets 257 (fun _ : Owner => 1) (2 * 3)
      (.exact [{ coefficient := 6, key := { centralFactors := [], orderedFactors := [] } }]
        .exactZero) := by
  apply operatorTensorMergeClaim (document := tensorFixtureDocument)
      (history := tensorMergeHistory) (frameStart := 7) (transferEvent := 2)
      (owner := boundFixtureOwner) (leftResult := 0) (rightResult := 1)
      (leftOwner := boundFixtureOwner) (rightOwner := boundFixtureOwner)
      (leftRaw := [mergeLeftTerm]) (rightRaw := [mergeRightTerm])
      (leftReference := .result 0 .coefficient) (rightReference := .result 1 .coefficient)
      (reconstruction := tensor_merge_reconstruction_fixture)
  · rfl
  · rfl
  · exact tensor_left_claim_fixture
  · exact tensor_right_claim_fixture
  · decide +kernel
  · decide

def addNoMergeFixtureDocument : TallDocument where
  schemaId := "mxx-operational-noise-certificate"
  schemaVersion := 1
  plaintextModulus := "2"
  ciphertextModulus := "257"
  ringDimension := 1
  expressions := .node 0
    { descriptor := .operation (.stable (.matrix .add)) atomType
      inputs := emptyExpressionInputs
      program := none } .empty .empty
  programs := .empty
  sources := .empty
  events := .empty
  indexUses := .empty
  sliceGroups := .empty
  residualRoot := .closed ⟨0⟩

def addNoMergeOutputTerm : Term :=
  { monomial := { centralFactors := [], orderedFactors := [] }, coefficient := 5 }

def addNoMergeHistory : EventHistory :=
  smallHistory #[
    annotated (.resultExact boundFixtureOwner [mergeLeftTerm] (.finite 0) 0 (.finite 1) none) 7,
    annotated (.resultExact boundFixtureOwner [mergeRightTerm] (.finite 0) 0 (.finite 2) none) 7,
    annotated (.predecessor boundFixtureOwner 0 ⟨0⟩ 0) 7,
    annotated (.predecessor boundFixtureOwner 1 ⟨0⟩ 1) 7,
    annotated (.boundTransfer boundFixtureOwner
      (.sum [.predecessor 0 2 .coefficient, .predecessor 1 3 .coefficient])) 7,
    annotated (.boundTransfer boundFixtureOwner
      (.sum [.result 0 .summary, .result 1 .summary])) 7,
    annotated (.resultExact boundFixtureOwner [addNoMergeOutputTerm]
      (.finite 0) 4 (.finite 3) (some 5)) 7]

theorem add_no_merge_left_claim_fixture :
    ExactClaimAt addNoMergeHistory 257 (fun _ : Owner ↦ 1) 0 boundFixtureOwner 3
      [mergeLeftTerm] (.finite 1) := by
  apply exactFiniteClaimAt (remainder := 1)
  · rfl
  · decide +kernel
  · simp [centeredNorm, centeredCoefficient]

theorem add_no_merge_right_claim_fixture :
    ExactClaimAt addNoMergeHistory 257 (fun _ : Owner ↦ 1) 1 boundFixtureOwner 5
      [mergeRightTerm] (.finite 2) := by
  apply exactFiniteClaimAt (remainder := 2)
  · rfl
  · decide +kernel
  · simp [centeredNorm, centeredCoefficient]

theorem operator_add_no_merge_claim_fixture :
    ExactClaimAt addNoMergeHistory 257 (fun _ : Owner ↦ 1) 6 boundFixtureOwner 8
      [addNoMergeOutputTerm] (.finite 3) := by
  apply operatorAddNoMergeClaim (document := addNoMergeFixtureDocument)
      (history := addNoMergeHistory) (modulus := 257) (frameStart := 7)
      (transferEvent := 4) (summaryTransferEvent := 5) (resultEvent := 6)
      (env := fun _ : Owner ↦ 1) (owner := boundFixtureOwner)
      (leftOwner := boundFixtureOwner) (rightOwner := boundFixtureOwner)
      (leftResult := 0) (rightResult := 1) (leftBinding := 2) (rightBinding := 3)
      (leftInputPosition := 0) (rightInputPosition := 1)
      (leftExpression := ⟨0⟩) (rightExpression := ⟨0⟩)
      (leftActual := 3) (rightActual := 5)
      (leftRaw := [mergeLeftTerm]) (rightRaw := [mergeRightTerm])
      (outputRaw := [addNoMergeOutputTerm]) (leftMaximum := 1) (rightMaximum := 2)
      (valueType := atomType) (coefficientBound := .finite 0)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact add_no_merge_left_claim_fixture
  · exact add_no_merge_right_claim_fixture
  · rfl
  · decide +kernel
  · decide

theorem wrong_merge_delta_reconstruction_rejected :
    ¬ CanonicalAgreement
      [{ coefficient := 5, key := { centralFactors := [], orderedFactors := [] } }]
      (add [] [{ coefficient := 6, key := { centralFactors := [], orderedFactors := [] } }]) := by
  decide +kernel

theorem bound_identity_fixture :
    BoundDerivedAt boundFixtureHistory 2 7 boundFixtureOwner
      (.identity boundFixtureReference) .exactZero 0 := by
  apply BoundDerivedAt.identity
  · rfl
  · exact .result rfl (.resultExactCoefficient (rule := .sum []) (by rfl)
      (by simp [RecordedBoundRefines]) (.sum (by rfl) .nil))

theorem bound_transfer_input_fixture :
    BoundInputAt boundFixtureHistory boundFixtureOwner (.transfer 2) .exactZero 0 := by
  exact boundDerived_to_transferInput bound_identity_fixture

theorem bound_identity_projection_fixture :
    ProjectedBoundAt boundFixtureHistory 3 boundFixtureOwner none .coefficient
      .exactZero 0 := by
  exact .resultCoefficient rfl (by rfl) bound_identity_fixture

theorem bound_summary_projection_fixture :
    ProjectedBoundAt boundFixtureHistory 1 boundFixtureOwner (some []) .summary
      .exactZero 0 := by
  exact boundTransfer_to_resultExactSummary (rule := .sum []) (by rfl)
    (.sum (by rfl) .nil)

theorem bound_result_summary_input_fixture :
    BoundInputAt boundFixtureHistory boundFixtureOwner (.result 1 .summary) .exactZero 0 := by
  exact projectedSummary_to_resultInput rfl bound_summary_projection_fixture

theorem bound_product_fixture :
    BoundDerivedAt boundFixtureHistory 4 7 boundFixtureOwner
      (.product productFixtureReference productFixtureReference productFixtureFacts)
      .exactZero 0 := by
  apply BoundDerivedAt.product (leftRows := 1) (leftColumns := 1) (rightRows := 1)
    (rightColumns := 1) (ringDimension := 1) (factor := 1)
    (leftBound := .exactZero) (rightBound := .exactZero)
    (leftActual := 0) (rightActual := 0)
  · rfl
  · rfl
  · exact .result rfl bound_identity_projection_fixture
  · exact .result rfl bound_identity_projection_fixture

theorem authority_leaf_fixture :
    AuthorityLeafAt boundFixtureHistory 5 6 7 boundFixtureOwner .operator .exactZero := by
  exact .resultCoefficient rfl (by rfl) (by rfl)

def authorityWitnessFixture : AuthorityWitness boundFixtureHistory where
  authorityMagnitude := fun _ ↦ 0
  authorityBound := by
    intros _ _ _ _ _ bound _
    cases bound <;> simp [CoeffClass.Interprets]

theorem bound_authority_fixture :
    BoundDerivedAt boundFixtureHistory 5 7 boundFixtureOwner (.authority .operator)
      .exactZero (authorityWitnessFixture.authorityMagnitude 6) := by
  exact .authority authorityWitnessFixture authority_leaf_fixture

theorem bound_scale_value_fixture :
    BoundDerivedAt boundFixtureHistory 7 7 boundFixtureOwner
      (.scale productFixtureReference (.value productFixtureReference)) .exactZero 0 := by
  apply BoundDerivedAt.scaleValue (valueBound := .exactZero) (scaleBound := .exactZero)
    (valueActual := 0) (scaleActual := 0)
  · rfl
  · exact .result rfl bound_identity_projection_fixture
  · exact .result rfl bound_identity_projection_fixture

def zeroOwnerEnv : Env Owner := fun _ ↦ 0

theorem finite_add_exact_zero_fixture :
    ValueClaim.Interprets 257 zeroOwnerEnv 1 (.exact [] (.finite 1)) := by
  apply exactValueClaim_add_right_mod_zero 257 zeroOwnerEnv 1 0 [] [] [] 1
  · refine ⟨1, by decide, ?_⟩
    simp [boundInterprets, centeredNorm, centeredCoefficient]
  · decide
  · rfl

def finiteRewriteOutput : Polynomial Owner :=
  [{ coefficient := 257, key := { centralFactors := [], orderedFactors := [] } }]

theorem finite_relation_rewrite_fixture :
    ValueClaim.Interprets 257 zeroOwnerEnv 1
      (.exact finiteRewriteOutput (.finite 1)) := by
  apply exactFiniteValueClaim_of_eval_mod 257 zeroOwnerEnv 1 [] finiteRewriteOutput 1
  · refine ⟨1, ?_, ?_⟩
    · decide
    · simp [boundInterprets, centeredNorm, centeredCoefficient]
  · decide

theorem exact_zero_recording_refines_finite_two_fixture :
    RecordedBoundRefines .exactZero (.finite ⟨2, by decide⟩) := by
  trivial

def exactBoundFixtureTerms : List Term := [canonicalSelfTerm boundFixtureOwner]

def exactBoundFixtureHistory : EventHistory :=
  smallHistory #[
    annotated (.boundTransfer boundFixtureOwner (.sum [])) 7,
    annotated (.resultExact boundFixtureOwner exactBoundFixtureTerms .exactZero 0
      (.finite 4) (some 0)) 7]

theorem exact_finite_claim_fixture :
    ExactClaimAt exactBoundFixtureHistory 257 (fun _ => 0) 1 boundFixtureOwner 0
      exactBoundFixtureTerms (.finite 4) := by
  apply exactFiniteClaimAt (actual := 0) (remainder := 0) (maximum := 4) (frameStart := 7)
    (coefficientProducer := 0) (coefficientBound := .exactZero)
    (summaryProducer := some 0) (by rfl)
  · decide
  · simp [centeredNorm, centeredCoefficient]

theorem terminal_atom_at :
    TerminalExactAt sourceAtomDocument terminalAtomHistory none 0 1 sourceAtomOwner
      [canonicalSelfTerm sourceAtomOwner] := by
  refine ⟨rfl, rfl, .authority .factStore, 0, .exactZero, ?_, rfl, rfl⟩
  exact ReachedTerminalRule.authorityFactStore

theorem terminal_exact_claim_fixture
    (witness : Witness sourceAtomDocument terminalAtomHistory none 257) :
    ExactClaimAt terminalAtomHistory 257 witness.env 1 sourceAtomOwner
      (witness.honestTerminalActual 1) [canonicalSelfTerm sourceAtomOwner] .exactZero := by
  exact terminalExactClaimAt witness terminal_atom_at

theorem source_atom_fixture
    (witness : Witness sourceAtomDocument sourceAtomHistory none 257) :
  DerivedResult sourceAtomDocument sourceAtomHistory none 257 witness
      sourceAtomOwner 1 := by
  apply ValueDerived.sourceAtom ⟨0⟩
  · refine ⟨0, .exactZero, 0, ?_⟩
    rfl
  · refine ⟨?_, ?_, ?_⟩
    · refine ⟨⟨.resultExact sourceAtomOwner [canonicalSelfTerm sourceAtomOwner]
          .exactZero 0 .exactZero none, 0⟩,
        ?_, ?_⟩
      · rfl
      · exact ⟨canonicalSelfTerm sourceAtomOwner,
          by simp [canonicalSelfTerm, termContains, monomialContains]⟩
    · rfl
    · refine ⟨⟨.source (.direct ⟨0⟩), emptyExpressionInputs, none⟩,
        ⟨.direct atomSourceIdentity none (some atomContract), ?_⟩⟩
      constructor
      · rfl
      constructor
      · rfl
      · rfl

theorem source_atom_interprets
    (witness : Witness sourceAtomDocument sourceAtomHistory none 257) :
    ValueClaim.Interprets 257 witness.env (witness.env sourceAtomOwner)
      (canonicalSelfClaim sourceAtomOwner) := by
  exact ValueDerived.interprets (source_atom_fixture witness)

theorem sampler_atom_fixture
    (witness : Witness samplerAtomDocument samplerAtomHistory none 257) :
  DerivedResult samplerAtomDocument samplerAtomHistory none 257 witness
      samplerAtomOwner 1 := by
  apply ValueDerived.samplerAtom ⟨0⟩
  · refine ⟨0, .exactZero, 0, ?_⟩
    rfl
  · refine ⟨?_, ?_, ?_⟩
    · refine ⟨⟨.resultExact samplerAtomOwner [canonicalSelfTerm samplerAtomOwner]
          .exactZero 0 .exactZero none, 0⟩,
        ?_, ?_⟩
      · rfl
      · exact ⟨canonicalSelfTerm samplerAtomOwner,
          by simp [canonicalSelfTerm, termContains, monomialContains]⟩
    · rfl
    · refine ⟨⟨.event (.sampler ⟨0⟩), emptyExpressionInputs, none⟩,
        ⟨EventRow.sampler fixtureWire (.preimage atomType "5") (some atomContract), ?_⟩⟩
      constructor
      · rfl
      constructor
      · exact List.mem_cons_self
      · rfl

theorem sampler_atom_interprets
    (witness : Witness samplerAtomDocument samplerAtomHistory none 257) :
    ValueClaim.Interprets 257 witness.env (witness.env samplerAtomOwner)
      (canonicalSelfClaim samplerAtomOwner) := by
  exact ValueDerived.interprets (sampler_atom_fixture witness)

def factorA : Owner := ⟨.program ⟨0⟩, ⟨10⟩⟩
def factorB : Owner := ⟨.program ⟨0⟩, ⟨11⟩⟩

def fixtureEnv : Env Owner
  | owner => if owner = factorA then 3 else if owner = factorB then -2 else 0

def fixturePolynomial : Polynomial Owner :=
  [⟨2, ⟨[factorA], [factorB]⟩⟩, ⟨-1, ⟨[], [factorA]⟩⟩]

theorem generic_evaluation_fixture : evalPolynomial fixtureEnv fixturePolynomial = -15 := by
  decide

theorem generic_value_claim_fixture :
    ValueClaim.Interprets 257 fixtureEnv (-14) (.exact fixturePolynomial (.finite 1)) := by
  apply exactValueClaim_of_remainder 257 fixtureEnv (-14) fixturePolynomial 1 1
  · decide
  · simp [centeredNorm, centeredCoefficient]

def finiteOneClass : CoeffClass := .finite ⟨1, by decide⟩

theorem reached_sum_class_fixture :
    (addKnownList [.exactZero, finiteOneClass]).Interprets ([0, 1] : List Nat).sum := by
  apply addKnownList_sound
  exact List.Forall₂.cons (by rfl)
    (List.Forall₂.cons (by simp [finiteOneClass, CoeffClass.Interprets]) List.Forall₂.nil)

theorem reached_finite_exact_claim_fixture :
    ValueClaim.Interprets 257 fixtureEnv (-14)
      (.exact fixturePolynomial (coeffClassToTallBound finiteOneClass)) := by
  apply exactValueClaim_of_coeffClass 257 fixtureEnv (-14) fixturePolynomial finiteOneClass 1
  · decide
  · simp [finiteOneClass, CoeffClass.Interprets, centeredNorm, centeredCoefficient]

theorem exact_zero_right_preserves_bound_fixture :
    ValueClaim.Interprets 257 fixtureEnv (-14 - 0)
      (.exact fixturePolynomial (.finite 1)) := by
  have rightClaim : ValueClaim.Interprets 257 fixtureEnv 0 (.exact [] .exactZero) := by
    refine ⟨0, ?_, ?_⟩
    · decide
    · simp [boundInterprets, centeredNorm, centeredCoefficient]
  apply exactValueClaim_sub_of_mod_zero 257 fixtureEnv (-14) 0
    fixturePolynomial [] fixturePolynomial 1 generic_value_claim_fixture
  · exact exactClaim_mod_zero 257 fixtureEnv 0 [] rightClaim (by decide)
  · rfl

theorem empty_finite_claim_bounds_actual_fixture
    (claim : ValueClaim.Interprets 257 fixtureEnv 1 (.exact [] (.finite 1))) :
    centeredNorm 257 1 ≤ 1 := by
  exact centeredNorm_le_of_empty_finite_claim 257 fixtureEnv 1 1 claim (by decide)

theorem empty_finite_claim_final_bound_fixture
    (claim : ValueClaim.Interprets 257 fixtureEnv 1 (.exact [] (.finite 1))) :
    2 * 2 * centeredNorm 257 1 < 257 := by
  exact finalStrictBound_of_empty_finite_claim 2 257 fixtureEnv 1 1 claim
    (by decide) (by decide)

def familyRoot : SchemaV1.ResidualRoot := .family ⟨0⟩ ⟨2, 5⟩

theorem statement_domain_fixture :
    ForStatement familyRoot (fun selector ↦ selector = some 2 ∨ selector = some 3 ∨
      selector = some 4) := by
  simp [ForStatement, familyRoot]
  intro selector lower upper
  omega

def finiteContract : RawValueContract :=
  { signedRange := none
    coefficientClass := some (.finite "3")
    canonicalCoefficientExclusiveUpper := none
    polynomialSupportUpper := none }

theorem constructive_raw_bound_fixture : rawValueContractInterprets 257 (-3) finiteContract := by
  exact ⟨.finite "3", rfl, 3, rfl, by decide⟩

def cp3Term (coefficient : Int) (central ordered : List Owner) : ExactTerm Owner :=
  { coefficient := coefficient
    key := { centralFactors := central, orderedFactors := ordered } }

def cp3CentralLeft : Polynomial Owner :=
  [cp3Term 1 [factorA, factorB] []]

def cp3CentralPermutation : Polynomial Owner :=
  [cp3Term 1 [factorB, factorA] []]

def cp3OrderedLeft : Polynomial Owner :=
  [cp3Term 1 [] [factorA, factorB]]

def cp3OrderedSwap : Polynomial Owner :=
  [cp3Term 1 [] [factorB, factorA]]

def cp3DuplicateLeft : Polynomial Owner :=
  [cp3Term 1 [] [factorA], cp3Term 2 [] [factorA]]

def cp3DuplicateCombined : Polynomial Owner :=
  [cp3Term 3 [] [factorA]]

def cp3ZeroCancellation : Polynomial Owner :=
  [cp3Term 1 [] [factorA], cp3Term (-1) [] [factorA]]

def cp3CoefficientMutation : Polynomial Owner :=
  [cp3Term 4 [] [factorA]]

def cp3OmittedTerm : Polynomial Owner := []

def cp3ExtraTerm : Polynomial Owner :=
  [cp3Term 1 [] [factorA], cp3Term 1 [] [factorB]]

def canonicalRelationKey : MonomialKey Owner :=
  { centralFactors := [], orderedFactors := [] }

def canonicalRelationContext := relationContext canonicalRelationKey [] 0 0

def canonicalRelationPolynomial : Polynomial Owner :=
  relationPoly [] canonicalRelationKey canonicalRelationContext 0 []

theorem cp3_central_permutation_accepted :
    CanonicalAgreement cp3CentralLeft cp3CentralPermutation := by
  rfl

theorem cp3_ordered_swap_rejected :
    ¬ CanonicalAgreement cp3OrderedLeft cp3OrderedSwap := by
  intro h
  cases h

theorem cp3_duplicate_coefficients_combine :
    CanonicalAgreement cp3DuplicateLeft cp3DuplicateCombined := by
  rfl

theorem cp3_zero_cancellation :
    CanonicalAgreement cp3ZeroCancellation [] := by
  rfl

theorem cp3_coefficient_mutation_rejected :
    ¬ CanonicalAgreement cp3DuplicateCombined cp3CoefficientMutation := by
  intro h
  cases h

theorem cp3_omitted_term_rejected :
    ¬ CanonicalAgreement cp3DuplicateCombined cp3OmittedTerm := by
  intro h
  cases h

theorem cp3_extra_term_rejected :
    ¬ CanonicalAgreement cp3DuplicateCombined cp3ExtraTerm := by
  intro h
  cases h

theorem canonical_relation_fixture :
    CanonicalAgreement [] canonicalRelationPolynomial := by
  decide

#print axioms canonicalPolynomial_eval
#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.forall₂_append
#print axioms canonicalAgreement_eval
#print axioms addCanonicalResultSound
#print axioms subCanonicalResultSound
#print axioms productCanonicalResultSound
#print axioms cp3_central_permutation_accepted
#print axioms cp3_ordered_swap_rejected
#print axioms cp3_duplicate_coefficients_combine
#print axioms cp3_zero_cancellation
#print axioms cp3_coefficient_mutation_rejected
#print axioms cp3_omitted_term_rejected
#print axioms cp3_extra_term_rejected
#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.relationCanonicalResultSound
#print axioms canonical_relation_fixture

#print axioms generic_evaluation_fixture
#print axioms generic_value_claim_fixture
#print axioms bound_identity_fixture
#print axioms operator_merge_delta_fixture
#print axioms relation_merge_delta_fixture
#print axioms balanced_merge_deltas_fixture
#print axioms operator_merge_reconstruction_fixture
#print axioms operator_tensor_merge_claim_fixture
#print axioms TallSemantics.exactValueClaim_add_finite
#print axioms TallSemantics.operatorAddNoMergeClaim
#print axioms operator_add_no_merge_claim_fixture
#print axioms wrong_merge_delta_reconstruction_rejected
#print axioms TallSemantics.operatorAddMergeClaim
#print axioms TallSemantics.operatorSubMergeClaim
#print axioms TallSemantics.operatorProductMergeClaim
#print axioms TallSemantics.operatorTensorMergeClaim
#print axioms TallSemantics.universalRelationMergeClaim
#print axioms TallSemantics.gadgetRelationMergeClaim
#print axioms TallSemantics.exactClaimAt_of_mergeClaim
#print axioms bound_transfer_input_fixture
#print axioms bound_identity_projection_fixture
#print axioms bound_summary_projection_fixture
#print axioms bound_result_summary_input_fixture
#print axioms bound_product_fixture
#print axioms authority_leaf_fixture
#print axioms bound_authority_fixture
#print axioms bound_scale_value_fixture
#print axioms finite_add_exact_zero_fixture
#print axioms TallSemantics.exactFiniteValueClaim_of_eval_mod
#print axioms finite_relation_rewrite_fixture
#print axioms exact_zero_recording_refines_finite_two_fixture
#print axioms TallSemantics.boundTransfer_to_resultCoefficient
#print axioms TallSemantics.ProjectedBoundAt.sound
#print axioms TallSemantics.BoundDerivedAt.sound
#print axioms exact_finite_claim_fixture
#print axioms TallSemantics.coeffClassInterprets_to_boundInterprets
#print axioms TallSemantics.addKnownList_sound
#print axioms TallSemantics.exactValueClaim_of_coeffClass
#print axioms reached_sum_class_fixture
#print axioms reached_finite_exact_claim_fixture
#print axioms TallSemantics.centeredNorm_eq_of_emod_eq
#print axioms TallSemantics.centeredNorm_le_of_empty_finite_claim
#print axioms TallSemantics.finalStrictBound_of_empty_finite_claim
#print axioms empty_finite_claim_bounds_actual_fixture
#print axioms empty_finite_claim_final_bound_fixture
#print axioms statement_domain_fixture
#print axioms constructive_raw_bound_fixture
#print axioms TallSemantics.ValueDerived.interprets
#print axioms TallSemantics.terminalExactClaimAt
#print axioms terminal_atom_at
#print axioms terminal_exact_claim_fixture
#print axioms source_atom_interprets
#print axioms sampler_atom_interprets

end Mxx.Certificate.OperationalNoise.TallSemanticsFixtures
