import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0ABIFixtures

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSemanticsFixtures

open Mxx.Certificate.OperationalNoise
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
  smallHistory #[annotated
    (.resultExact sourceAtomOwner [canonicalSelfTerm sourceAtomOwner] .exactZero) 0]

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
  smallHistory #[annotated
    (.resultExact samplerAtomOwner [canonicalSelfTerm samplerAtomOwner] .exactZero) 0]

theorem source_atom_fixture
    (witness : Witness sourceAtomDocument sourceAtomHistory none 257) :
  DerivedResult sourceAtomDocument sourceAtomHistory none 257 witness
      sourceAtomOwner 0 := by
  apply ValueDerived.sourceAtom ⟨0⟩
  · refine ⟨0, ?_⟩
    rfl
  · refine ⟨?_, ?_, ?_⟩
    · refine ⟨⟨.resultExact sourceAtomOwner [canonicalSelfTerm sourceAtomOwner] .exactZero, 0⟩,
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
      samplerAtomOwner 0 := by
  apply ValueDerived.samplerAtom ⟨0⟩
  · refine ⟨0, ?_⟩
    rfl
  · refine ⟨?_, ?_, ?_⟩
    · refine ⟨⟨.resultExact samplerAtomOwner [canonicalSelfTerm samplerAtomOwner] .exactZero, 0⟩,
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
#print axioms statement_domain_fixture
#print axioms constructive_raw_bound_fixture

end Mxx.Certificate.OperationalNoise.TallSemanticsFixtures
