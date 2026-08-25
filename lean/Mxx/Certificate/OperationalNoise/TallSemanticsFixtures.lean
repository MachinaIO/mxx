import Mxx.Certificate.OperationalNoise.TallSemantics

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSemanticsFixtures

open Mxx.Certificate.OperationalNoise
open SchemaV1
open TallSecurity0ABI
open TallSemantics

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
  refine ⟨1, ?_, ?_⟩
  · decide
  · simp [boundInterprets, centeredNorm, centeredCoefficient]

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

#print axioms generic_evaluation_fixture
#print axioms generic_value_claim_fixture
#print axioms statement_domain_fixture
#print axioms constructive_raw_bound_fixture

end Mxx.Certificate.OperationalNoise.TallSemanticsFixtures
