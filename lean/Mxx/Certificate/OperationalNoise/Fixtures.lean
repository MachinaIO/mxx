import Mxx.Certificate.OperationalNoise.Core

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise

def shape9 : MatrixShape := { modulus := 9, ringDimension := 1, rows := 1, columns := 1 }

def zeroMatrix : Matrix := { shape := shape9, coefficients := [0] }
def sourceMatrix : Matrix := { shape := shape9, coefficients := [1] }
def familyMatrix0 : Matrix := { shape := shape9, coefficients := [1] }
def familyMatrix1 : Matrix := { shape := shape9, coefficients := [2] }

theorem zeroMatrix_valid : zeroMatrix.valid 9 1 = true := by rfl
theorem sourceMatrix_valid : sourceMatrix.valid 9 1 = true := by rfl
theorem familyMatrix0_valid : familyMatrix0.valid 9 1 = true := by rfl
theorem familyMatrix1_valid : familyMatrix1.valid 9 1 = true := by rfl

def singleton {α : Type} (id : Nat) (value : α) : RowTable α :=
  .node id value .empty .empty

def closedCert : Cert :=
  { plaintextModulus := 2
    ciphertextModulus := 9
    ringDimension := 1
    expressions := singleton 0 (.source { source := ⟨0⟩, selector := none })
    programs := .empty
    sources := singleton 0 { shape := shape9, coefficientBound := 2 }
    events := .empty
    residualRoot := .closed ⟨0⟩ }

def closedInputs : InputAssignment := fun _ => .matrix sourceMatrix
def closedSamplers : SamplerAssignment := fun _ => .invalid

theorem closed_wellFormed : closedCert.wellFormed = true := by rfl

def closedChecked : CheckedCert := { val := closedCert, valid := closed_wellFormed }

theorem closed_input_contract : InputContract closedCert closedInputs := by
  simp [InputContract, closedCert, singleton, RowTable.AllFrom, closedInputs, sourceMatrix,
    shape9, Matrix.valid, MatrixShape.valid, MatrixShape.matchesCertificate,
    MatrixShape.coefficientCount, Matrix.CoefficientBound, centeredNorm,
    centeredCoefficient]

theorem closed_sampler_contract : SamplerContract closedCert closedSamplers := by
  simp [SamplerContract, closedCert, RowTable.AllFrom]

theorem closed_eval :
    evalResidual 4 closedCert none closedSamplers closedInputs = some sourceMatrix := by
  simp [evalResidual, evalExpr, closedCert, singleton, closedInputs, sourceMatrix, shape9,
    Matrix.valid, MatrixShape.valid, MatrixShape.matchesCertificate,
    MatrixShape.coefficientCount, RowTable.lookup]

theorem closed_fallback_unreachable :
    evalResidual 4 closedCert none closedSamplers closedInputs ≠ none := by
  rw [closed_eval]
  simp

theorem source_bound_implies_norm_bound {matrix : Matrix}
    (hbound : matrix.CoefficientBound 9 2) : matrix.maxCenteredCoefficientNorm 9 ≤ 2 :=
  Matrix.maxCenteredCoefficientNorm_le_of_coefficientBound hbound

theorem closed_operational_proof : OperationalClaim closedChecked := by
  intro samplers inputs inputContract samplerContract
  have inputAt := inputContract
  simp [InputContract, closedChecked, closedCert, singleton, RowTable.AllFrom] at inputAt
  obtain ⟨matrix, assignment, valid, shape, bound⟩ := inputAt none
  have normBound := source_bound_implies_norm_bound bound
  change match evalResidual 4 closedCert none samplers inputs with
    | none => False
    | some matrix => 2 * closedCert.plaintextModulus * matrix.maxCenteredCoefficientNorm 9 <
        closedCert.ciphertextModulus
  have evalResult : evalResidual 4 closedCert none samplers inputs = some matrix := by
    simp [evalResidual, evalExpr, closedCert, singleton, SourceAccess.withSelector, assignment,
      RowTable.lookup]
    simp [valid]
  rw [evalResult]
  simp only [closedCert]
  omega

def familyCert : Cert :=
  { plaintextModulus := 2
    ciphertextModulus := 9
    ringDimension := 1
    expressions := singleton 0 (.sampler { event := ⟨0⟩, selector := none })
    programs := singleton 0 { body := ⟨0⟩ }
    sources := .empty
    events := singleton 0 { shape := shape9, coefficientBound := 2 }
    residualRoot := .family ⟨0⟩ { lower := 0, upper := 2 } }

def familyInputs : InputAssignment := fun _ => .invalid

def familySamplers : SamplerAssignment := fun access =>
  match access.event.value, access.selector with
  | 0, some 0 => .matrix familyMatrix0
  | 0, some 1 => .matrix familyMatrix1
  | _, _ => .matrix zeroMatrix

theorem family_wellFormed : familyCert.wellFormed = true := by rfl

def familyChecked : CheckedCert := { val := familyCert, valid := family_wellFormed }

theorem family_sampler_assignments_differ :
    familySamplers { event := ⟨0⟩, selector := some 0 } ≠
      familySamplers { event := ⟨0⟩, selector := some 1 } := by decide

theorem family_input_contract : InputContract familyCert familyInputs := by
  simp [InputContract, familyCert, RowTable.AllFrom]

theorem family_sampler_contract : SamplerContract familyCert familySamplers := by
  simp only [SamplerContract, familyCert, singleton, RowTable.AllFrom]
  refine ⟨?_, trivial, trivial⟩
  intro selector
  cases selector with
  | none =>
      exact ⟨zeroMatrix, by rfl, zeroMatrix_valid, rfl,
        by simp [zeroMatrix, Matrix.CoefficientBound, centeredNorm, centeredCoefficient]⟩
  | some value =>
      cases value with
      | zero =>
          exact ⟨familyMatrix0, by rfl, familyMatrix0_valid, rfl,
            by simp [familyMatrix0, Matrix.CoefficientBound, centeredNorm, centeredCoefficient]⟩
      | succ value =>
          cases value with
          | zero =>
              exact ⟨familyMatrix1, by rfl, familyMatrix1_valid, rfl,
                by simp [familyMatrix1, Matrix.CoefficientBound, centeredNorm, centeredCoefficient]⟩
          | succ value =>
              exact ⟨zeroMatrix, by rfl, zeroMatrix_valid, rfl,
                by simp [zeroMatrix, Matrix.CoefficientBound, centeredNorm, centeredCoefficient]⟩

theorem family_operational_proof : OperationalClaim familyChecked := by
  intro samplers inputs inputContract samplerContract selector selectorInDomain
  have samplerAt := samplerContract
  simp [SamplerContract, familyChecked, familyCert, singleton, RowTable.AllFrom] at samplerAt
  obtain ⟨matrix, assignment, valid, shape, bound⟩ := samplerAt (some selector)
  have normBound : matrix.maxCenteredCoefficientNorm 9 ≤ 2 := source_bound_implies_norm_bound bound
  have modulusBound : 2 * 2 * matrix.maxCenteredCoefficientNorm 9 < 9 := by omega
  change match evalResidual 4 familyCert (some selector) samplers inputs with
    | none => False
    | some result => 2 * familyCert.plaintextModulus * result.maxCenteredCoefficientNorm 9 <
        familyCert.ciphertextModulus
  have evalResult : evalResidual 4 familyCert (some selector) samplers inputs = some matrix := by
    simp [evalResidual, evalProgram, evalExpr, familyCert, singleton,
      EventAccess.withSelector, assignment, RowTable.lookup]
    simp [valid]
  rw [evalResult]
  exact modulusBound

theorem strict_bound_example : 2 * 2 * 2 < 9 := by decide
theorem equality_boundary_rejected : ¬ (2 * 2 * 2 < 8) := by decide

def modulusEightMatrix : Matrix := { shape := shape9, coefficients := [8] }

theorem centered_wraps : modulusEightMatrix.maxCenteredCoefficientNorm 9 = 1 := by decide

def zeroModulusCert : Cert := { closedCert with ciphertextModulus := 0 }
def emptyDomainCert : Cert := { familyCert with residualRoot := .family ⟨0⟩ { lower := 0, upper := 0 } }
def danglingRootCert : Cert := { closedCert with residualRoot := .closed ⟨99⟩ }
def wrongSourceModulusCert : Cert :=
  { closedCert with sources := singleton 0 { shape := { shape9 with modulus := 8 }, coefficientBound := 2 } }
def wrongSourceRingDimensionCert : Cert :=
  { closedCert with sources := singleton 0 { shape := { shape9 with ringDimension := 2 }, coefficientBound := 2 } }
def oversizedFamilyCert : Cert :=
  { familyCert with
      residualRoot := .family ⟨0⟩ { lower := 0, upper := 18446744073709551616 } }

theorem zero_modulus_rejected : zeroModulusCert.wellFormed = false := by rfl
theorem empty_domain_rejected : emptyDomainCert.wellFormed = false := by rfl
theorem dangling_root_rejected : danglingRootCert.wellFormed = false := by rfl
theorem wrong_source_modulus_rejected : wrongSourceModulusCert.wellFormed = false := by rfl
theorem wrong_source_ring_dimension_rejected : wrongSourceRingDimensionCert.wellFormed = false := by rfl
theorem oversized_family_rejected : oversizedFamilyCert.wellFormed = false := by rfl

theorem malformed_certs_not_valid :
    ¬ zeroModulusCert.Valid ∧ ¬ emptyDomainCert.Valid ∧ ¬ danglingRootCert.Valid ∧
      ¬ wrongSourceModulusCert.Valid ∧ ¬ wrongSourceRingDimensionCert.Valid ∧
      ¬ oversizedFamilyCert.Valid := by
  refine ⟨?_, ?_⟩
  · intro h
    change zeroModulusCert.wellFormed = true at h
    rw [zero_modulus_rejected] at h
    contradiction
  refine ⟨?_, ?_⟩
  · intro h
    change emptyDomainCert.wellFormed = true at h
    rw [empty_domain_rejected] at h
    contradiction
  refine ⟨?_, ?_⟩
  · intro h
    change danglingRootCert.wellFormed = true at h
    rw [dangling_root_rejected] at h
    contradiction
  refine ⟨?_, ?_⟩
  · intro h
    change wrongSourceModulusCert.wellFormed = true at h
    rw [wrong_source_modulus_rejected] at h
    contradiction
  refine ⟨?_, ?_⟩
  · intro h
    change wrongSourceRingDimensionCert.wellFormed = true at h
    rw [wrong_source_ring_dimension_rejected] at h
    contradiction
  intro h
  change oversizedFamilyCert.wellFormed = true at h
  rw [oversized_family_rejected] at h
  contradiction

#print axioms closed_wellFormed
#print axioms closed_operational_proof
#print axioms family_operational_proof

end Mxx.Certificate.OperationalNoise

namespace Mxx.Certificate.OperationalNoise.EventReplay

open Mxx.Certificate.OperationalNoise

def exampleContext : MonomialContext :=
  { exteriorCentral := [1], prefixFactors := [10], suffixFactors := [20] }

def exampleRhsKeyA : MonomialKey := { centralFactors := [3], orderedFactors := [40] }
def exampleRhsKeyB : MonomialKey := { centralFactors := [4], orderedFactors := [50] }
def exampleRhs : Polynomial :=
  [ { coefficient := 3, key := exampleRhsKeyA }, { coefficient := -1, key := exampleRhsKeyB } ]

def exampleContextualA : MonomialKey := exampleContext.plug exampleRhsKeyA
def exampleContextualB : MonomialKey := exampleContext.plug exampleRhsKeyB
def exampleReplacement : Polynomial := relationReplacement exampleContext 2 exampleRhs

def exampleProductLeft : ExactTerm :=
  { coefficient := -2, key := { centralFactors := [], orderedFactors := [10] } }
def exampleProductRight : ExactTerm :=
  { coefficient := 3, key := { centralFactors := [1, 3], orderedFactors := [40, 20] } }
def exampleCancellation : Polynomial := [productMerge_contribution exampleProductLeft exampleProductRight]
def exampleFolded : Polynomial := add exampleReplacement exampleCancellation
def exampleSubtracted : Polynomial := subtract exampleReplacement exampleCancellation

def exampleSurvivorMonomialActual : Nat := 4
def exampleSurvivorMonomialBound : Nat := 5
def exampleSurvivorCoefficient : Nat := (coefficient exampleContextualB exampleFolded).natAbs
def exampleResolvedActual : Nat := exampleSurvivorCoefficient * exampleSurvivorMonomialActual
def exampleResolvedBound : Nat := exampleSurvivorCoefficient * exampleSurvivorMonomialBound
def exampleResolvedContributions : List Nat := [exampleResolvedActual]
def exampleResolvedBounds : List Nat := [exampleResolvedBound]
def exampleSummaryActual : Nat := 7
def exampleSummaryBound : Nat := 9

theorem example_contextual_sources :
    exampleContextualA = exampleContext.plug exampleRhsKeyA ∧
      exampleContextualB = exampleContext.plug exampleRhsKeyB := by
  exact ⟨rfl, rfl⟩

theorem example_replacement_shape :
    exampleReplacement =
      [ { coefficient := 6,
          key := { centralFactors := [1, 3], orderedFactors := [10, 40, 20] } },
        { coefficient := -2,
          key := { centralFactors := [1, 4], orderedFactors := [10, 50, 20] } } ] := by
  decide

theorem example_product_cancellation :
    productMerge_contribution exampleProductLeft exampleProductRight =
      { coefficient := -6, key := exampleContextualA } := by
  decide

theorem example_add_cancellation : coefficient exampleContextualA exampleFolded = 0 := by decide

theorem example_add_survivor : coefficient exampleContextualB exampleFolded = -2 := by decide

theorem example_sub_contribution : coefficient exampleContextualA exampleSubtracted = 12 := by decide

theorem example_survivor_magnitude : exampleSurvivorCoefficient = 2 := by
  unfold exampleSurvivorCoefficient
  rw [show coefficient exampleContextualB exampleFolded = -2 from example_add_survivor]
  decide

theorem example_monomial_bound : exampleSurvivorMonomialActual ≤ exampleSurvivorMonomialBound := by decide

theorem example_resolved_transfer : exampleResolvedActual ≤ exampleResolvedBound := by
  unfold exampleResolvedActual exampleResolvedBound
  exact boundTransfer_scale example_monomial_bound

theorem example_resolved_transfer_values :
    exampleResolvedActual = 8 ∧ exampleResolvedBound = 10 := by
  unfold exampleResolvedActual exampleResolvedBound exampleSurvivorCoefficient
  rw [example_add_survivor]
  decide

theorem example_resolved_transfer_list :
    List.Forall₂ (fun value bound => value ≤ bound)
      exampleResolvedContributions exampleResolvedBounds := by
  constructor
  · exact example_resolved_transfer
  · exact List.Forall₂.nil

theorem example_summary_bound : exampleSummaryActual ≤ exampleSummaryBound := by decide

theorem example_prefold_invocation_end : 7 + 8 ≤ 9 + 10 := by
  have hfinal := preFold_to_invocationEnd
    (summaryActual := exampleSummaryActual) (summaryBound := exampleSummaryBound)
    (survivorContributions := exampleResolvedContributions)
    (survivorBounds := exampleResolvedBounds) example_summary_bound example_resolved_transfer_list
  change exampleSummaryActual + exampleResolvedContributions.sum ≤
    exampleSummaryBound + exampleResolvedBounds.sum
  exact hfinal

theorem example_event_replay :
    exampleContextualA = exampleContext.plug exampleRhsKeyA ∧
      exampleContextualB = exampleContext.plug exampleRhsKeyB ∧
      exampleReplacement =
        [ { coefficient := 6,
            key := { centralFactors := [1, 3], orderedFactors := [10, 40, 20] } },
          { coefficient := -2,
            key := { centralFactors := [1, 4], orderedFactors := [10, 50, 20] } } ] ∧
      productMerge_contribution exampleProductLeft exampleProductRight =
        { coefficient := -6, key := exampleContextualA } ∧
      coefficient exampleContextualA exampleFolded = 0 ∧
      coefficient exampleContextualB exampleFolded = -2 ∧
      coefficient exampleContextualA exampleSubtracted = 12 ∧
      exampleSurvivorCoefficient = 2 ∧
      exampleSurvivorMonomialActual ≤ exampleSurvivorMonomialBound ∧
      (exampleResolvedActual = 8 ∧ exampleResolvedBound = 10) ∧
      exampleResolvedActual ≤ exampleResolvedBound ∧
      exampleSummaryActual ≤ exampleSummaryBound ∧
      7 + 8 ≤ 9 + 10 := by
  exact ⟨example_contextual_sources.1, example_contextual_sources.2, example_replacement_shape,
    example_product_cancellation, example_add_cancellation, example_add_survivor,
    example_sub_contribution, example_survivor_magnitude, example_monomial_bound,
    example_resolved_transfer_values, example_resolved_transfer, example_summary_bound,
    example_prefold_invocation_end⟩

#print axioms example_event_replay
#print axioms survivorFold_sound
#print axioms preFold_to_invocationEnd

end Mxx.Certificate.OperationalNoise.EventReplay
