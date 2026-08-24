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

def toyContext : MonomialContext :=
  { exteriorCentral := [1], prefixFactors := [10], suffixFactors := [20] }

def toyRhsKeyA : MonomialKey := { centralFactors := [3], orderedFactors := [40] }
def toyRhsKeyB : MonomialKey := { centralFactors := [4], orderedFactors := [50] }
def toyRhs : Polynomial :=
  [ { coefficient := 3, key := toyRhsKeyA }, { coefficient := -1, key := toyRhsKeyB } ]

def toyContextualA : MonomialKey := toyContext.plug toyRhsKeyA
def toyContextualB : MonomialKey := toyContext.plug toyRhsKeyB
def toyReplacement : Polynomial := relationReplacement toyContext 2 toyRhs

def toyProductLeft : ExactTerm :=
  { coefficient := -2, key := { centralFactors := [], orderedFactors := [10] } }
def toyProductRight : ExactTerm :=
  { coefficient := 3, key := { centralFactors := [1, 3], orderedFactors := [40, 20] } }
def toyCancellation : Polynomial := [productMerge_contribution toyProductLeft toyProductRight]
def toyFolded : Polynomial := add toyReplacement toyCancellation
def toySubtracted : Polynomial := subtract toyReplacement toyCancellation

def toySurvivorMonomialActual : Nat := 4
def toySurvivorMonomialBound : Nat := 5
def toySurvivorCoefficient : Nat := (coefficient toyContextualB toyFolded).natAbs
def toyResolvedActual : Nat := toySurvivorCoefficient * toySurvivorMonomialActual
def toyResolvedBound : Nat := toySurvivorCoefficient * toySurvivorMonomialBound
def toyResolvedContributions : List Nat := [toyResolvedActual]
def toyResolvedBounds : List Nat := [toyResolvedBound]
def toySummaryActual : Nat := 7
def toySummaryBound : Nat := 9

theorem toy_contextual_sources :
    toyContextualA = toyContext.plug toyRhsKeyA ∧
      toyContextualB = toyContext.plug toyRhsKeyB := by
  exact ⟨rfl, rfl⟩

theorem toy_replacement_shape :
    toyReplacement =
      [ { coefficient := 6,
          key := { centralFactors := [1, 3], orderedFactors := [10, 40, 20] } },
        { coefficient := -2,
          key := { centralFactors := [1, 4], orderedFactors := [10, 50, 20] } } ] := by
  decide

theorem toy_product_cancellation :
    productMerge_contribution toyProductLeft toyProductRight =
      { coefficient := -6, key := toyContextualA } := by
  decide

theorem toy_add_cancellation : coefficient toyContextualA toyFolded = 0 := by decide

theorem toy_add_survivor : coefficient toyContextualB toyFolded = -2 := by decide

theorem toy_sub_contribution : coefficient toyContextualA toySubtracted = 12 := by decide

theorem toy_survivor_magnitude : toySurvivorCoefficient = 2 := by
  unfold toySurvivorCoefficient
  rw [show coefficient toyContextualB toyFolded = -2 from toy_add_survivor]
  decide

theorem toy_monomial_bound : toySurvivorMonomialActual ≤ toySurvivorMonomialBound := by decide

theorem toy_resolved_transfer : toyResolvedActual ≤ toyResolvedBound := by
  unfold toyResolvedActual toyResolvedBound
  exact boundTransfer_scale toy_monomial_bound

theorem toy_resolved_transfer_values :
    toyResolvedActual = 8 ∧ toyResolvedBound = 10 := by
  unfold toyResolvedActual toyResolvedBound toySurvivorCoefficient
  rw [toy_add_survivor]
  decide

theorem toy_resolved_transfer_list :
    List.Forall₂ (fun value bound => value ≤ bound)
      toyResolvedContributions toyResolvedBounds := by
  constructor
  · exact toy_resolved_transfer
  · exact List.Forall₂.nil

theorem toy_summary_bound : toySummaryActual ≤ toySummaryBound := by decide

theorem toy_prefold_invocation_end : 7 + 8 ≤ 9 + 10 := by
  have hfinal := preFold_to_invocationEnd
    (summaryActual := toySummaryActual) (summaryBound := toySummaryBound)
    (survivorContributions := toyResolvedContributions)
    (survivorBounds := toyResolvedBounds) toy_summary_bound toy_resolved_transfer_list
  change toySummaryActual + toyResolvedContributions.sum ≤ toySummaryBound + toyResolvedBounds.sum
  exact hfinal

theorem toy_event_replay :
    toyContextualA = toyContext.plug toyRhsKeyA ∧
      toyContextualB = toyContext.plug toyRhsKeyB ∧
      toyReplacement =
        [ { coefficient := 6,
            key := { centralFactors := [1, 3], orderedFactors := [10, 40, 20] } },
          { coefficient := -2,
            key := { centralFactors := [1, 4], orderedFactors := [10, 50, 20] } } ] ∧
      productMerge_contribution toyProductLeft toyProductRight =
        { coefficient := -6, key := toyContextualA } ∧
      coefficient toyContextualA toyFolded = 0 ∧
      coefficient toyContextualB toyFolded = -2 ∧
      coefficient toyContextualA toySubtracted = 12 ∧
      toySurvivorCoefficient = 2 ∧
      toySurvivorMonomialActual ≤ toySurvivorMonomialBound ∧
      (toyResolvedActual = 8 ∧ toyResolvedBound = 10) ∧
      toyResolvedActual ≤ toyResolvedBound ∧
      toySummaryActual ≤ toySummaryBound ∧
      7 + 8 ≤ 9 + 10 := by
  exact ⟨toy_contextual_sources.1, toy_contextual_sources.2, toy_replacement_shape,
    toy_product_cancellation, toy_add_cancellation, toy_add_survivor, toy_sub_contribution,
    toy_survivor_magnitude, toy_monomial_bound, toy_resolved_transfer_values,
    toy_resolved_transfer, toy_summary_bound, toy_prefold_invocation_end⟩

#print axioms toy_event_replay
#print axioms survivorFold_sound
#print axioms preFold_to_invocationEnd

end Mxx.Certificate.OperationalNoise.EventReplay
