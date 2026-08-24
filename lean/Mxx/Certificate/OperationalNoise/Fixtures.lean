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

def toyLhs : MonomialKey := { centralFactors := [2], orderedFactors := [30] }
def toySource : MonomialKey := toyContext.plug toyLhs
def toyRhsKeyA : MonomialKey := { centralFactors := [3], orderedFactors := [40] }
def toyRhsKeyB : MonomialKey := { centralFactors := [4], orderedFactors := [50] }

def toyRhs : Polynomial :=
  [ { coefficient := 3, key := toyRhsKeyA }, { coefficient := -1, key := toyRhsKeyB } ]

def toyReplacement : Polynomial := relationReplacement toyContext 2 toyRhs

def toyPositive : Polynomial := [{ coefficient := 5, key := toyLhs }]
def toyNegative : Polynomial := [{ coefficient := -2, key := toyLhs }]
def toyCancelPositive : Polynomial := [{ coefficient := 5, key := toyLhs }]
def toyCancelNegative : Polynomial := [{ coefficient := -5, key := toyLhs }]
def toyProductLeft : ExactTerm := { coefficient := -2, key := toyLhs }
def toyProductRight : ExactTerm := { coefficient := 3, key := toyRhsKeyA }
def toyProduct : ExactTerm := productMerge_contribution toyProductLeft toyProductRight

def toyContributions : List Nat := [3, 1]
def toyResolvedBounds : List Nat := [5, 2]
def toyPreFoldSummary : Nat := 7
def toyPostFoldSummary : Nat := 9
def toyFinalSummary : Nat := toyPostFoldSummary + toyResolvedBounds.sum

theorem toy_context_source : toySource = toyContext.plug toyLhs := by rfl

theorem toy_replacement_shape :
    toyReplacement =
      [ { coefficient := 6,
          key := { centralFactors := [1, 3], orderedFactors := [10, 40, 20] } },
        { coefficient := -2,
          key := { centralFactors := [1, 4], orderedFactors := [10, 50, 20] } } ] := by
  decide

theorem toy_add_value : coefficient toyLhs (add toyPositive toyNegative) = 3 := by decide

theorem toy_sub_value : coefficient toyLhs (subtract toyPositive toyNegative) = 7 := by decide

theorem toy_exact_cancellation :
    coefficient toyLhs (add toyCancelPositive toyCancelNegative) = 0 := by decide

theorem toy_product_value :
    toyProduct.coefficient = -6 ∧ toyProduct.key.orderedFactors = [30, 40] := by decide

theorem toy_survivor_transfer :
    List.Forall₂ (fun value bound => value ≤ bound) toyContributions toyResolvedBounds := by
  constructor
  · decide
  · constructor
    · decide
    · exact List.Forall₂.nil

theorem toy_survivor_fold_exact :
    toyContributions.sum = 4 ∧ toyResolvedBounds.sum = 7 := by decide

theorem toy_bound_transfer_chain :
    3 + 1 ≤ 5 + 2 ∧ 2 * 3 ≤ 2 * 5 ∧ 3 * 1 ≤ 5 * 2 := by
  exact ⟨boundTransfer_sum (by decide) (by decide),
    boundTransfer_scale (by decide), boundTransfer_product (by decide) (by decide)⟩

theorem toy_prefold_invocation_end :
    toyPreFoldSummary + toyContributions.sum ≤ toyPostFoldSummary + toyResolvedBounds.sum := by
  exact preFold_to_invocationEnd (by decide) toy_survivor_transfer

theorem toy_event_replay :
    toySource = toyContext.plug toyLhs ∧
      toyReplacement =
        [ { coefficient := 6,
            key := { centralFactors := [1, 3], orderedFactors := [10, 40, 20] } },
          { coefficient := -2,
            key := { centralFactors := [1, 4], orderedFactors := [10, 50, 20] } } ] ∧
      coefficient toyLhs (add toyPositive toyNegative) = 3 ∧
      coefficient toyLhs (subtract toyPositive toyNegative) = 7 ∧
      coefficient toyLhs (add toyCancelPositive toyCancelNegative) = 0 ∧
      (3 + 1 ≤ 5 + 2 ∧ 2 * 3 ≤ 2 * 5 ∧ 3 * 1 ≤ 5 * 2) ∧
      toyProduct.coefficient = -6 ∧
      toyContributions.sum = 4 ∧
      toyResolvedBounds.sum = 7 ∧
      toyFinalSummary = 16 ∧
      toyPreFoldSummary + toyContributions.sum ≤
        toyPostFoldSummary + toyResolvedBounds.sum := by
  exact ⟨toy_context_source, toy_replacement_shape, toy_add_value, toy_sub_value,
    toy_exact_cancellation, toy_bound_transfer_chain, toy_product_value.1,
    toy_survivor_fold_exact.1,
    toy_survivor_fold_exact.2, by decide, toy_prefold_invocation_end⟩

#print axioms toy_event_replay
#print axioms survivorFold_sound
#print axioms preFold_to_invocationEnd

end Mxx.Certificate.OperationalNoise.EventReplay
