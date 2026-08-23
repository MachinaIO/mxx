import Mxx.Certificate.OperationalNoise.Core

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise

def shape9 : MatrixShape := { modulus := 9, ringDimension := 1, rows := 1, columns := 1 }

def zeroMatrix : Matrix := { shape := shape9, coefficients := [0] }
def sourceMatrix : Matrix := { shape := shape9, coefficients := [1] }
def familyMatrix0 : Matrix := { shape := shape9, coefficients := [1] }
def familyMatrix1 : Matrix := { shape := shape9, coefficients := [2] }

theorem zeroMatrix_valid : zeroMatrix.valid = true := by rfl
theorem sourceMatrix_valid : sourceMatrix.valid = true := by rfl
theorem familyMatrix0_valid : familyMatrix0.valid = true := by rfl
theorem familyMatrix1_valid : familyMatrix1.valid = true := by rfl

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
    shape9, Matrix.valid, MatrixShape.valid, MatrixShape.coefficientCount,
    Matrix.CoefficientBound, centeredCoefficient]

theorem closed_sampler_contract : SamplerContract closedCert closedSamplers := by
  simp [SamplerContract, closedCert, RowTable.AllFrom]

theorem closed_eval :
    evalResidual 4 closedCert none closedSamplers closedInputs = some sourceMatrix := by
  simp [evalResidual, evalExpr, closedCert, singleton, closedInputs, sourceMatrix, shape9,
    Matrix.valid, MatrixShape.valid, MatrixShape.coefficientCount, RowTable.lookup]

theorem closed_fallback_unreachable :
    evalResidual 4 closedCert none closedSamplers closedInputs ≠ none := by
  rw [closed_eval]
  simp

theorem source_bound_implies_norm_bound {matrix : Matrix}
    (hbound : matrix.CoefficientBound 2) : matrix.maxCenteredCoefficientNorm ≤ 2 :=
  Matrix.maxCenteredCoefficientNorm_le_of_coefficientBound hbound

theorem closed_operational_proof : OperationalClaim closedChecked := by
  intro samplers inputs inputContract samplerContract
  have inputAt := inputContract
  simp [InputContract, closedChecked, closedCert, singleton, RowTable.AllFrom] at inputAt
  obtain ⟨matrix, assignment, valid, shape, bound⟩ := inputAt none
  have normBound := source_bound_implies_norm_bound bound
  change match evalResidual 4 closedCert none samplers inputs with
    | none => False
    | some matrix => 2 * closedCert.plaintextModulus * matrix.maxCenteredCoefficientNorm <
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
      exact ⟨zeroMatrix, by rfl, zeroMatrix_valid, rfl, by simp [zeroMatrix, Matrix.CoefficientBound, centeredCoefficient]⟩
  | some value =>
      cases value with
      | zero =>
          exact ⟨familyMatrix0, by rfl, familyMatrix0_valid, rfl,
            by simp [familyMatrix0, Matrix.CoefficientBound, centeredCoefficient]⟩
      | succ value =>
          cases value with
          | zero =>
              exact ⟨familyMatrix1, by rfl, familyMatrix1_valid, rfl,
                by simp [familyMatrix1, Matrix.CoefficientBound, centeredCoefficient]⟩
          | succ value =>
              exact ⟨zeroMatrix, by rfl, zeroMatrix_valid, rfl,
                by simp [zeroMatrix, Matrix.CoefficientBound, centeredCoefficient]⟩

theorem family_operational_proof : OperationalClaim familyChecked := by
  intro samplers inputs inputContract samplerContract selector selectorInDomain
  have samplerAt := samplerContract
  simp [SamplerContract, familyChecked, familyCert, singleton, RowTable.AllFrom] at samplerAt
  obtain ⟨matrix, assignment, valid, shape, bound⟩ := samplerAt (some selector)
  have normBound : matrix.maxCenteredCoefficientNorm ≤ 2 := source_bound_implies_norm_bound bound
  have modulusBound : 2 * 2 * matrix.maxCenteredCoefficientNorm < 9 := by omega
  change match evalResidual 4 familyCert (some selector) samplers inputs with
    | none => False
    | some result => 2 * familyCert.plaintextModulus * result.maxCenteredCoefficientNorm <
        familyCert.ciphertextModulus
  have evalResult : evalResidual 4 familyCert (some selector) samplers inputs = some matrix := by
    simp [evalResidual, evalProgram, evalExpr, familyCert, singleton,
      EventAccess.withSelector, assignment, RowTable.lookup]
    simp [valid]
  rw [evalResult]
  exact modulusBound

theorem strict_bound_example : 2 * 2 * 2 < 9 := by decide
theorem equality_boundary_rejected : ¬ (2 * 2 * 2 < 8) := by decide

def zeroModulusCert : Cert := { closedCert with ciphertextModulus := 0 }
def emptyDomainCert : Cert := { familyCert with residualRoot := .family ⟨0⟩ { lower := 0, upper := 0 } }
def danglingRootCert : Cert := { closedCert with residualRoot := .closed ⟨99⟩ }

theorem zero_modulus_rejected : zeroModulusCert.wellFormed = false := by rfl
theorem empty_domain_rejected : emptyDomainCert.wellFormed = false := by rfl
theorem dangling_root_rejected : danglingRootCert.wellFormed = false := by rfl

theorem malformed_certs_not_valid :
    ¬ zeroModulusCert.Valid ∧ ¬ emptyDomainCert.Valid ∧ ¬ danglingRootCert.Valid := by
  constructor
  · intro h
    change zeroModulusCert.wellFormed = true at h
    rw [zero_modulus_rejected] at h
    contradiction
  constructor
  · intro h
    change emptyDomainCert.wellFormed = true at h
    rw [empty_domain_rejected] at h
    contradiction
  · intro h
    change danglingRootCert.wellFormed = true at h
    rw [dangling_root_rejected] at h
    contradiction

#print axioms closed_wellFormed
#print axioms closed_operational_proof
#print axioms family_operational_proof

end Mxx.Certificate.OperationalNoise
