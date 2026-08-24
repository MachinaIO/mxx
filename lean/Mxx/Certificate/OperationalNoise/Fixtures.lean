import Mxx.Certificate.OperationalNoise.Core
import Mxx.Certificate.OperationalNoise.Replay

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

/-! The event-replay lane keeps only the finite algebra needed by the G0 proof.
    Central factors are canonicalized as an ordered multiset; ordered factors retain
    their concatenation order. -/

def insertCentral (factor : Nat) : List Nat → List Nat
  | [] => [factor]
  | head :: tail =>
      if factor ≤ head then factor :: head :: tail else head :: insertCentral factor tail

def canonicalCentral : List Nat → List Nat
  | [] => []
  | head :: tail => insertCentral head (canonicalCentral tail)

theorem canonicalCentral_nil : canonicalCentral [] = [] := by rfl

theorem insertCentral_mem (factor : Nat) (factors : List Nat) :
    factor ∈ insertCentral factor factors := by
  induction factors with
  | nil => simp [insertCentral]
  | cons head tail ih =>
      by_cases h : factor ≤ head
      · simp [insertCentral, h]
      · simp [insertCentral, h, ih]

theorem canonicalCentral_mem (factor : Nat) (factors : List Nat) :
    factor ∈ canonicalCentral (factor :: factors) := by
  simp [canonicalCentral, insertCentral_mem]

def MonomialKey.product (left right : MonomialKey) : MonomialKey :=
  { centralFactors := canonicalCentral (left.centralFactors ++ right.centralFactors)
    orderedFactors := left.orderedFactors ++ right.orderedFactors }

theorem product_central (left right : MonomialKey) :
    (MonomialKey.product left right).centralFactors =
      canonicalCentral (left.centralFactors ++ right.centralFactors) := by rfl

theorem product_ordered (left right : MonomialKey) :
    (MonomialKey.product left right).orderedFactors =
      left.orderedFactors ++ right.orderedFactors := by rfl

structure MonomialContext where
  exteriorCentral : List Nat
  prefixFactors : List Nat
  suffixFactors : List Nat
deriving DecidableEq, Repr

def MonomialContext.plug (context : MonomialContext) (key : MonomialKey) : MonomialKey :=
  { centralFactors := canonicalCentral (context.exteriorCentral ++ key.centralFactors)
    orderedFactors := context.prefixFactors ++ key.orderedFactors ++ context.suffixFactors }

theorem context_plug_central (context : MonomialContext) (key : MonomialKey) :
    (context.plug key).centralFactors =
      canonicalCentral (context.exteriorCentral ++ key.centralFactors) := by
  rfl

theorem context_plug_ordered (context : MonomialContext) (key : MonomialKey) :
    (context.plug key).orderedFactors =
      context.prefixFactors ++ key.orderedFactors ++ context.suffixFactors := by
  rfl

def scalePolynomial (scalar : Int) (polynomial : Polynomial) : Polynomial :=
  polynomial.map (fun term => { term with coefficient := scalar * term.coefficient })

def contextualize (context : MonomialContext) (polynomial : Polynomial) : Polynomial :=
  polynomial.map (fun term => { term with key := context.plug term.key })

def relationReplacement (context : MonomialContext) (outerCoefficient : Int)
    (rhs : Polynomial) : Polynomial :=
  scalePolynomial outerCoefficient (contextualize context rhs)

theorem relationReplacement_coefficient (context : MonomialContext) (outerCoefficient : Int)
    (rhs : Polynomial) (key : MonomialKey) :
    coefficient key (relationReplacement context outerCoefficient rhs) =
      coefficient key (scalePolynomial outerCoefficient (contextualize context rhs)) := by
  rfl

theorem coefficient_add_replay (key : MonomialKey) (left right : Polynomial) :
    coefficient key (add left right) = coefficient key left + coefficient key right :=
  coefficient_add key left right

theorem coefficient_subtract_replay (key : MonomialKey) (left right : Polynomial) :
    coefficient key (subtract left right) = coefficient key left - coefficient key right :=
  coefficient_subtract key left right

def productMerge_contribution (left right : ExactTerm) : ExactTerm :=
  { coefficient := left.coefficient * right.coefficient
    key := MonomialKey.product left.key right.key }

theorem productMerge_contribution_coefficient (left right : ExactTerm) :
    (productMerge_contribution left right).coefficient = left.coefficient * right.coefficient := by
  rfl

theorem productMerge_contribution_key (left right : ExactTerm) :
    (productMerge_contribution left right).key = MonomialKey.product left.key right.key := by
  rfl

def sumBound (left right : Nat) : Nat := left + right
def scaleBound (scalar bound : Nat) : Nat := scalar * bound
def productBound (left right : Nat) : Nat := left * right

structure BoundTransfer where
  source : Nat
  target : Nat
  proof : source ≤ target
deriving Repr

def BoundTransfer.sum (left right : Nat) : BoundTransfer :=
  { source := sumBound left right, target := sumBound left right, proof := Nat.le_refl _ }

def BoundTransfer.scale (scalar bound : Nat) : BoundTransfer :=
  { source := scaleBound scalar bound, target := scaleBound scalar bound, proof := Nat.le_refl _ }

def BoundTransfer.product (left right : Nat) : BoundTransfer :=
  { source := productBound left right, target := productBound left right, proof := Nat.le_refl _ }

theorem boundTransfer_sum (left right : Nat) :
    (BoundTransfer.sum left right).source = left + right := by rfl

theorem boundTransfer_scale (scalar bound : Nat) :
    (BoundTransfer.scale scalar bound).source = scalar * bound := by rfl

theorem boundTransfer_product (left right : Nat) :
    (BoundTransfer.product left right).source = left * right := by rfl

theorem boundTransfer_zero_product (bound : Nat) :
    productBound 0 bound = 0 := by simp [productBound]

theorem boundTransfer_product_zero (bound : Nat) :
    productBound bound 0 = 0 := by simp [productBound]

structure SurvivorFold where
  coefficient : Int
  bound : Nat
deriving DecidableEq, Repr

def survivorContribution (fold : SurvivorFold) : Nat := fold.coefficient.natAbs

def survivorFold : List SurvivorFold → Nat
  | [] => 0
  | fold :: folds => survivorContribution fold + survivorFold folds

def survivorBounds : List SurvivorFold → List Nat
  | [] => []
  | fold :: folds => fold.bound :: survivorBounds folds

def listSum : List Nat → Nat
  | [] => 0
  | value :: values => value + listSum values

theorem survivorBounds_length (folds : List SurvivorFold) :
    (survivorBounds folds).length = folds.length := by
  induction folds with
  | nil => rfl
  | cons fold folds ih => simp [survivorBounds, ih]

inductive Forall₂ {α β : Type} (relation : α → β → Prop) : List α → List β → Prop
  | nil : Forall₂ relation [] []
  | cons {head tail bound bounds} :
      relation head bound → Forall₂ relation tail bounds →
        Forall₂ relation (head :: tail) (bound :: bounds)

theorem survivorFold_sound {folds : List SurvivorFold}
    (hbound : Forall₂ (fun fold bound => survivorContribution fold ≤ bound) folds
      (survivorBounds folds)) :
    survivorFold folds ≤ listSum (survivorBounds folds) := by
  induction folds with
  | nil => simp [survivorFold, survivorBounds, listSum]
  | cons fold folds ih =>
      cases hbound with
      | cons head tail =>
          simp only [survivorFold, survivorBounds, listSum]
          exact Nat.add_le_add head (ih tail)

structure PreFoldPolynomial where
  polynomial : Polynomial
  bound : Nat
deriving Repr

structure InvocationEnd where
  polynomial : Polynomial
  bound : Nat
deriving Repr

def preFold_to_invocationEnd (preFold : PreFoldPolynomial) : InvocationEnd :=
  { polynomial := preFold.polynomial, bound := preFold.bound }

theorem preFold_to_invocationEnd_polynomial (preFold : PreFoldPolynomial) :
    (preFold_to_invocationEnd preFold).polynomial = preFold.polynomial := by rfl

theorem preFold_to_invocationEnd_bound (preFold : PreFoldPolynomial) :
    (preFold_to_invocationEnd preFold).bound = preFold.bound := by rfl

def toyContext : MonomialContext :=
  { exteriorCentral := [], prefixFactors := [10], suffixFactors := [20] }

def toyLhs : MonomialKey := { centralFactors := [], orderedFactors := [2] }
def toyRhsKeyA : MonomialKey := { centralFactors := [], orderedFactors := [3] }
def toyRhsKeyB : MonomialKey := { centralFactors := [], orderedFactors := [4] }

def toyRhs : Polynomial :=
  [ { coefficient := 3, key := toyRhsKeyA }, { coefficient := -1, key := toyRhsKeyB } ]

def toyReplacement : Polynomial := relationReplacement toyContext 2 toyRhs

def toyPositive : Polynomial := [{ coefficient := 5, key := toyLhs }]
def toyNegative : Polynomial := [{ coefficient := -2, key := toyLhs }]
def toyProductLeft : ExactTerm := { coefficient := -2, key := toyLhs }
def toyProductRight : ExactTerm := { coefficient := 3, key := toyRhsKeyA }
def toyProduct : ExactTerm := productMerge_contribution toyProductLeft toyProductRight

def toyFolds : List SurvivorFold :=
  [ { coefficient := 3, bound := 3 }, { coefficient := -1, bound := 1 } ]

def toyPreFold : PreFoldPolynomial := { polynomial := toyRhs, bound := 4 }

theorem toy_replacement_shape :
    toyReplacement =
      [ { coefficient := 6, key := { centralFactors := [], orderedFactors := [10, 3, 20] } },
        { coefficient := -2,
          key := { centralFactors := [], orderedFactors := [10, 4, 20] } } ] := by
  decide

theorem toy_add_value : coefficient toyLhs (add toyPositive toyNegative) = 3 := by decide

theorem toy_sub_value : coefficient toyLhs (subtract toyPositive toyNegative) = 7 := by decide

theorem toy_product_value :
    toyProduct.coefficient = -6 ∧ toyProduct.key.orderedFactors = [2, 3] := by decide

theorem toy_survivor_fold_value : survivorFold toyFolds = 4 := by decide

theorem toy_survivor_bound :
    Forall₂ (fun fold bound => survivorContribution fold ≤ bound) toyFolds
      (survivorBounds toyFolds) := by
  constructor
  · decide
  · constructor
    · decide
    · exact Forall₂.nil

theorem toy_survivor_fold_exact :
    survivorFold toyFolds = listSum (survivorBounds toyFolds) := by
  have hbound := survivorFold_sound (folds := toyFolds) toy_survivor_bound
  decide

theorem toy_prefold_invocation_end :
    (preFold_to_invocationEnd toyPreFold).polynomial = toyRhs ∧
      (preFold_to_invocationEnd toyPreFold).bound = 4 := by
  decide

theorem toy_event_replay :
    toyReplacement =
        [ { coefficient := 6, key := { centralFactors := [], orderedFactors := [10, 3, 20] } },
          { coefficient := -2,
            key := { centralFactors := [], orderedFactors := [10, 4, 20] } } ] ∧
      coefficient toyLhs (add toyPositive toyNegative) = 3 ∧
      coefficient toyLhs (subtract toyPositive toyNegative) = 7 ∧
      toyProduct.coefficient = -6 ∧
      survivorFold toyFolds = 4 ∧
      survivorFold toyFolds = listSum (survivorBounds toyFolds) ∧
      (preFold_to_invocationEnd toyPreFold).polynomial = toyRhs := by
  exact ⟨toy_replacement_shape, toy_add_value, toy_sub_value, toy_product_value.1,
    toy_survivor_fold_value, toy_survivor_fold_exact, toy_prefold_invocation_end.1⟩

#print axioms toy_event_replay
#print axioms survivorFold_sound
#print axioms preFold_to_invocationEnd

end Mxx.Certificate.OperationalNoise.EventReplay
