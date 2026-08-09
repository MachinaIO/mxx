/-
Deferred proof source retained for the later end-to-end correctness milestone.

This implementation targets the superseded pre-flat operational fact shape and is intentionally
commented out rather than imported by the executable operational checker. Re-enable it only after
porting the semantic interface to the current flat `OperationalMatrixFact` representation.

import Mxx.Certificate.OperationalBounds
import Mxx.Certificate.MatrixTypeSemantics
import Mxx.Certificate.LocalSoundness
import Mxx.Certificate.OperationalScalarSoundness
import Mxx.Ir.ExecutionFacts

/-! # Execution-aligned semantics for operational facts

This module contains proof-only objects used to connect one checked operational derivation to one
actual executable support path. It does not reconstruct the graph and is not imported by the
operational evaluator.
-/

namespace Mxx.Certificate

open Mxx.Ir

inductive ExecutionFrame where
  | call (node : Nat)
  | parallel (node indexSlot index : Nat)
  | sequential (node indexSlot index : Nat)
  deriving DecidableEq

structure ExecutionAddress where
  root : ProgramInstanceKey
  frames : List ExecutionFrame
  deriving DecidableEq

inductive SemanticMatrixKey where
  | executionValue (address : ExecutionAddress) (wire : WireRef)
      (representative : Mxx.Matrix)
  | deterministicHash (query : Mxx.HashQuery)
  | protocolInput (input : ProtocolInputId) (representative : Mxx.Matrix)
  | protocolFamilyElement
      (input : ProtocolInputId) (index : Nat) (representative : Mxx.Matrix)
  | gadget
      (paramsId : Mxx.SamplerParamsId)
      (params : Mxx.SamplerParams)
      (inputRows : Nat)
      (base : Int)
      (small : Bool)
      (digitCount : Nat)
  deriving DecidableEq

def SemanticMatrixKey.Holds
    (samplers : MxxSamplerFamily)
    (key : SemanticMatrixKey)
    (matrix : Mxx.Matrix) : Prop :=
  match key with
  | .executionValue _ _ representative => Mxx.MatrixModEq matrix representative
  | .deterministicHash query =>
      Mxx.MatrixModEq matrix ((samplers.hashSample query).withSamplerParams query.params)
  | .protocolInput _ representative | .protocolFamilyElement _ _ representative =>
      Mxx.MatrixModEq matrix representative
  | .gadget paramsId params inputRows base small digitCount =>
      ∃ publicMatrix,
        samplers.gadgetPublicMatrix paramsId params inputRows base small digitCount =
          some publicMatrix ∧
        Mxx.MatrixModEq matrix (publicMatrix.withSamplerParams params)

theorem SemanticMatrixKey.modEq_of_holds
    {samplers : MxxSamplerFamily}
    {key : SemanticMatrixKey}
    {left right : Mxx.Matrix}
    (leftHolds : key.Holds samplers left)
    (rightHolds : key.Holds samplers right) :
    Mxx.MatrixModEq left right := by
  cases key with
  | executionValue address wire representative =>
      exact Mxx.MatrixModEq.trans leftHolds (Mxx.MatrixModEq.symm rightHolds)
  | deterministicHash query =>
      exact Mxx.MatrixModEq.trans leftHolds (Mxx.MatrixModEq.symm rightHolds)
  | protocolInput input representative | protocolFamilyElement input index representative =>
      exact Mxx.MatrixModEq.trans leftHolds (Mxx.MatrixModEq.symm rightHolds)
  | gadget paramsId params inputRows base small digitCount =>
      rcases leftHolds with ⟨leftPublic, leftPublicEq, leftRelation⟩
      rcases rightHolds with ⟨rightPublic, rightPublicEq, rightRelation⟩
      have publicEq : leftPublic = rightPublic := by
        exact Option.some.inj (leftPublicEq.symm.trans rightPublicEq)
      subst rightPublic
      exact Mxx.MatrixModEq.trans leftRelation (Mxx.MatrixModEq.symm rightRelation)

theorem SemanticMatrixKey.holds_of_modEq
    {samplers : MxxSamplerFamily}
    {key : SemanticMatrixKey}
    {left right : Mxx.Matrix}
    (relation : Mxx.MatrixModEq left right)
    (rightHolds : key.Holds samplers right) :
    key.Holds samplers left := by
  cases key with
  | executionValue address wire representative
  | deterministicHash query
  | protocolInput input representative
  | protocolFamilyElement input index representative =>
      exact Mxx.MatrixModEq.trans relation rightHolds
  | gadget paramsId params inputRows base small digitCount =>
      rcases rightHolds with ⟨publicMatrix, publicResult, rightRelation⟩
      exact ⟨publicMatrix, publicResult, Mxx.MatrixModEq.trans relation rightRelation⟩

def ExecutionAddress.template (address : ExecutionAddress) : ScopeTemplateKey :=
  address.frames.foldl (init := .root address.root) fun scope frame =>
    match frame with
    | .call node => .callBody scope node
    | .parallel node _ _ => .parallelBody scope node
    | .sequential node _ _ => .sequentialBody scope node

def ExecutionAddress.instantiateHashContext
    (address : ExecutionAddress)
    (environment : ParamEnvironment)
    (domains : List OperationalParameterDomain) :
    Except OperationalError (ParamEnvironment × List OperationalParameterDomain) := do
  let (environment, domains) := address.frames.foldl (init := (environment, domains))
    fun (environment, domains) frame =>
      match frame with
      | .call _ => (environment, domains)
      | .parallel _ slot index | .sequential _ slot index =>
          (replaceLoopIndex environment slot index,
            instantiateParameterDomains slot index domains)
  return (← materializeInvariantParameters environment domains, domains)

def instantiateHashQuery
    (raw : DeterministicHashIdentity)
    (address : ExecutionAddress)
    (keyOrigin : OperationalValueOrigin)
    (key : ByteArray)
    (trailingOrigins : List OperationalValueOrigin)
    (trailingValues : List Int) : Option Mxx.HashQuery := do
  if raw.keyOrigin = keyOrigin then pure () else none
  if raw.trailingIntegerOrigins = trailingOrigins then pure () else none
  let (environment, domains) ←
    (address.instantiateHashContext raw.parameterEnvironment raw.parameterDomains).toOption
  let matrixParams ← raw.matrixType.evaluate environment (.constant 0)
  let tagValues ← (raw.tagExpressions.mapM
    (fun expression => evaluateIntInvariant environment domains expression)).toOption
  let tagDecimalValues ← (raw.tagDecimalExpressions.mapM
    (fun expression => evaluateIntInvariant environment domains expression)).toOption
  let tagU64LeValues ← (raw.tagU64LeExpressions.mapM
    (fun expression => evaluateIntInvariant environment domains expression)).toOption
  pure {
    params := matrixParams
    key
    variant := .plain
    tagPrefix := raw.tagPrefix
    tagValues
    tagDecimalValues
    tagU64LeValues
    trailingIntegerTagValues := trailingValues
    base := none
    digitCount := none
  }

structure DeterministicHashEvidence where
  address : ExecutionAddress
  keyOrigin : OperationalValueOrigin
  key : ByteArray
  trailingOrigins : List OperationalValueOrigin
  trailingValues : List Int

structure RelationTargetEvidence where
  key : SemanticMatrixKey
  matrix : Mxx.Matrix
  hashOrigin : Option DeterministicHashEvidence := none

structure DecompositionEvidence where
  producerKey : SemanticMatrixKey
  publicKey : SemanticMatrixKey
  input : RelationTargetEvidence
  publicMatrix : Mxx.Matrix
  decomposition : Mxx.Matrix

structure PreimageEvidence where
  producerKey : SemanticMatrixKey
  publicKey : SemanticMatrixKey
  target : RelationTargetEvidence
  publicMatrix : Mxx.Matrix
  preimage : Mxx.Matrix

inductive MatrixRelationEvidence where
  | decomposition (evidence : DecompositionEvidence)
  | preimage (evidence : PreimageEvidence)

structure MatrixFactEvidence where
  originKey : SemanticMatrixKey
  hashOrigin : Option DeterministicHashEvidence := none
  publicKey : Option SemanticMatrixKey := none
  relations : List MatrixRelationEvidence := []

structure TrapdoorFactEvidence where
  publicKey : SemanticMatrixKey

inductive RuntimeFactEvidence where
  | matrix (evidence : MatrixFactEvidence)
  | trapdoor (evidence : TrapdoorFactEvidence)
  | integer (value : Int)
  | bytes (value : ByteArray)
  | boolean (value : Bool)
  | real (value : Rat)
  | family (elements : List (ExecutionAddress × Mxx.Ir.Value × RuntimeFactEvidence))
  | opaque (description : String)

def CanonicalRange.Holds (range : CanonicalRange) (matrix : Mxx.Matrix) : Prop :=
  match range with
  | .unknown => True
  | .below upper => Mxx.maxCanonicalCoefficient matrix < upper

def OperationalBoundExpr.Holds
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (maximum : OperationalBoundExpr)
    (matrix : Mxx.Matrix) : Prop :=
  ∃ bound,
    maximum.evaluateWithStates parameters states = .ok bound ∧
    0 ≤ bound ∧ Mxx.maxCenteredCoefficientNorm matrix ≤ bound.toNat

theorem MatrixTypeExpr.Holds.transport
    {matrixType : MatrixTypeExpr}
    {parameters : ParamEnvironment}
    {left right : Mxx.Matrix}
    (relation : Mxx.MatrixModEq left right)
    (leftWellFormed : left.WellFormed)
    (rightHolds : matrixType.Holds parameters right) :
    matrixType.Holds parameters left := by
  rcases rightHolds with
    ⟨evaluated, evaluatedEq, modulus, ringDimension, rows, columns, rightWellFormed⟩
  exact ⟨evaluated, evaluatedEq, relation.modulus.trans modulus,
    relation.ringDimension.trans ringDimension, relation.rows.trans rows,
    relation.columns.trans columns, leftWellFormed⟩

theorem MatrixTypeExpr.Holds.wellFormed
    {matrixType : MatrixTypeExpr}
    {parameters : ParamEnvironment}
    {matrix : Mxx.Matrix}
    (holds : matrixType.Holds parameters matrix) : matrix.WellFormed := by
  rcases holds with ⟨_, _, _, _, _, _, wellFormed⟩
  exact wellFormed

theorem OperationalBoundExpr.Holds.transport
    {parameters : ParamEnvironment}
    {states : List OperationalNumericState}
    {maximum : OperationalBoundExpr}
    {left right : Mxx.Matrix}
    (relation : Mxx.MatrixModEq left right)
    (leftWellFormed : left.WellFormed)
    (rightWellFormed : right.WellFormed)
    (rightHolds : maximum.Holds parameters states right) :
    maximum.Holds parameters states left := by
  rcases rightHolds with ⟨maximum, evaluated, nonnegative, bound⟩
  exact ⟨maximum, evaluated, nonnegative, by
    rw [Mxx.maxCenteredCoefficientNorm_eq_of_matrixModEq relation leftWellFormed rightWellFormed]
    exact bound⟩

theorem OperationalBoundExpr.Holds.minimumCentered
    {parameters : ParamEnvironment}
    {states : List OperationalNumericState}
    {maximum : OperationalBoundExpr}
    {matrix : Mxx.Matrix}
    {bound modulus : Int}
    (maximumHolds : maximum.Holds parameters states matrix)
    (maximumEvaluates : maximum.evaluateWithStates parameters states = .ok bound)
    (modulusEq : matrix.modulus = modulus)
    (modulusPositive : 0 < modulus) :
    (OperationalBoundExpr.minimum (.closedInt (.constant (modulus / 2))) maximum).Holds
      parameters states matrix := by
  rcases maximumHolds with ⟨evaluated, evaluatedEq, evaluatedNonnegative, normBound⟩
  have boundEq : evaluated = bound := by
    exact Except.ok.inj (evaluatedEq.symm.trans maximumEvaluates)
  subst evaluated
  let capped := min (modulus / 2) bound
  refine ⟨capped, ?_, le_min (Int.ediv_nonneg modulusPositive.le (by omega))
    evaluatedNonnegative, ?_⟩
  · simp only [OperationalBoundExpr.evaluateWithStates,
      OperationalBoundExpr.evaluateWithStates_closedConstant]
    rw [maximumEvaluates]
    rfl
  · apply (Int.le_toNat (le_min
        (Int.ediv_nonneg modulusPositive.le (by omega)) evaluatedNonnegative)).2
    have radius := matrix_norm_le_centered_radius matrix (modulusEq.trans_gt modulusPositive)
    rw [modulusEq] at radius
    have radiusInt : (Mxx.maxCenteredCoefficientNorm matrix : Int) ≤ modulus / 2 := by
      omega
    have boundInt : (Mxx.maxCenteredCoefficientNorm matrix : Int) ≤ bound := by
      omega
    exact le_min radiusInt boundInt

theorem selectedTwoBoundHolds
    {parameters : ParamEnvironment}
    {states : List OperationalNumericState}
    {leftMaximum rightMaximum : OperationalBoundExpr}
    {left right output : Mxx.Matrix}
    {index : Int}
    (leftHolds : leftMaximum.Holds parameters states left)
    (rightHolds : rightMaximum.Holds parameters states right)
    (selected : [left, right][index.toNat]? = some output) :
    (OperationalBoundExpr.maximum
      (.maximum (.closedInt (.constant 0)) leftMaximum) rightMaximum).Holds
        parameters states output := by
  rcases leftHolds with ⟨leftBound, leftEvaluates, leftNonnegative, leftNorm⟩
  rcases rightHolds with ⟨rightBound, rightEvaluates, rightNonnegative, rightNorm⟩
  have evaluated :
      (OperationalBoundExpr.maximum
        (OperationalBoundExpr.maximum
          (OperationalBoundExpr.closedInt (.constant 0)) leftMaximum)
        rightMaximum).evaluateWithStates parameters states =
          Except.ok (max (max 0 leftBound) rightBound) := by
    simp only [OperationalBoundExpr.evaluateWithStates,
      OperationalBoundExpr.evaluateWithStates_closedConstant]
    rw [leftEvaluates, rightEvaluates]
    rfl
  refine ⟨max (max 0 leftBound) rightBound, evaluated, by omega, ?_⟩
  cases indexNat : index.toNat with
  | zero =>
      simp [indexNat] at selected
      subst output
      exact le_trans leftNorm (by omega)
  | succ first =>
      cases first with
      | zero =>
          simp [indexNat] at selected
          subst output
          exact le_trans rightNorm (by omega)
      | succ rest => simp [indexNat] at selected

theorem CanonicalRange.Holds.transport
    {range : CanonicalRange}
    {left right : Mxx.Matrix}
    (relation : Mxx.MatrixModEq left right)
    (leftWellFormed : left.WellFormed)
    (rightWellFormed : right.WellFormed)
    (rightHolds : range.Holds right) :
    range.Holds left := by
  cases range with
  | unknown => trivial
  | below upper =>
      change Mxx.maxCanonicalCoefficient left < upper
      change Mxx.maxCanonicalCoefficient right < upper at rightHolds
      rw [Mxx.maxCanonicalCoefficient_eq_of_matrixModEq relation leftWellFormed rightWellFormed]
      exact rightHolds

def MatrixOriginIdentity.HoldsEvidence
    (samplers : MxxSamplerFamily)
    (origin : MatrixOriginIdentity)
    (key : SemanticMatrixKey)
    (hashOrigin : Option DeterministicHashEvidence)
    (matrix : Mxx.Matrix) : Prop :=
  match origin, key with
  | .value scope wire, .executionValue creator candidate representative =>
      scope = creator.template ∧ wire = candidate ∧ Mxx.MatrixModEq matrix representative
  | .protocolInput input, .protocolInput candidate representative =>
      input = candidate ∧ Mxx.MatrixModEq matrix representative
  | .protocolFamilyElement input index,
      .protocolFamilyElement candidate candidateIndex representative =>
      input = candidate ∧ index = candidateIndex ∧ Mxx.MatrixModEq matrix representative
  | .deterministicHash raw, .deterministicHash query =>
      ∃ evidence,
        hashOrigin = some evidence ∧
        instantiateHashQuery raw evidence.address evidence.keyOrigin evidence.key
          evidence.trailingOrigins evidence.trailingValues = some query ∧
        key.Holds samplers matrix
  | _, _ => False

theorem MatrixOriginIdentity.HoldsEvidence.transport
    {samplers : MxxSamplerFamily}
    {origin : MatrixOriginIdentity}
    {key : SemanticMatrixKey}
    {hashOrigin : Option DeterministicHashEvidence}
    {left right : Mxx.Matrix}
    (relation : Mxx.MatrixModEq left right)
    (rightHolds : origin.HoldsEvidence samplers key hashOrigin right) :
    origin.HoldsEvidence samplers key hashOrigin left := by
  cases origin with
  | value scope wire =>
      cases key with
      | executionValue creator candidate representative =>
          rcases rightHolds with ⟨scopeEq, wireEq, rightRelation⟩
          exact ⟨scopeEq, wireEq, Mxx.MatrixModEq.trans relation rightRelation⟩
      | _ => exact False.elim rightHolds

  | protocolInput input =>
      cases key with
      | protocolInput candidate representative =>
          rcases rightHolds with ⟨inputEq, rightRelation⟩
          exact ⟨inputEq, Mxx.MatrixModEq.trans relation rightRelation⟩
      | _ => exact False.elim rightHolds
  | protocolFamilyElement input index =>
      cases key with
      | protocolFamilyElement candidate candidateIndex representative =>
          rcases rightHolds with ⟨inputEq, indexEq, rightRelation⟩
          exact ⟨inputEq, indexEq, Mxx.MatrixModEq.trans relation rightRelation⟩
      | _ => exact False.elim rightHolds
  | deterministicHash identity =>
      cases key with
      | deterministicHash query =>
          rcases rightHolds with ⟨evidence, evidenceEq, queryEq, keyHolds⟩
          exact ⟨evidence, evidenceEq, queryEq,
            SemanticMatrixKey.holds_of_modEq relation keyHolds⟩
      | _ => exact False.elim rightHolds

theorem deterministicHashOrigin_holds
    (samplers : MxxSamplerFamily)
    (raw : DeterministicHashIdentity)
    (evidence : DeterministicHashEvidence)
    (query : Mxx.HashQuery)
    (queryExact : instantiateHashQuery raw evidence.address evidence.keyOrigin evidence.key
      evidence.trailingOrigins evidence.trailingValues = some query) :
    (MatrixOriginIdentity.deterministicHash raw).HoldsEvidence samplers
      (.deterministicHash query) (some evidence)
      ((samplers.hashSample query).withSamplerParams query.params) := by
  exact ⟨evidence, rfl, queryExact, Mxx.MatrixModEq.refl _⟩

def PublicMatrixIdentity.HoldsEvidence
    (samplers : MxxSamplerFamily)
    (identity : PublicMatrixIdentity)
    (key : SemanticMatrixKey)
    (matrix : Mxx.Matrix) : Prop :=
  match identity, key with
  | .sampledTrapdoor scope wire, .executionValue creator candidate representative =>
      scope = creator.template ∧ wire = candidate ∧ Mxx.MatrixModEq matrix representative
  | .gadget paramsId params inputRows base small digitCount,
      .gadget candidateId candidateParams candidateRows candidateBase candidateSmall
        candidateCount =>
      paramsId = candidateId ∧ params = candidateParams ∧ inputRows = candidateRows ∧
        base = candidateBase ∧ small = candidateSmall ∧ digitCount = candidateCount ∧
        key.Holds samplers matrix
  | .loopInstance _ _ source, _ => source.HoldsEvidence samplers key matrix
  | .selected _ _ _, _ => False
  | _, _ => False

theorem PublicMatrixIdentity.HoldsEvidence.transport
    {samplers : MxxSamplerFamily}
    {identity : PublicMatrixIdentity}
    {key : SemanticMatrixKey}
    {left right : Mxx.Matrix}
    (relation : Mxx.MatrixModEq left right)
    (rightHolds : identity.HoldsEvidence samplers key right) :
    identity.HoldsEvidence samplers key left := by
  induction identity generalizing key with
  | sampledTrapdoor scope wire =>
      cases key with
      | executionValue creator candidate representative =>
          rcases rightHolds with ⟨scopeEq, wireEq, rightRelation⟩
          exact ⟨scopeEq, wireEq, Mxx.MatrixModEq.trans relation rightRelation⟩
      | _ => exact False.elim rightHolds
  | gadget paramsId params inputRows base small digitCount =>
      cases key with
      | gadget candidateId candidateParams candidateRows candidateBase candidateSmall
          candidateCount =>
          rcases rightHolds with
            ⟨idEq, paramsEq, rowsEq, baseEq, smallEq, countEq, keyHolds⟩
          exact ⟨idEq, paramsEq, rowsEq, baseEq, smallEq, countEq,
            SemanticMatrixKey.holds_of_modEq relation keyHolds⟩
      | _ => exact False.elim rightHolds
  | selected binder selection source =>
      exact False.elim rightHolds
  | loopInstance slot index source inductionHypothesis =>
      exact inductionHypothesis rightHolds

def RelationTargetSummary.HoldsEvidence
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (summary : RelationTargetSummary)
    (evidence : RelationTargetEvidence) : Prop :=
  summary.origin.HoldsEvidence samplers evidence.key evidence.hashOrigin evidence.matrix ∧
  evidence.key.Holds samplers evidence.matrix ∧
  summary.matrixType.Holds parameters evidence.matrix ∧
  evidence.matrix.modulus = summary.matrixParams.modulus ∧
  evidence.matrix.ringDimension = summary.matrixParams.ringDimension ∧
  evidence.matrix.rows = summary.matrixParams.rows ∧
  evidence.matrix.columns = summary.matrixParams.columns ∧
  summary.totalHardBound.Holds parameters states evidence.matrix ∧
  summary.canonicalRange.Holds evidence.matrix

theorem RelationTargetSummary.HoldsEvidence.transport
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {states : List OperationalNumericState}
    {summary : RelationTargetSummary}
    {evidence : RelationTargetEvidence}
    {output : Mxx.Matrix}
    (relation : Mxx.MatrixModEq output evidence.matrix)
    (outputWellFormed : output.WellFormed)
    (targetHolds : summary.HoldsEvidence samplers parameters states evidence) :
    summary.HoldsEvidence samplers parameters states { evidence with matrix := output } := by
  rcases targetHolds with
    ⟨originHolds, keyHolds, typeHolds, modulus, ringDimension, rows, columns,
      magnitudeHolds, canonicalHolds⟩
  have targetWellFormed := typeHolds.wellFormed
  exact ⟨originHolds.transport relation,
    SemanticMatrixKey.holds_of_modEq relation keyHolds,
    typeHolds.transport relation outputWellFormed,
    relation.modulus.trans modulus, relation.ringDimension.trans ringDimension,
    relation.rows.trans rows, relation.columns.trans columns,
    magnitudeHolds.transport relation outputWellFormed targetWellFormed,
    canonicalHolds.transport relation outputWellFormed targetWellFormed⟩

def MatrixRelationEvidence.Holds
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (relation : OperationalMatrixRelation)
    (evidence : MatrixRelationEvidence) : Prop :=
  match relation, evidence with
  | .decomposition relation, .decomposition witness =>
      relation.producer.HoldsEvidence samplers witness.producerKey none witness.decomposition ∧
      relation.publicIdentity.HoldsEvidence samplers witness.publicKey
        witness.publicMatrix ∧
      relation.inputSummary.HoldsEvidence samplers parameters states witness.input ∧
      witness.producerKey.Holds samplers witness.decomposition ∧
      witness.publicKey.Holds samplers witness.publicMatrix ∧
      witness.input.key.Holds samplers witness.input.matrix ∧
      (relation.status = .available →
        Mxx.MatrixModEq (Mxx.matrixMul witness.publicMatrix witness.decomposition)
          witness.input.matrix)
  | .preimage relation, .preimage witness =>
      relation.producer.HoldsEvidence samplers witness.producerKey none witness.preimage ∧
      relation.publicIdentity.HoldsEvidence samplers witness.publicKey
        witness.publicMatrix ∧
      relation.targetSummary.HoldsEvidence samplers parameters states witness.target ∧
      witness.producerKey.Holds samplers witness.preimage ∧
      witness.publicKey.Holds samplers witness.publicMatrix ∧
      witness.target.key.Holds samplers witness.target.matrix ∧
      Mxx.MatrixModEq (Mxx.matrixMul witness.publicMatrix witness.preimage)
        witness.target.matrix
  | _, _ => False

theorem MatrixRelationEvidence.decomposition_holds
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {states : List OperationalNumericState}
    {relation : DecompositionRelation}
    {witness : DecompositionEvidence}
    (producerHolds : relation.producer.HoldsEvidence samplers witness.producerKey none
      witness.decomposition)
    (publicIdentityHolds : relation.publicIdentity.HoldsEvidence samplers witness.publicKey
      witness.publicMatrix)
    (inputSummaryHolds : relation.inputSummary.HoldsEvidence samplers parameters states
      witness.input)
    (producerKeyHolds : witness.producerKey.Holds samplers witness.decomposition)
    (publicKeyHolds : witness.publicKey.Holds samplers witness.publicMatrix)
    (inputKeyHolds : witness.input.key.Holds samplers witness.input.matrix)
    (reconstruction : relation.status = .available →
      Mxx.MatrixModEq (Mxx.matrixMul witness.publicMatrix witness.decomposition)
        witness.input.matrix) :
    MatrixRelationEvidence.Holds samplers parameters states (.decomposition relation)
      (.decomposition witness) := by
  exact ⟨producerHolds, publicIdentityHolds, inputSummaryHolds, producerKeyHolds,
    publicKeyHolds, inputKeyHolds, reconstruction⟩

theorem MatrixRelationEvidence.preimage_holds
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {states : List OperationalNumericState}
    {relation : PreimageRelation}
    {witness : PreimageEvidence}
    (producerHolds : relation.producer.HoldsEvidence samplers witness.producerKey none
      witness.preimage)
    (publicIdentityHolds : relation.publicIdentity.HoldsEvidence samplers witness.publicKey
      witness.publicMatrix)
    (targetSummaryHolds : relation.targetSummary.HoldsEvidence samplers parameters states
      witness.target)
    (producerKeyHolds : witness.producerKey.Holds samplers witness.preimage)
    (publicKeyHolds : witness.publicKey.Holds samplers witness.publicMatrix)
    (targetKeyHolds : witness.target.key.Holds samplers witness.target.matrix)
    (reconstruction :
      Mxx.MatrixModEq (Mxx.matrixMul witness.publicMatrix witness.preimage)
        witness.target.matrix) :
    MatrixRelationEvidence.Holds samplers parameters states (.preimage relation)
      (.preimage witness) := by
  exact ⟨producerHolds, publicIdentityHolds, targetSummaryHolds, producerKeyHolds,
    publicKeyHolds, targetKeyHolds, reconstruction⟩

theorem relationProduct_modEq
    (q ringDimension rows inner columns : Nat)
    [Fact (1 < q)] [NeZero q] [NeZero ringDimension]
    (basis subject target : Mxx.Matrix)
    (basisLayout : Mxx.Toolkit.MatrixLayout basis q ringDimension rows inner)
    (subjectLayout : Mxx.Toolkit.MatrixLayout subject q ringDimension inner columns)
    (targetLayout : Mxx.Toolkit.MatrixLayout target q ringDimension rows columns)
    (relation : Mxx.MatrixModEq (Mxx.matrixMul basis subject) target) :
    Mxx.MatrixModEq (Mxx.matrixMultiply basis subject) target := by
  have outputLayout := Mxx.Toolkit.matrixMultiply_layout basis subject basisLayout subjectLayout
  apply Mxx.Toolkit.modEq_of_matrixValue_eq q ringDimension rows columns
    (Mxx.matrixMultiply basis subject) target outputLayout targetLayout
  rw [Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension rows inner columns
    basis subject basisLayout subjectLayout]
  rw [← Mxx.Toolkit.matrixValue_mul q ringDimension rows inner columns basis subject
    ⟨basisLayout.modulus, basisLayout.ringDimension, basisLayout.rows, basisLayout.columns⟩
    ⟨subjectLayout.modulus, subjectLayout.ringDimension, subjectLayout.rows,
      subjectLayout.columns⟩]
  exact Mxx.Toolkit.matrixValue_eq_of_modEq q ringDimension rows columns
    (Mxx.matrixMul basis subject) target
    (Mxx.Toolkit.matrixMul_layout basis subject basisLayout subjectLayout) targetLayout relation

theorem relationProduct_target_holds
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (q ringDimension rows inner columns : Nat)
    [Fact (1 < q)] [NeZero q] [NeZero ringDimension]
    (leftKey rightKey : SemanticMatrixKey)
    (left right publicMatrix subject : Mxx.Matrix)
    (targetSummary : RelationTargetSummary)
    (targetEvidence : RelationTargetEvidence)
    (leftLayout : Mxx.Toolkit.MatrixLayout left q ringDimension rows inner)
    (publicLayout : Mxx.Toolkit.MatrixLayout publicMatrix q ringDimension rows inner)
    (rightLayout : Mxx.Toolkit.MatrixLayout right q ringDimension inner columns)
    (subjectLayout : Mxx.Toolkit.MatrixLayout subject q ringDimension inner columns)
    (targetLayout : Mxx.Toolkit.MatrixLayout targetEvidence.matrix q ringDimension rows columns)
    (leftHolds : leftKey.Holds samplers left)
    (publicHolds : leftKey.Holds samplers publicMatrix)
    (rightHolds : rightKey.Holds samplers right)
    (subjectHolds : rightKey.Holds samplers subject)
    (targetHolds : targetSummary.HoldsEvidence samplers parameters states targetEvidence)
    (storedRelation : Mxx.MatrixModEq (Mxx.matrixMul publicMatrix subject)
      targetEvidence.matrix) :
    targetSummary.HoldsEvidence samplers parameters states {
      targetEvidence with matrix := Mxx.matrixMultiply left right
    } := by
  have operandsCongruent := Mxx.Toolkit.MatrixModEq.multiply q ringDimension rows inner columns
    left publicMatrix right subject leftLayout publicLayout rightLayout subjectLayout
    (SemanticMatrixKey.modEq_of_holds leftHolds publicHolds)
    (SemanticMatrixKey.modEq_of_holds rightHolds subjectHolds)
  have storedProduct := relationProduct_modEq q ringDimension rows inner columns publicMatrix
    subject targetEvidence.matrix publicLayout subjectLayout targetLayout storedRelation
  have outputRelation := Mxx.MatrixModEq.trans operandsCongruent storedProduct
  have outputLayout := Mxx.Toolkit.matrixMultiply_layout left right leftLayout rightLayout
  have outputWellFormed : (Mxx.matrixMultiply left right).WellFormed := by
    simpa [Mxx.Matrix.WellFormed, outputLayout.rows, outputLayout.columns,
      outputLayout.ringDimension] using outputLayout.coefficientCount
  exact targetHolds.transport outputRelation outputWellFormed

def OperationalFact.HoldsEvidence
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress) :
    OperationalFact → Mxx.Ir.Value → RuntimeFactEvidence → Prop
  | OperationalFact.matrix fact, Mxx.Ir.Value.matrix actualMatrix,
      RuntimeFactEvidence.matrix evidence =>
      fact.matrixType.Holds parameters actualMatrix ∧
      actualMatrix.modulus = fact.matrixParams.modulus ∧
      actualMatrix.ringDimension = fact.matrixParams.ringDimension ∧
      actualMatrix.rows = fact.matrixParams.rows ∧
      actualMatrix.columns = fact.matrixParams.columns ∧
      fact.origin.HoldsEvidence samplers evidence.originKey evidence.hashOrigin actualMatrix ∧
      fact.totalHardBound.Holds parameters states actualMatrix ∧
      fact.canonicalRange.Holds actualMatrix ∧
      (match fact.identity, evidence.publicKey with
        | none, none => True
        | some identity, some key => identity.HoldsEvidence samplers key actualMatrix
        | _, _ => False) ∧
      List.Forall₂
        (MatrixRelationEvidence.Holds samplers parameters states)
        fact.relations evidence.relations
  | OperationalFact.trapdoor fact, Mxx.Ir.Value.trapdoor publicMatrix _,
      RuntimeFactEvidence.trapdoor evidence =>
      fact.matrixType.Holds parameters publicMatrix ∧
      fact.publicIdentity.HoldsEvidence samplers evidence.publicKey publicMatrix ∧
      ∃ maximum,
        fact.maximum.evaluateWithStates parameters states = .ok maximum ∧
        0 ≤ maximum ∧ Mxx.maxCenteredCoefficientNorm publicMatrix ≤ maximum.toNat
  | OperationalFact.integer fact, Mxx.Ir.Value.integer value,
      RuntimeFactEvidence.integer evidenceValue =>
      value = evidenceValue ∧ fact.lower ≤ value ∧ value ≤ fact.upper
  | OperationalFact.bytes fact, Mxx.Ir.Value.bytes value,
      RuntimeFactEvidence.bytes evidenceValue =>
      value = evidenceValue ∧ 0 ≤ fact.length ∧ value.size = fact.length.toNat
  | OperationalFact.boolean, Mxx.Ir.Value.boolean value,
      RuntimeFactEvidence.boolean evidenceValue => value = evidenceValue
  | OperationalFact.real, Mxx.Ir.Value.rational value,
      RuntimeFactEvidence.real evidenceValue => value = evidenceValue
  | OperationalFact.familyUniform _ _ element count, Mxx.Ir.Value.family values,
      RuntimeFactEvidence.family evidence =>
      0 ≤ count ∧ values.length = count.toNat ∧ evidence.length = values.length ∧
      ∀ (index : Nat) (value : Mxx.Ir.Value)
          (item : ExecutionAddress × Mxx.Ir.Value × RuntimeFactEvidence),
        values[index]? = some value → evidence[index]? = some item →
        item.2.1 = value ∧
        OperationalFact.HoldsEvidence samplers parameters states item.1 element value item.2.2
  | OperationalFact.familyPackedNil, Mxx.Ir.Value.family [], RuntimeFactEvidence.family [] => True
  | OperationalFact.familyPackedCons head tail, Mxx.Ir.Value.family (value :: values),
      RuntimeFactEvidence.family (headEvidence :: tailEvidence) =>
      OperationalFact.HoldsEvidence samplers parameters states headEvidence.1 head value
          headEvidence.2.2 ∧
        OperationalFact.HoldsEvidence samplers parameters states address tail (.family values)
          (RuntimeFactEvidence.family tailEvidence)
  | OperationalFact.typedBlob typeName, Mxx.Ir.Value.opaque description,
      RuntimeFactEvidence.opaque evidenceDescription =>
      description = evidenceDescription ∧ typeName = description
  | _, _, _ => False

theorem OperationalFact.HoldsEvidence.withMatrixRelations
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {states : List OperationalNumericState}
    {address : ExecutionAddress}
    {fact : OperationalMatrixFact}
    {matrix : Mxx.Matrix}
    {evidence : MatrixFactEvidence}
    {relations : List OperationalMatrixRelation}
    {relationEvidence : List MatrixRelationEvidence}
    (base : OperationalFact.HoldsEvidence samplers parameters states address
      (.matrix { fact with relations := [] }) (.matrix matrix)
      (.matrix { evidence with relations := [] }))
    (relationsHold : List.Forall₂
      (MatrixRelationEvidence.Holds samplers parameters states)
      relations relationEvidence) :
    OperationalFact.HoldsEvidence samplers parameters states address
      (.matrix { fact with relations }) (.matrix matrix)
      (.matrix { evidence with relations := relationEvidence }) := by
  rcases base with ⟨typeHolds, modulus, ringDimension, rows, columns, originHolds,
    magnitudeHolds, canonicalHolds, identityHolds, _⟩
  exact ⟨typeHolds, modulus, ringDimension, rows, columns, originHolds,
    magnitudeHolds, canonicalHolds, identityHolds, relationsHold⟩

theorem OperationalFact.HoldsEvidence.capFiniteMatrix
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {states : List OperationalNumericState}
    {address : ExecutionAddress}
    {fact : OperationalMatrixFact}
    {matrix : Mxx.Matrix}
    {evidence : MatrixFactEvidence}
    {maximum : OperationalBoundExpr}
    {bound : Int}
    (factMagnitude : fact.totalHardBound = .finite maximum)
    (holds : OperationalFact.HoldsEvidence samplers parameters states address
      (.matrix fact) (.matrix matrix) (.matrix evidence))
    (maximumEvaluates : maximum.evaluateWithStates parameters states = .ok bound)
    (modulusPositive : 0 < matrix.modulus) :
    OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
      fact with totalHardBound := (.minimum
        (.closedInt (.constant (matrix.modulus / 2))) maximum)
    }) (.matrix matrix) (.matrix evidence) := by
  rcases holds with ⟨typeHolds, modulus, ringDimension, rows, columns, originHolds,
    magnitudeHolds, canonicalHolds, publicHolds, relationsHolds⟩
  rw [factMagnitude] at magnitudeHolds
  exact ⟨typeHolds, modulus, ringDimension, rows, columns, originHolds,
    magnitudeHolds.minimumCentered maximumEvaluates rfl modulusPositive,
    canonicalHolds, publicHolds, relationsHolds⟩

theorem OperationalFact.HoldsEvidence.capMatrix
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {states : List OperationalNumericState}
    {address : ExecutionAddress}
    {fact : OperationalMatrixFact}
    {matrix : Mxx.Matrix}
    {evidence : MatrixFactEvidence}
    {bound : Int}
    (holds : OperationalFact.HoldsEvidence samplers parameters states address
      (.matrix fact) (.matrix matrix) (.matrix evidence))
    (maximumEvaluates :
      fact.totalHardBound.boundExpression.evaluateWithStates parameters states = .ok bound)
    (modulusPositive : 0 < matrix.modulus) :
    OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
      fact with magnitude := fact.totalHardBound.centeredCap matrix.modulus
    }) (.matrix matrix) (.matrix evidence) := by
  rcases holds with ⟨typeHolds, modulus, ringDimension, rows, columns, originHolds,
    magnitudeHolds, canonicalHolds, publicHolds, relationsHolds⟩
  exact ⟨typeHolds, modulus, ringDimension, rows, columns, originHolds,
    magnitudeHolds.minimumCenteredAny maximumEvaluates rfl modulusPositive,
    canonicalHolds, publicHolds, relationsHolds⟩

theorem relationProductFact_operational_sound
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (outputWire : WireRef)
    (matrixType : MatrixTypeExpr)
    (q ringDimension rows inner columns : Nat)
    [Fact (1 < q)] [NeZero q] [NeZero ringDimension]
    (leftKey rightKey : SemanticMatrixKey)
    (left right publicMatrix subject : Mxx.Matrix)
    (targetSummary : RelationTargetSummary)
    (targetEvidence : RelationTargetEvidence)
    (leftLayout : Mxx.Toolkit.MatrixLayout left q ringDimension rows inner)
    (publicLayout : Mxx.Toolkit.MatrixLayout publicMatrix q ringDimension rows inner)
    (rightLayout : Mxx.Toolkit.MatrixLayout right q ringDimension inner columns)
    (subjectLayout : Mxx.Toolkit.MatrixLayout subject q ringDimension inner columns)
    (targetLayout : Mxx.Toolkit.MatrixLayout targetEvidence.matrix q ringDimension rows columns)
    (outputTypeHolds : matrixType.Holds parameters (Mxx.matrixMultiply left right))
    (leftHolds : leftKey.Holds samplers left)
    (publicHolds : leftKey.Holds samplers publicMatrix)
    (rightHolds : rightKey.Holds samplers right)
    (subjectHolds : rightKey.Holds samplers subject)
    (targetHolds : targetSummary.HoldsEvidence samplers parameters states targetEvidence)
    (storedRelation : Mxx.MatrixModEq (Mxx.matrixMul publicMatrix subject)
      targetEvidence.matrix) :
    OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
      subject := outputWire
      origin := targetSummary.origin
      matrixType
      matrixParams := targetSummary.matrixParams
      totalHardBound := targetSummary.totalHardBound
      canonicalRange := targetSummary.canonicalRange
    }) (.matrix (Mxx.matrixMultiply left right)) (.matrix {
      originKey := targetEvidence.key
      hashOrigin := targetEvidence.hashOrigin
    }) := by
  have targetResult := relationProduct_target_holds samplers parameters states q ringDimension rows
    inner columns leftKey rightKey left right publicMatrix subject targetSummary targetEvidence
    leftLayout publicLayout rightLayout subjectLayout targetLayout leftHolds publicHolds rightHolds
    subjectHolds targetHolds storedRelation
  rcases targetResult with
    ⟨originHolds, keyHolds, _, modulus, ring, outputRows, outputColumns, magnitude,
      canonical⟩
  exact ⟨outputTypeHolds, modulus, ring, outputRows, outputColumns, originHolds, magnitude,
    canonical, trivial, List.Forall₂.nil⟩

structure AlignedWireEvidence where
  wire : WireRef
  fact : OperationalFact
  value : Mxx.Ir.Value
  evidence : RuntimeFactEvidence
  address : ExecutionAddress

def AlignedWireEvidence.Holds
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (entry : AlignedWireEvidence) : Prop :=
  entry.fact.HoldsEvidence samplers parameters states entry.address entry.value entry.evidence

def AlignedWireEvidence.Matches
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (facts : OperationalScopeFacts)
    (wireValue : WireRef × Value)
    (entry : AlignedWireEvidence) : Prop :=
  entry.wire = wireValue.1 ∧
  entry.value = wireValue.2 ∧
  facts[entry.wire.node]?.bind (fun outputs => outputs[entry.wire.port]?) = some entry.fact ∧
  entry.Holds samplers parameters states

/-- The sole straight-line proof invariant. It is positionally aligned with the executable wire
environment and performs no graph or identity search. -/
def OperationalScopeAlignment
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (facts : OperationalScopeFacts)
    (wires : WireEnvironment)
    (entries : List AlignedWireEvidence) : Prop :=
  List.Forall₂ (AlignedWireEvidence.Matches samplers parameters states facts) wires entries

theorem OperationalScopeAlignment.nil
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState) :
    OperationalScopeAlignment samplers parameters states #[] [] [] := by
  exact List.Forall₂.nil

theorem OperationalScopeAlignment.lookup
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {states : List OperationalNumericState}
    {facts : OperationalScopeFacts}
    {wires : WireEnvironment}
    {entries : List AlignedWireEvidence}
    (alignment : OperationalScopeAlignment samplers parameters states facts wires entries)
    (wire : WireRef)
    (value : Value)
    (lookup : lookupWire wire wires = some value) :
    ∃ entry ∈ entries,
      entry.wire = wire ∧ entry.value = value ∧
      facts[wire.node]?.bind (fun outputs => outputs[wire.port]?) = some entry.fact ∧
      entry.Holds samplers parameters states := by
  induction alignment with
  | nil => simp [lookupWire] at lookup
  | @cons wireValue entry wireTail entryTail headMatches _ tailInduction =>
      rcases wireValue with ⟨candidate, candidateValue⟩
      rcases headMatches with ⟨entryWire, entryValue, factLookup, entryHolds⟩
      simp only [lookupWire] at lookup
      by_cases same : candidate = wire
      · simp only [same, if_pos] at lookup
        have valueEq : candidateValue = value := Option.some.inj lookup
        subst candidate
        subst candidateValue
        refine ⟨entry, by simp, entryWire, entryValue, ?_, entryHolds⟩
        simpa [entryWire] using factLookup
      · simp only [if_neg same] at lookup
        obtain ⟨result, resultMember, resultWire, resultValue, resultFact, resultHolds⟩ :=
          tailInduction lookup
        exact ⟨result, by simp [resultMember], resultWire, resultValue, resultFact, resultHolds⟩

theorem OperationalScopeAlignment.pushFacts
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {states : List OperationalNumericState}
    {facts : OperationalScopeFacts}
    {wires : WireEnvironment}
    {entries : List AlignedWireEvidence}
    (alignment : OperationalScopeAlignment samplers parameters states facts wires entries)
    (outputs : Array OperationalFact) :
    OperationalScopeAlignment samplers parameters states (facts.push outputs) wires entries := by
  induction alignment with
  | nil => exact List.Forall₂.nil
  | @cons wireValue entry wireTail entryTail headMatches _ tailInduction =>
      apply List.Forall₂.cons
      · rcases headMatches with ⟨wireEq, valueEq, factLookup, entryHolds⟩
        refine ⟨wireEq, valueEq, ?_, entryHolds⟩
        have inBounds : entry.wire.node < facts.size := by
          cases lookupNode : facts[entry.wire.node]? with
          | none => simp [lookupNode] at factLookup
          | some nodeFacts => exact (Array.getElem?_eq_some_iff.mp lookupNode).1
        rw [Array.getElem?_push_lt inBounds]
        rw [show facts[entry.wire.node]? = some facts[entry.wire.node] by simp [inBounds]]
          at factLookup
        exact factLookup
      · exact tailInduction

private theorem forall₂_append
    {α β : Type}
    {relation : α → β → Prop}
    {leftA rightA : List α}
    {leftB rightB : List β}
    (left : List.Forall₂ relation leftA leftB)
    (right : List.Forall₂ relation rightA rightB) :
    List.Forall₂ relation (leftA ++ rightA) (leftB ++ rightB) := by
  induction left with
  | nil => exact right
  | cons head tail induction => exact .cons head induction

theorem OperationalScopeAlignment.appendOutputs
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {states : List OperationalNumericState}
    {facts : OperationalScopeFacts}
    {wires : WireEnvironment}
    {entries : List AlignedWireEvidence}
    (alignment : OperationalScopeAlignment samplers parameters states facts wires entries)
    (outputs : Array OperationalFact)
    (outputWires : WireEnvironment)
    (outputEntries : List AlignedWireEvidence)
    (outputsAligned : List.Forall₂
      (AlignedWireEvidence.Matches samplers parameters states (facts.push outputs))
      outputWires outputEntries) :
    OperationalScopeAlignment samplers parameters states (facts.push outputs)
      (wires ++ outputWires) (entries ++ outputEntries) := by
  exact forall₂_append (alignment.pushFacts outputs) outputsAligned

theorem singletonOutputAligned
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (facts : OperationalScopeFacts)
    (address : ExecutionAddress)
    (fact : OperationalFact)
    (value : Value)
    (evidence : RuntimeFactEvidence)
    (holds : fact.HoldsEvidence samplers parameters states address value evidence) :
    List.Forall₂
      (AlignedWireEvidence.Matches samplers parameters states (facts.push #[fact]))
      (bindOutputs facts.size [value])
      [{
        wire := { node := facts.size, port := 0 }
        fact
        value
        evidence
        address
      }] := by
  apply List.Forall₂.cons
  · exact ⟨by simp, by simp, by simp, holds⟩
  · exact List.Forall₂.nil

theorem constantIntegerFact_holdsEvidence
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (subject : WireRef)
    (origin : OperationalValueOrigin)
    (value : Int) :
    OperationalFact.HoldsEvidence samplers parameters states address (.integer {
      subject
      origin
      lower := value
      upper := value
    }) (.integer value) (.integer value) := by
  exact ⟨rfl, le_rfl, le_rfl⟩

theorem integerFact_holdsEvidence
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (subject : WireRef)
    (origin : OperationalValueOrigin)
    (value lower upper : Int)
    (lowerHolds : lower ≤ value)
    (upperHolds : value ≤ upper) :
    OperationalFact.HoldsEvidence samplers parameters states address (.integer {
      subject
      origin
      lower
      upper
    }) (.integer value) (.integer value) := by
  exact ⟨rfl, lowerHolds, upperHolds⟩

theorem booleanToIntegerFact_holdsEvidence
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (subject : WireRef)
    (origin : OperationalValueOrigin)
    (value : Bool) :
    OperationalFact.HoldsEvidence samplers parameters states address (.integer {
      subject
      origin
      lower := 0
      upper := 1
    }) (.integer (if value then 1 else 0)) (.integer (if value then 1 else 0)) := by
  cases value <;> simp [OperationalFact.HoldsEvidence]

theorem booleanFact_holdsEvidence
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (value : Bool) :
    OperationalFact.HoldsEvidence samplers parameters states address .boolean
      (.boolean value) (.boolean value) := by
  rfl

theorem localMatrixFact_holdsEvidence
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (wire : WireRef)
    (matrixType : MatrixTypeExpr)
    (matrixParams : Mxx.SamplerParams)
    (magnitude : OperationalBoundExpr)
    (canonicalRange : CanonicalRange)
    (matrix : Mxx.Matrix)
    (typeHolds : matrixType.Holds parameters matrix)
    (modulus : matrix.modulus = matrixParams.modulus)
    (ringDimension : matrix.ringDimension = matrixParams.ringDimension)
    (rows : matrix.rows = matrixParams.rows)
    (columns : matrix.columns = matrixParams.columns)
    (magnitudeHolds : magnitude.Holds parameters states matrix)
    (canonicalHolds : canonicalRange.Holds matrix) :
    OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
      subject := wire
      origin := .value address.template wire
      matrixType
      matrixParams
      magnitude
      canonicalRange
    }) (.matrix matrix) (.matrix {
      originKey := .executionValue address wire matrix
    }) := by
  exact ⟨typeHolds, modulus, ringDimension, rows, columns, ⟨rfl, rfl, Mxx.MatrixModEq.refl matrix⟩,
    magnitudeHolds, canonicalHolds, trivial, .nil⟩

private theorem zeroConstant_canonicalRange
    (matrixParams : Mxx.SamplerParams) :
    CanonicalRange.Holds (.below 1) (zeroConstantOutput matrixParams) := by
  have foldZeros : ∀ count : Nat,
      List.foldl (fun maximum coefficient =>
        max maximum (Mxx.canonicalCoefficient matrixParams.modulus coefficient)) 0
        (List.replicate count 0) = 0 := by
    intro count
    induction count with
    | zero => rfl
    | succ count induction =>
        rw [List.replicate_succ, List.foldl_cons]
        have zero : Mxx.canonicalCoefficient matrixParams.modulus 0 = 0 := by
          simp [Mxx.canonicalCoefficient, Mxx.reduceCoefficient]
        simpa [zero] using induction
  simpa [CanonicalRange.Holds, zeroConstantOutput, Mxx.maxCanonicalCoefficient,
    Mxx.Matrix.withSamplerParams] using
    foldZeros (matrixParams.rows * matrixParams.columns * matrixParams.ringDimension)

private theorem initial_le_foldAbsoluteMaximum
    (initial : Int) (values : List Int) :
    initial ≤ values.foldl (fun maximum value => max maximum (absolute value)) initial := by
  induction values generalizing initial with
  | nil => exact le_rfl
  | cons head tail induction =>
      exact le_trans (le_max_left _ _) (induction _)

private theorem member_le_foldAbsoluteMaximum
    (initial value : Int) (values : List Int)
    (member : value ∈ values) :
    absolute value ≤
      values.foldl (fun maximum value => max maximum (absolute value)) initial := by
  induction values generalizing initial with
  | nil => simp at member
  | cons head tail induction =>
      simp only [List.mem_cons] at member
      rw [List.foldl_cons]
      rcases member with rfl | member
      · exact le_trans (le_max_right _ _) (initial_le_foldAbsoluteMaximum _ _)
      · exact induction _ member

private theorem constantOutput_norm_le_absoluteMaximum
    (params : Mxx.SamplerParams)
    (values : List Int)
    (modulusPositive : 0 < params.modulus) :
    Mxx.maxCenteredCoefficientNorm (Mxx.Matrix.withSamplerParams {
      coefficients := values.map (Mxx.reduceCoefficient params.modulus)
    } params) ≤
      (values.foldl (fun maximum value => max maximum (absolute value)) 0).toNat := by
  have modulusEq : (params.modulus.toNat : Int) = params.modulus :=
    Int.toNat_of_nonneg modulusPositive.le
  letI : NeZero params.modulus.toNat := ⟨by omega⟩
  unfold Mxx.maxCenteredCoefficientNorm
  apply Mxx.Toolkit.coefficientNorm_le
  intro centered centeredMember
  obtain ⟨coefficient, coefficientMember, rfl⟩ := List.mem_map.mp centeredMember
  simp only [Mxx.Matrix.withSamplerParams, List.mem_append, List.mem_replicate]
    at coefficientMember
  rcases coefficientMember with coefficientMember | ⟨_, rfl⟩
  · have reducedMember : coefficient ∈ values.map (Mxx.reduceCoefficient params.modulus) :=
      List.mem_of_mem_take coefficientMember
    obtain ⟨value, valueMember, rfl⟩ := List.mem_map.mp reducedMember
    change (Mxx.centeredCoefficient params.modulus
      (Mxx.reduceCoefficient params.modulus value)).natAbs ≤ _
    rw [← modulusEq, Mxx.Toolkit.centeredCoefficient_reduce]
    exact le_trans (Mxx.Toolkit.centeredCoefficient_natAbs_le params.modulus.toNat value) (by
      have bound := member_le_foldAbsoluteMaximum 0 value values valueMember
      have absoluteEq : absolute value = |value| := by
        by_cases negative : value < 0
        · simp [absolute, negative, abs_of_neg negative]
        · simp [absolute, negative, abs_of_nonneg (le_of_not_gt negative)]
      rw [absoluteEq, ← Int.natCast_natAbs] at bound
      exact Int.toNat_le_toNat bound)
  · change (Mxx.centeredCoefficient params.modulus 0).natAbs ≤ _
    simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient, not_le_of_gt modulusPositive,
      not_lt_of_ge modulusPositive.le]

private theorem initial_le_foldCanonicalMaximum
    (modulus : Int) (initial : Nat) (values : List Int) :
    initial ≤ values.foldl (fun maximum value =>
      max maximum (Mxx.reduceCoefficient modulus value).toNat) initial := by
  induction values generalizing initial with
  | nil => exact le_rfl
  | cons head tail induction =>
      exact le_trans (le_max_left _ _) (induction _)

private theorem member_le_foldCanonicalMaximum
    (modulus : Int) (initial : Nat) (value : Int) (values : List Int)
    (member : value ∈ values) :
    (Mxx.reduceCoefficient modulus value).toNat ≤
      values.foldl (fun maximum value =>
        max maximum (Mxx.reduceCoefficient modulus value).toNat) initial := by
  induction values generalizing initial with
  | nil => simp at member
  | cons head tail induction =>
      simp only [List.mem_cons] at member
      rw [List.foldl_cons]
      rcases member with rfl | member
      · exact le_trans (le_max_right _ _) (initial_le_foldCanonicalMaximum _ _ _)
      · exact induction _ member

private theorem foldCanonical_le
    (modulus : Int) (bound initial : Nat) (coefficients : List Int)
    (initialBound : initial ≤ bound)
    (coefficientBound : ∀ coefficient ∈ coefficients,
      Mxx.canonicalCoefficient modulus coefficient ≤ bound) :
    coefficients.foldl (fun maximum coefficient =>
      max maximum (Mxx.canonicalCoefficient modulus coefficient)) initial ≤ bound := by
  induction coefficients generalizing initial with
  | nil => exact initialBound
  | cons head tail induction =>
      rw [List.foldl_cons]
      apply induction
      · exact max_le initialBound (coefficientBound head (by simp))
      · intro coefficient member
        exact coefficientBound coefficient (by simp [member])

private theorem constantOutput_canonical_le_declared
    (params : Mxx.SamplerParams)
    (values : List Int)
    (modulusPositive : 0 < params.modulus) :
    Mxx.maxCanonicalCoefficient (Mxx.Matrix.withSamplerParams {
      coefficients := values.map (Mxx.reduceCoefficient params.modulus)
    } params) ≤
      values.foldl (fun maximum value =>
        max maximum (Mxx.reduceCoefficient params.modulus value).toNat) 0 := by
  unfold Mxx.maxCanonicalCoefficient
  apply foldCanonical_le params.modulus _ 0 _ (Nat.zero_le _)
  intro coefficient coefficientMember
  simp only [Mxx.Matrix.withSamplerParams, List.mem_append, List.mem_replicate]
    at coefficientMember
  rcases coefficientMember with coefficientMember | ⟨_, rfl⟩
  · have reducedMember : coefficient ∈ values.map (Mxx.reduceCoefficient params.modulus) :=
      List.mem_of_mem_take coefficientMember
    obtain ⟨value, valueMember, rfl⟩ := List.mem_map.mp reducedMember
    have reducedIdempotent :
        Mxx.reduceCoefficient params.modulus (Mxx.reduceCoefficient params.modulus value) =
          Mxx.reduceCoefficient params.modulus value := by
      simp [Mxx.reduceCoefficient, not_le_of_gt modulusPositive]
    rw [Mxx.canonicalCoefficient, reducedIdempotent]
    exact member_le_foldCanonicalMaximum params.modulus 0 value values valueMember
  · simp [Mxx.canonicalCoefficient, Mxx.reduceCoefficient]

private theorem withSamplerParams_zeroOne_canonical_le_one
    (params : Mxx.SamplerParams)
    (coefficients : List Int)
    (coefficientsZeroOne : ∀ coefficient ∈ coefficients, coefficient = 0 ∨ coefficient = 1)
    (modulusPositive : 0 < params.modulus) :
    Mxx.maxCanonicalCoefficient
      (Mxx.Matrix.withSamplerParams { coefficients } params) ≤ 1 := by
  unfold Mxx.maxCanonicalCoefficient
  apply foldCanonical_le params.modulus 1 0 _ (by omega)
  intro coefficient coefficientMember
  simp only [Mxx.Matrix.withSamplerParams, List.mem_append, List.mem_replicate]
    at coefficientMember
  rcases coefficientMember with coefficientMember | ⟨_, rfl⟩
  · have sourceMember : coefficient ∈ coefficients := List.mem_of_mem_take coefficientMember
    rcases coefficientsZeroOne coefficient sourceMember with rfl | rfl
    · simp [Mxx.canonicalCoefficient, Mxx.reduceCoefficient]
    · simp only [Mxx.canonicalCoefficient]
      by_cases modulusOne : params.modulus = 1
      · simp [Mxx.reduceCoefficient, modulusOne]
      · have oneLt : 1 < params.modulus := by omega
        simp [Mxx.reduceCoefficient, not_le_of_gt modulusPositive,
          Int.emod_eq_of_lt (by omega) oneLt]
  · simp [Mxx.canonicalCoefficient, Mxx.reduceCoefficient]

theorem zeroMatrixFact_holdsEvidence
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (wire : WireRef)
    (matrixType : MatrixTypeExpr)
    (matrixParams : Mxx.SamplerParams)
    (typeEvaluates : matrixType.evaluate parameters = some matrixParams) :
    OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
      subject := wire
      origin := .value address.template wire
      matrixType
      matrixParams
      totalHardBound := (.closedInt (.constant 0))
      canonicalRange := .below 1
    }) (.matrix (zeroConstantOutput matrixParams)) (.matrix {
      originKey := .executionValue address wire (zeroConstantOutput matrixParams)
    }) := by
  apply localMatrixFact_holdsEvidence
  · exact ⟨matrixParams, typeEvaluates,
      by simp [zeroConstantOutput, Mxx.Matrix.withSamplerParams],
      by simp [zeroConstantOutput, Mxx.Matrix.withSamplerParams],
      by simp [zeroConstantOutput, Mxx.Matrix.withSamplerParams],
      by simp [zeroConstantOutput, Mxx.Matrix.withSamplerParams],
      Mxx.Matrix.withSamplerParams_wellFormed _ _⟩
  · simp [zeroConstantOutput, Mxx.Matrix.withSamplerParams]
  · simp [zeroConstantOutput, Mxx.Matrix.withSamplerParams]
  · simp [zeroConstantOutput, Mxx.Matrix.withSamplerParams]
  · simp [zeroConstantOutput, Mxx.Matrix.withSamplerParams]
  · exact ⟨0, by simp, by omega, by simpa using (zeroConstant_norm_eq_zero matrixParams).le⟩
  · exact zeroConstant_canonicalRange matrixParams

theorem constantIntNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (subject : WireRef)
    (origin : OperationalValueOrigin)
    (value : Int)
    (outputCount : Nat)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .constantInt value
      arguments := []
      outputCount
    }) :
    values = [.integer value] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.integer {
        subject
        origin
        lower := value
        upper := value
      }) (.integer value) (.integer value) := by
  exact ⟨constantIntNode_execution runChild samplers parameters inputs wires value outputCount
    member, constantIntegerFact_holdsEvidence samplers parameters states address subject origin value⟩

theorem evaluateIntNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (subject : WireRef)
    (origin : OperationalValueOrigin)
    (expression : IntExpr)
    (value lower upper : Int)
    (outputCount : Nat)
    (evaluates : expression.evaluate parameters = some value)
    (lowerHolds : lower ≤ value)
    (upperHolds : value ≤ upper)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .evaluateInt expression
      arguments := []
      outputCount
    }) :
    values = [.integer value] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.integer {
        subject
        origin
        lower
        upper
      }) (.integer value) (.integer value) := by
  exact ⟨evaluateIntNode_execution runChild samplers parameters inputs wires expression value
    outputCount evaluates member, integerFact_holdsEvidence samplers parameters states address
      subject origin value lower upper lowerHolds upperHolds⟩

theorem constantBoolNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (value : Bool)
    (outputCount : Nat)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .constantBool value
      arguments := []
      outputCount
    }) :
    values = [.boolean value] ∧
      OperationalFact.HoldsEvidence samplers parameters states address .boolean
        (.boolean value) (.boolean value) := by
  exact ⟨constantBoolNode_execution runChild samplers parameters inputs wires value outputCount
    member, booleanFact_holdsEvidence samplers parameters states address value⟩

theorem intBinaryNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (leftRef rightRef subject : WireRef)
    (origin : OperationalValueOrigin)
    (operation : IntBinaryOp)
    (left right result lower upper : Int)
    (outputCount : Nat)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => lookupWire wire wires) =
      some [.integer left, .integer right])
    (operationEvaluate : evaluateIntBinary operation left right = some result)
    (lowerHolds : lower ≤ result)
    (upperHolds : result ≤ upper)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .intBinary operation
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.integer result] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.integer {
        subject
        origin
        lower
        upper
      }) (.integer result) (.integer result) := by
  exact ⟨intBinaryNode_execution runChild samplers parameters inputs wires leftRef rightRef
    operation left right result outputCount argumentsEvaluate operationEvaluate member,
    integerFact_holdsEvidence samplers parameters states address subject origin result lower upper
      lowerHolds upperHolds⟩

theorem intCompareNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (leftRef rightRef : WireRef)
    (operation : IntCompareOp)
    (left right : Int)
    (outputCount : Nat)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => lookupWire wire wires) =
      some [.integer left, .integer right])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .intCompare operation
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    let result := evaluateIntCompare operation left right
    values = [.boolean result] ∧
      OperationalFact.HoldsEvidence samplers parameters states address .boolean
        (.boolean result) (.boolean result) := by
  dsimp
  exact ⟨intCompareNode_execution runChild samplers parameters inputs wires leftRef rightRef
    operation left right outputCount argumentsEvaluate member,
    booleanFact_holdsEvidence samplers parameters states address _⟩

theorem boolToIntNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (inputRef subject : WireRef)
    (origin : OperationalValueOrigin)
    (value : Bool)
    (outputCount : Nat)
    (argumentsEvaluate : [inputRef].mapM (fun wire => lookupWire wire wires) =
      some [.boolean value])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .boolToInt
      arguments := [inputRef]
      outputCount
    }) :
    values = [.integer (if value then 1 else 0)] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.integer {
        subject
        origin
        lower := 0
        upper := 1
      }) (.integer (if value then 1 else 0)) (.integer (if value then 1 else 0)) := by
  exact ⟨boolToIntNode_execution runChild samplers parameters inputs wires inputRef value
    outputCount argumentsEvaluate member,
    booleanToIntegerFact_holdsEvidence samplers parameters states address subject origin value⟩

theorem zeroMatrixNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (wire : WireRef)
    (matrixType : MatrixTypeExpr)
    (matrixParams : Mxx.SamplerParams)
    (outputCount : Nat)
    (typeEvaluates : matrixType.evaluate parameters = some matrixParams)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .zeroMatrix matrixType
      arguments := []
      outputCount
    }) :
    values = [.matrix (zeroConstantOutput matrixParams)] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := wire
        origin := .value address.template wire
        matrixType
        matrixParams
        totalHardBound := (.closedInt (.constant 0))
        canonicalRange := .below 1
      }) (.matrix (zeroConstantOutput matrixParams)) (.matrix {
        originKey := .executionValue address wire (zeroConstantOutput matrixParams)
      }) := by
  exact ⟨by simpa [evaluateNode, typeEvaluates, zeroConstantOutput] using member,
    zeroMatrixFact_holdsEvidence samplers parameters states address wire matrixType matrixParams
      typeEvaluates⟩

theorem identityMatrixNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (wire : WireRef)
    (matrixType : MatrixTypeExpr)
    (matrixParams : Mxx.SamplerParams)
    (outputCount : Nat)
    (typeEvaluates : matrixType.evaluate parameters = some matrixParams)
    (modulusPositive : 0 < matrixParams.modulus)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .identityMatrix matrixType
      arguments := []
      outputCount
    }) :
    let output := identityConstantOutput matrixParams
    values = [.matrix output] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := wire
        origin := .value address.template wire
        matrixType
        matrixParams
        totalHardBound := (.closedInt (.constant 1))
        canonicalRange := .below 2
      }) (.matrix output) (.matrix {
        originKey := .executionValue address wire output
      }) := by
  dsimp
  let output := identityConstantOutput matrixParams
  have valuesExact : values = [.matrix output] := by
    simpa [evaluateNode, typeEvaluates, output, identityConstantOutput] using member
  refine ⟨valuesExact, ?_⟩
  apply localMatrixFact_holdsEvidence
  · exact ⟨matrixParams, typeEvaluates,
      by simp [identityConstantOutput, Mxx.Matrix.withSamplerParams],
      by simp [identityConstantOutput, Mxx.Matrix.withSamplerParams],
      by simp [identityConstantOutput, Mxx.Matrix.withSamplerParams],
      by simp [identityConstantOutput, Mxx.Matrix.withSamplerParams],
      Mxx.Matrix.withSamplerParams_wellFormed _ matrixParams⟩
  · simp [identityConstantOutput, Mxx.Matrix.withSamplerParams]
  · simp [identityConstantOutput, Mxx.Matrix.withSamplerParams]
  · simp [identityConstantOutput, Mxx.Matrix.withSamplerParams]
  · simp [identityConstantOutput, Mxx.Matrix.withSamplerParams]
  · exact ⟨1, by simp, by omega, identityConstant_norm_le_one matrixParams modulusPositive⟩
  · change Mxx.maxCanonicalCoefficient output < 2
    exact lt_of_le_of_lt
      (withSamplerParams_zeroOne_canonical_le_one matrixParams _ (by
        intro value valueMember
        simp only [List.mem_flatMap, List.mem_range, List.mem_map] at valueMember
        obtain ⟨row, _, column, _, coefficient, _, rfl⟩ := valueMember
        by_cases isUnit : row = column ∧ coefficient = 0 <;> simp [isUnit]) modulusPositive)
      (by omega)

theorem constantMatrixNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (wire : WireRef)
    (matrixType : MatrixTypeExpr)
    (coefficients : List IntExpr)
    (runtimeParams factParams : Mxx.SamplerParams)
    (evaluated : List Int)
    (outputCount : Nat)
    (typeEvaluates : matrixType.evaluate parameters = some runtimeParams)
    (coefficientsEvaluate : coefficients.mapM (IntExpr.evaluate parameters) = some evaluated)
    (modulusPositive : 0 < runtimeParams.modulus)
    (modulus : runtimeParams.modulus = factParams.modulus)
    (ringDimension : runtimeParams.ringDimension = factParams.ringDimension)
    (rows : runtimeParams.rows = factParams.rows)
    (columns : runtimeParams.columns = factParams.columns)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .constantMatrix matrixType coefficients
      arguments := []
      outputCount
    }) :
    let bound := evaluated.foldl (fun maximum value => max maximum (absolute value)) 0
    let canonicalMaximum := evaluated.foldl (fun maximum value =>
      max maximum (Mxx.reduceCoefficient runtimeParams.modulus value).toNat) 0
    let output := Mxx.Matrix.withSamplerParams {
      coefficients := evaluated.map (Mxx.reduceCoefficient runtimeParams.modulus)
    } runtimeParams
    values = [.matrix output] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := wire
        origin := .value address.template wire
        matrixType
        matrixParams := factParams
        totalHardBound := (.closedInt (.constant bound))
        canonicalRange := .below (canonicalMaximum + 1)
      }) (.matrix output) (.matrix {
        originKey := .executionValue address wire output
      }) := by
  dsimp
  let output := Mxx.Matrix.withSamplerParams {
    coefficients := evaluated.map (Mxx.reduceCoefficient runtimeParams.modulus)
  } runtimeParams
  have valuesExact : values = [.matrix output] := by
    simpa [evaluateNode, typeEvaluates, coefficientsEvaluate, output] using member
  refine ⟨valuesExact, ?_⟩
  apply localMatrixFact_holdsEvidence
  · exact ⟨runtimeParams, typeEvaluates,
      by simp [Mxx.Matrix.withSamplerParams],
      by simp [Mxx.Matrix.withSamplerParams],
      by simp [Mxx.Matrix.withSamplerParams],
      by simp [Mxx.Matrix.withSamplerParams],
      Mxx.Matrix.withSamplerParams_wellFormed _ _⟩
  · simpa [Mxx.Matrix.withSamplerParams] using modulus
  · simpa [Mxx.Matrix.withSamplerParams] using ringDimension
  · simpa [Mxx.Matrix.withSamplerParams] using rows
  · simpa [Mxx.Matrix.withSamplerParams] using columns
  · refine ⟨_, by simp, initial_le_foldAbsoluteMaximum 0 evaluated, ?_⟩
    exact constantOutput_norm_le_absoluteMaximum runtimeParams evaluated modulusPositive
  · change Mxx.maxCanonicalCoefficient output < _ + 1
    have canonicalBound := constantOutput_canonical_le_declared runtimeParams evaluated
      modulusPositive
    have canonicalBound' : Mxx.maxCanonicalCoefficient output ≤
        evaluated.foldl (fun maximum value =>
          max maximum (Mxx.reduceCoefficient runtimeParams.modulus value).toNat) 0 := by
      simpa [output] using canonicalBound
    omega

theorem gaussianNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (contract : MxxBoundedSamplerContract samplers)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (wire : WireRef)
    (matrixType : MatrixTypeExpr)
    (cutoff : IntExpr)
    (runtimeParams layoutParams factParams : Mxx.SamplerParams)
    (bound : Int)
    (outputCount : Nat)
    (typeEvaluates : matrixType.evaluate parameters cutoff = some runtimeParams)
    (layoutEvaluates : matrixType.evaluate parameters = some layoutParams)
    (cutoffEvaluates : cutoff.evaluate parameters = some bound)
    (boundNonnegative : 0 ≤ bound)
    (modulusPositive : 0 < runtimeParams.modulus)
    (runtimeBound : runtimeParams.maxCoefficientBound = bound.toNat)
    (layoutModulus : runtimeParams.modulus = layoutParams.modulus)
    (layoutRingDimension : runtimeParams.ringDimension = layoutParams.ringDimension)
    (layoutRows : runtimeParams.rows = layoutParams.rows)
    (layoutColumns : runtimeParams.columns = layoutParams.columns)
    (modulus : runtimeParams.modulus = factParams.modulus)
    (ringDimension : runtimeParams.ringDimension = factParams.ringDimension)
    (rows : runtimeParams.rows = factParams.rows)
    (columns : runtimeParams.columns = factParams.columns)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .gaussianSample matrixType cutoff
      arguments := []
      outputCount
    }) :
    ∃ output,
      values = [.matrix output] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := wire
        origin := .value address.template wire
        matrixType
        matrixParams := factParams
        totalHardBound := (.minimum
          (.closedInt (.constant (runtimeParams.modulus / 2)))
          (.contextual .maximum parameters [] cutoff))
      }) (.matrix output) (.matrix {
        originKey := .executionValue address wire output
      }) := by
  obtain ⟨sample, sampleMember, rfl⟩ := Mxx.Ir.mem_evaluateNode_gaussianSample runChild
    samplers parameters inputs wires matrixType cutoff runtimeParams outputCount typeEvaluates member
  let output := sample.withSamplerParams runtimeParams
  have outputNorm : Mxx.maxCenteredCoefficientNorm output ≤ runtimeParams.maxCoefficientBound :=
    contract.gaussianHardSupport runtimeParams sample sampleMember
  refine ⟨output, rfl, ?_⟩
  apply localMatrixFact_holdsEvidence
  · refine ⟨layoutParams, layoutEvaluates, ?_, ?_, ?_, ?_, ?_⟩
    · simp [output, Mxx.Matrix.withSamplerParams]
      exact layoutModulus
    · simp [output, Mxx.Matrix.withSamplerParams]
      exact layoutRingDimension
    · simp [output, Mxx.Matrix.withSamplerParams]
      exact layoutRows
    · simp [output, Mxx.Matrix.withSamplerParams]
      exact layoutColumns
    · exact Mxx.Matrix.withSamplerParams_wellFormed _ _
  · simpa [output, Mxx.Matrix.withSamplerParams] using modulus
  · simpa [output, Mxx.Matrix.withSamplerParams] using ringDimension
  · simpa [output, Mxx.Matrix.withSamplerParams] using rows
  · simpa [output, Mxx.Matrix.withSamplerParams] using columns
  · let cappedBound := min (runtimeParams.modulus / 2) bound
    refine ⟨cappedBound, ?_, ?_, ?_⟩
    · simp [cappedBound, OperationalBoundExpr.evaluateWithStates,
        OperationalBoundExpr.evaluateWithStates_contextualMaximum_nil, cutoffEvaluates]
      rfl
    · exact le_min (Int.ediv_nonneg modulusPositive.le (by omega))
        boundNonnegative
    · apply (Int.le_toNat (le_min
          (Int.ediv_nonneg modulusPositive.le (by omega)) boundNonnegative)).2
      have radius := matrix_norm_le_centered_radius output (by
        simpa [output, Mxx.Matrix.withSamplerParams] using modulusPositive)
      have outputModulus : output.modulus = runtimeParams.modulus := by
        simp [output, Mxx.Matrix.withSamplerParams]
      rw [outputModulus] at radius
      have cutoffNat : Mxx.maxCenteredCoefficientNorm output ≤ bound.toNat := by
        rw [← runtimeBound]
        exact outputNorm
      have radiusInt : (Mxx.maxCenteredCoefficientNorm output : Int) ≤
          runtimeParams.modulus / 2 := by
        omega
      have cutoffInt : (Mxx.maxCenteredCoefficientNorm output : Int) ≤ bound := by
        omega
      exact le_min radiusInt cutoffInt
  · trivial

theorem plainHashNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (keyRef outputWire : WireRef)
    (trailingRefs : List WireRef)
    (keyOrigin : OperationalValueOrigin)
    (trailingOrigins : List OperationalValueOrigin)
    (key : ByteArray)
    (trailingValues : List Int)
    (matrixType : MatrixTypeExpr)
    (identityParameters : ParamEnvironment)
    (identityDomains currentDomains : List OperationalParameterDomain)
    (tagPrefix : List Nat)
    (tagExpressions tagDecimalExpressions tagU64LeExpressions : List IntExpr)
    (runtimeParams layoutParams factParams : Mxx.SamplerParams)
    (tagValues tagDecimalValues tagU64LeValues : List Int)
    (raw : DeterministicHashIdentity)
    (query : Mxx.HashQuery)
    (outputCount : Nat)
    (rawExact : raw = {
      keyOrigin
      matrixType
      parameterEnvironment := identityParameters
      parameterDomains := identityDomains
      tagPrefix
      tagExpressions
      tagDecimalExpressions
      tagU64LeExpressions
      trailingIntegerOrigins := trailingOrigins
    })
    (queryExact : query = {
      params := runtimeParams
      key
      variant := .plain
      tagPrefix
      tagValues
      tagDecimalValues
      tagU64LeValues
      trailingIntegerTagValues := trailingValues
      base := none
      digitCount := none
    })
    (contextExact : address.instantiateHashContext identityParameters identityDomains =
      .ok (parameters, currentDomains))
    (argumentsEvaluate : (keyRef :: trailingRefs).mapM (fun wire => lookupWire wire wires) =
      some (.bytes key :: trailingValues.map Value.integer))
    (matrixTypeEvaluate : matrixType.evaluate parameters (.constant 0) = some runtimeParams)
    (layoutEvaluate : matrixType.evaluate parameters = some layoutParams)
    (tagsEvaluate : tagExpressions.mapM (IntExpr.evaluate parameters) = some tagValues)
    (decimalTagsEvaluate :
      tagDecimalExpressions.mapM (IntExpr.evaluate parameters) = some tagDecimalValues)
    (u64TagsEvaluate :
      tagU64LeExpressions.mapM (IntExpr.evaluate parameters) = some tagU64LeValues)
    (tagsInvariantEvaluate :
      tagExpressions.mapM (evaluateIntInvariant parameters currentDomains) = .ok tagValues)
    (decimalTagsInvariantEvaluate :
      tagDecimalExpressions.mapM (evaluateIntInvariant parameters currentDomains) =
        .ok tagDecimalValues)
    (u64TagsInvariantEvaluate :
      tagU64LeExpressions.mapM (evaluateIntInvariant parameters currentDomains) =
        .ok tagU64LeValues)
    (modulusPositive : 0 < runtimeParams.modulus)
    (layoutModulus : runtimeParams.modulus = layoutParams.modulus)
    (layoutRingDimension : runtimeParams.ringDimension = layoutParams.ringDimension)
    (layoutRows : runtimeParams.rows = layoutParams.rows)
    (layoutColumns : runtimeParams.columns = layoutParams.columns)
    (factModulus : runtimeParams.modulus = factParams.modulus)
    (factRingDimension : runtimeParams.ringDimension = factParams.ringDimension)
    (factRows : runtimeParams.rows = factParams.rows)
    (factColumns : runtimeParams.columns = factParams.columns)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .hashSample matrixType .plain tagPrefix tagExpressions tagDecimalExpressions
        tagU64LeExpressions none none
      arguments := keyRef :: trailingRefs
      outputCount
    }) :
    values = [.matrix ((samplers.hashSample query).withSamplerParams runtimeParams)] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := outputWire
        origin := .deterministicHash raw
        matrixType
        matrixParams := factParams
        totalHardBound := (.closedInt (.constant (runtimeParams.modulus / 2)))
      }) (.matrix ((samplers.hashSample query).withSamplerParams runtimeParams)) (.matrix {
        originKey := .deterministicHash query
        hashOrigin := some {
          address
          keyOrigin
          key
          trailingOrigins
          trailingValues
        }
      }) := by
  subst raw
  subst query
  let query : Mxx.HashQuery := {
    params := runtimeParams
    key
    variant := .plain
    tagPrefix
    tagValues
    tagDecimalValues
    tagU64LeValues
    trailingIntegerTagValues := trailingValues
    base := none
    digitCount := none
  }
  let output := (samplers.hashSample query).withSamplerParams runtimeParams
  have valuesExact : values = [.matrix output] := by
    simpa [query, output] using
      (mem_evaluateNode_hashSample_of_arguments runChild samplers parameters inputs wires keyRef
        trailingRefs key trailingValues matrixType .plain tagPrefix tagExpressions
        tagDecimalExpressions tagU64LeExpressions none none runtimeParams tagValues
        tagDecimalValues tagU64LeValues none none outputCount argumentsEvaluate
        matrixTypeEvaluate tagsEvaluate decimalTagsEvaluate u64TagsEvaluate rfl rfl member)
  refine ⟨valuesExact, ?_⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨layoutParams, layoutEvaluate, ?_, ?_, ?_, ?_, ?_⟩
    · exact (by rfl : output.modulus = runtimeParams.modulus).trans layoutModulus
    · exact (by rfl : output.ringDimension = runtimeParams.ringDimension).trans
        layoutRingDimension
    · exact (by rfl : output.rows = runtimeParams.rows).trans layoutRows
    · exact (by rfl : output.columns = runtimeParams.columns).trans layoutColumns
    · exact Mxx.Matrix.withSamplerParams_wellFormed _ _
  · exact (by rfl : output.modulus = runtimeParams.modulus).trans factModulus
  · exact (by rfl : output.ringDimension = runtimeParams.ringDimension).trans factRingDimension
  · exact (by rfl : output.rows = runtimeParams.rows).trans factRows
  · exact (by rfl : output.columns = runtimeParams.columns).trans factColumns
  · apply deterministicHashOrigin_holds
    unfold instantiateHashQuery
    rw [contextExact]
    simp
    rw [show (Except.ok (parameters, currentDomains)).toOption =
      some (parameters, currentDomains) by rfl]
    simp only [Option.bind_some]
    rw [matrixTypeEvaluate]
    simp only [Option.bind_some]
    rw [tagsInvariantEvaluate, decimalTagsInvariantEvaluate, u64TagsInvariantEvaluate]
    rfl
  · refine ⟨runtimeParams.modulus / 2, ?_,
      Int.ediv_nonneg modulusPositive.le (by omega), ?_⟩
    · simp
    have radius := matrix_norm_le_centered_radius output (by
      change 0 < runtimeParams.modulus
      exact modulusPositive)
    have outputModulus : output.modulus = runtimeParams.modulus := by rfl
    rw [outputModulus] at radius
    have radiusInt : (Mxx.maxCenteredCoefficientNorm output : Int) ≤
        runtimeParams.modulus / 2 := by
      omega
    exact (Int.le_toNat (Int.ediv_nonneg modulusPositive.le (by omega))).2 radiusInt
  · trivial
  · trivial
  · exact List.Forall₂.nil

/-- A validated deterministic-hash identity supplies the exact target summary used by a
decomposed-hash relation, even when the executable graph does not materialize the matching plain
hash as a separate node. -/
theorem deterministicHashTarget_operational_sound
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (raw : DeterministicHashIdentity)
    (hashEvidence : DeterministicHashEvidence)
    (plain : Mxx.HashQuery)
    (matrixType : MatrixTypeExpr)
    (typeHolds : matrixType.Holds parameters
      ((samplers.hashSample plain).withSamplerParams plain.params))
    (queryExact : instantiateHashQuery raw hashEvidence.address hashEvidence.keyOrigin
      hashEvidence.key hashEvidence.trailingOrigins hashEvidence.trailingValues = some plain)
    (modulusPositive : 0 < plain.params.modulus) :
    RelationTargetSummary.HoldsEvidence samplers parameters states {
      origin := .deterministicHash raw
      matrixType
      matrixParams := plain.params
      totalHardBound := (.closedInt (.constant (plain.params.modulus / 2)))
      canonicalRange := .unknown
    } {
      key := .deterministicHash plain
      matrix := (samplers.hashSample plain).withSamplerParams plain.params
      hashOrigin := some hashEvidence
    } := by
  let output := (samplers.hashSample plain).withSamplerParams plain.params
  refine ⟨deterministicHashOrigin_holds samplers raw hashEvidence plain queryExact,
    Mxx.MatrixModEq.refl output, typeHolds, rfl, rfl, rfl, rfl, ?_, trivial⟩
  refine ⟨plain.params.modulus / 2, by simp,
    Int.ediv_nonneg modulusPositive.le (by omega), ?_⟩
  have radius := matrix_norm_le_centered_radius output (by
    change 0 < plain.params.modulus
    exact modulusPositive)
  have outputModulus : output.modulus = plain.params.modulus := by rfl
  rw [outputModulus] at radius
  have radiusInt : (Mxx.maxCenteredCoefficientNorm output : Int) ≤
      plain.params.modulus / 2 := by
    omega
  exact (Int.le_toNat (Int.ediv_nonneg modulusPositive.le (by omega))).2 radiusInt

def decomposedHashOperationalFact
    (address : ExecutionAddress)
    (outputWire : WireRef)
    (matrixType plainMatrixType : MatrixTypeExpr)
    (paramsId : Mxx.SamplerParamsId)
    (plain decomposed : Mxx.HashQuery)
    (raw : DeterministicHashIdentity)
    (base : Int)
    (small : Bool)
    (digitCount : Nat)
    (status : ReconstructionStatus) : OperationalFact := .matrix {
  subject := outputWire
  origin := .value address.template outputWire
  matrixType
  matrixParams := decomposed.params
  totalHardBound := (.minimum
    (.closedInt (.constant (decomposed.params.modulus / 2)))
    (.closedInt (.constant (Int.ofNat (Mxx.gadgetDecompositionBound base small)))))
  canonicalRange := if small then .below base.natAbs else .unknown
  relations := [.decomposition {
    producer := .value address.template outputWire
    publicIdentity := .gadget paramsId decomposed.params plain.params.rows base small digitCount
    inputOrigin := .deterministicHash raw
    inputSummary := {
      origin := .deterministicHash raw
      matrixType := plainMatrixType
      matrixParams := plain.params
      totalHardBound := (.closedInt (.constant (plain.params.modulus / 2)))
      canonicalRange := .unknown
    }
    base
    small
    digitCount
    status
  }]
}

def decomposedHashOperationalEvidence
    (samplers : MxxSamplerFamily)
    (address : ExecutionAddress)
    (outputWire : WireRef)
    (paramsId : Mxx.SamplerParamsId)
    (plain decomposed : Mxx.HashQuery)
    (hashEvidence : DeterministicHashEvidence)
    (base : Int)
    (small : Bool)
    (digitCount : Nat)
    (publicMatrix : Mxx.Matrix) : RuntimeFactEvidence := .matrix {
  originKey := .executionValue address outputWire
    ((samplers.hashSample decomposed).withSamplerParams decomposed.params)
  relations := [.decomposition {
    producerKey := .executionValue address outputWire
      ((samplers.hashSample decomposed).withSamplerParams decomposed.params)
    publicKey := .gadget paramsId decomposed.params plain.params.rows base small digitCount
    input := {
      key := .deterministicHash plain
      matrix := (samplers.hashSample plain).withSamplerParams plain.params
      hashOrigin := some hashEvidence
    }
    publicMatrix
    decomposition := (samplers.hashSample decomposed).withSamplerParams decomposed.params
  }]
}

theorem preimageNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (contract : MxxBoundedSamplerContract samplers)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (publicRef trapdoorRef targetRef outputWire : WireRef)
    (publicMatrix target : Mxx.Matrix)
    (publicIdentity : PublicMatrixIdentity)
    (publicKey : SemanticMatrixKey)
    (targetSummary : RelationTargetSummary)
    (targetEvidence : RelationTargetEvidence)
    (matrixType : MatrixTypeExpr)
    (cutoff : IntExpr)
    (runtimeParams layoutParams factParams : Mxx.SamplerParams)
    (bound : Int)
    (outputCount : Nat)
    (typeEvaluates : matrixType.evaluate parameters cutoff = some runtimeParams)
    (layoutEvaluates : matrixType.evaluate parameters = some layoutParams)
    (cutoffEvaluates : cutoff.evaluate parameters = some bound)
    (boundNonnegative : 0 ≤ bound)
    (modulusPositive : 0 < runtimeParams.modulus)
    (runtimeBound : runtimeParams.maxCoefficientBound = bound.toNat)
    (layoutModulus : runtimeParams.modulus = layoutParams.modulus)
    (layoutRingDimension : runtimeParams.ringDimension = layoutParams.ringDimension)
    (layoutRows : runtimeParams.rows = layoutParams.rows)
    (layoutColumns : runtimeParams.columns = layoutParams.columns)
    (factModulus : runtimeParams.modulus = factParams.modulus)
    (factRingDimension : runtimeParams.ringDimension = factParams.ringDimension)
    (factRows : runtimeParams.rows = factParams.rows)
    (factColumns : runtimeParams.columns = factParams.columns)
    (publicIdentityHolds : publicIdentity.HoldsEvidence samplers publicKey publicMatrix)
    (publicKeyHolds : publicKey.Holds samplers publicMatrix)
    (targetMatrix : targetEvidence.matrix = target)
    (targetSummaryHolds :
      targetSummary.HoldsEvidence samplers parameters states targetEvidence)
    (argumentsEvaluate :
      [publicRef, trapdoorRef, targetRef].mapM (fun wire => lookupWire wire wires) =
        some [.matrix publicMatrix, .trapdoor publicMatrix .sampled, .matrix target])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .preimageSample matrixType cutoff
      arguments := [publicRef, trapdoorRef, targetRef]
      outputCount
    }) :
    ∃ output,
      values = [.matrix output] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := outputWire
        origin := .value address.template outputWire
        matrixType
        matrixParams := factParams
        totalHardBound := (.minimum
          (.closedInt (.constant (runtimeParams.modulus / 2)))
          (.contextual .maximum parameters [] cutoff))
        relations := [.preimage {
          producer := .value address.template outputWire
          publicIdentity
          targetOrigin := targetSummary.origin
          targetSummary
        }]
      }) (.matrix output) (.matrix {
        originKey := .executionValue address outputWire output
        relations := [.preimage {
          producerKey := .executionValue address outputWire output
          publicKey
          target := targetEvidence
          publicMatrix
          preimage := output
        }]
      }) := by
  obtain ⟨output, valuesExact, relation, outputNorm, outputWellFormed, outputModulus,
    outputRingDimension, outputRows, outputColumns⟩ := preimageNode_local_sound runChild
    samplers contract parameters inputs wires publicRef trapdoorRef targetRef publicMatrix target
    matrixType cutoff runtimeParams outputCount argumentsEvaluate typeEvaluates member
  refine ⟨output, valuesExact, ?_⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨layoutParams, layoutEvaluates, ?_, ?_, ?_, ?_, ?_⟩
    · exact outputModulus.trans layoutModulus
    · exact outputRingDimension.trans layoutRingDimension
    · exact outputRows.trans layoutRows
    · exact outputColumns.trans layoutColumns
    · exact outputWellFormed
  · exact outputModulus.trans factModulus
  · exact outputRingDimension.trans factRingDimension
  · exact outputRows.trans factRows
  · exact outputColumns.trans factColumns
  · exact ⟨rfl, rfl, Mxx.MatrixModEq.refl output⟩
  · let cappedBound := min (runtimeParams.modulus / 2) bound
    refine ⟨cappedBound, ?_, ?_, ?_⟩
    · simp [cappedBound, OperationalBoundExpr.evaluateWithStates,
        OperationalBoundExpr.evaluateWithStates_contextualMaximum_nil, cutoffEvaluates]
      rfl
    · exact le_min (Int.ediv_nonneg modulusPositive.le (by omega)) boundNonnegative
    · apply (Int.le_toNat (le_min
          (Int.ediv_nonneg modulusPositive.le (by omega)) boundNonnegative)).2
      have radius := matrix_norm_le_centered_radius output
        (outputModulus.trans_gt modulusPositive)
      rw [outputModulus] at radius
      have cutoffNat : Mxx.maxCenteredCoefficientNorm output ≤ bound.toNat := by
        rw [← runtimeBound]
        exact outputNorm
      have radiusInt : (Mxx.maxCenteredCoefficientNorm output : Int) ≤
          runtimeParams.modulus / 2 := by omega
      have cutoffInt : (Mxx.maxCenteredCoefficientNorm output : Int) ≤ bound := by omega
      exact le_min radiusInt cutoffInt
  · trivial
  · trivial
  · apply List.Forall₂.cons
    · refine ⟨?_, publicIdentityHolds, targetSummaryHolds, ?_, publicKeyHolds, ?_, ?_⟩
      · exact ⟨rfl, rfl, Mxx.MatrixModEq.refl output⟩
      · exact Mxx.MatrixModEq.refl output
      · simpa [targetMatrix] using targetSummaryHolds.2.1
      · simpa [targetMatrix] using relation
    · exact List.Forall₂.nil

theorem gadgetDecomposeRelation_operational_sound
    (samplers : MxxSamplerFamily)
    (contract : MxxBoundedSamplerContract samplers)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (paramsId : Mxx.SamplerParamsId)
    (params : Mxx.SamplerParams)
    (base : Int)
    (small : Bool)
    (digitCount : Nat)
    (producer : MatrixOriginIdentity)
    (producerKey : SemanticMatrixKey)
    (inputSummary : RelationTargetSummary)
    (inputEvidence : RelationTargetEvidence)
    (publicMatrix rawOutput : Mxx.Matrix)
    (status : ReconstructionStatus)
    (producerHolds : producer.HoldsEvidence samplers producerKey none
      (rawOutput.withSamplerParams params))
    (producerKeyHolds : producerKey.Holds samplers (rawOutput.withSamplerParams params))
    (inputSummaryHolds : inputSummary.HoldsEvidence samplers parameters states inputEvidence)
    (publicResult : samplers.gadgetPublicMatrix paramsId params inputEvidence.matrix.rows base
      small digitCount = some publicMatrix)
    (decompositionResult : samplers.gadgetDecompose paramsId params base small digitCount
      inputEvidence.matrix = some rawOutput)
    (statusSound : status = .available →
      (small = false ∨ ∃ limit,
        samplers.smallDecompositionInputLimit paramsId params = some limit ∧
        Mxx.maxCanonicalCoefficient inputEvidence.matrix < limit))
    (publicKeyHolds : (SemanticMatrixKey.gadget paramsId params inputEvidence.matrix.rows base
      small digitCount).Holds samplers publicMatrix) :
    MatrixRelationEvidence.Holds samplers parameters states (.decomposition {
      producer
      publicIdentity := .gadget paramsId params inputEvidence.matrix.rows base small digitCount
      inputOrigin := inputSummary.origin
      inputSummary
      base
      small
      digitCount
      status
    }) (.decomposition {
      producerKey
      publicKey := .gadget paramsId params inputEvidence.matrix.rows base small digitCount
      input := inputEvidence
      publicMatrix
      decomposition := rawOutput.withSamplerParams params
    }) := by
  apply MatrixRelationEvidence.decomposition_holds producerHolds
  · exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, publicKeyHolds⟩
  · exact inputSummaryHolds
  · exact producerKeyHolds
  · exact publicKeyHolds
  · exact inputSummaryHolds.2.1
  · intro available
    exact (gadgetDecompose_local_sound samplers contract paramsId params base small digitCount
      inputEvidence.matrix publicMatrix rawOutput publicResult decompositionResult
      (statusSound available)).1

theorem gadgetDecompositionFact_operational_sound
    (samplers : MxxSamplerFamily)
    (contract : MxxBoundedSamplerContract samplers)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (outputWire : WireRef)
    (matrixType : MatrixTypeExpr)
    (paramsId : Mxx.SamplerParamsId)
    (params : Mxx.SamplerParams)
    (base : Int)
    (small : Bool)
    (digitCount : Nat)
    (inputSummary : RelationTargetSummary)
    (inputEvidence : RelationTargetEvidence)
    (publicMatrix rawOutput : Mxx.Matrix)
    (status : ReconstructionStatus)
    (typeHolds : matrixType.Holds parameters (rawOutput.withSamplerParams params))
    (modulusPositive : 0 < params.modulus)
    (inputSummaryHolds : inputSummary.HoldsEvidence samplers parameters states inputEvidence)
    (publicResult : samplers.gadgetPublicMatrix paramsId params inputEvidence.matrix.rows base
      small digitCount = some publicMatrix)
    (decompositionResult : samplers.gadgetDecompose paramsId params base small digitCount
      inputEvidence.matrix = some rawOutput)
    (statusSound : status = .available →
      (small = false ∨ ∃ limit,
        samplers.smallDecompositionInputLimit paramsId params = some limit ∧
        Mxx.maxCanonicalCoefficient inputEvidence.matrix < limit))
    (publicKeyHolds : (SemanticMatrixKey.gadget paramsId params inputEvidence.matrix.rows base
      small digitCount).Holds samplers publicMatrix) :
    OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
      subject := outputWire
      origin := .value address.template outputWire
      matrixType
      matrixParams := params
      totalHardBound := (.minimum
        (.closedInt (.constant (params.modulus / 2)))
        (.closedInt (.constant (Int.ofNat (Mxx.gadgetDecompositionBound base small)))))
      canonicalRange := if small then .below base.natAbs else .unknown
      relations := [.decomposition {
        producer := .value address.template outputWire
        publicIdentity := .gadget paramsId params inputEvidence.matrix.rows base small digitCount
        inputOrigin := inputSummary.origin
        inputSummary
        base
        small
        digitCount
        status
      }]
    }) (.matrix (rawOutput.withSamplerParams params)) (.matrix {
      originKey := .executionValue address outputWire (rawOutput.withSamplerParams params)
      relations := [.decomposition {
        producerKey := .executionValue address outputWire (rawOutput.withSamplerParams params)
        publicKey := .gadget paramsId params inputEvidence.matrix.rows base small digitCount
        input := inputEvidence
        publicMatrix
        decomposition := rawOutput.withSamplerParams params
      }]
    }) := by
  let normalized := rawOutput.withSamplerParams params
  let maximum := OperationalBoundExpr.minimum
    (.closedInt (.constant (params.modulus / 2)))
    (.closedInt (.constant (Int.ofNat (Mxx.gadgetDecompositionBound base small))))
  let baseFact : OperationalMatrixFact := {
    subject := outputWire
    origin := .value address.template outputWire
    matrixType
    matrixParams := params
    magnitude := .finite maximum
    canonicalRange := if small then .below base.natAbs else .unknown
  }
  let baseEvidence : MatrixFactEvidence := {
    originKey := .executionValue address outputWire normalized
  }
  have normBound := contract.gadgetDecomposeHardBound paramsId params base small digitCount
    inputEvidence.matrix rawOutput decompositionResult
  have baseHolds : OperationalFact.HoldsEvidence samplers parameters states address
      (.matrix baseFact) (.matrix normalized) (.matrix baseEvidence) := by
    apply localMatrixFact_holdsEvidence
    · exact typeHolds
    · rfl
    · rfl
    · rfl
    · rfl
    · let capped := min (params.modulus / 2)
          (Int.ofNat (Mxx.gadgetDecompositionBound base small))
      refine ⟨capped, ?_, ?_, ?_⟩
      · simp only [maximum, OperationalBoundExpr.evaluateWithStates,
          OperationalBoundExpr.evaluateWithStates_closedConstant]
        rfl
      · exact le_min (Int.ediv_nonneg modulusPositive.le (by omega))
          (Int.natCast_nonneg _)
      · apply (Int.le_toNat (le_min
            (Int.ediv_nonneg modulusPositive.le (by omega)) (Int.natCast_nonneg _))).2
        have normalizedModulus : normalized.modulus = params.modulus := by rfl
        have radius := matrix_norm_le_centered_radius normalized
          (normalizedModulus.trans_gt modulusPositive)
        rw [normalizedModulus] at radius
        have radiusInt : (Mxx.maxCenteredCoefficientNorm normalized : Int) ≤
            params.modulus / 2 := by omega
        have boundInt : (Mxx.maxCenteredCoefficientNorm normalized : Int) ≤
            Int.ofNat (Mxx.gadgetDecompositionBound base small) := by
          change Mxx.maxCenteredCoefficientNorm normalized ≤
            Mxx.gadgetDecompositionBound base small at normBound
          exact Int.ofNat_le.mpr normBound
        exact le_min radiusInt boundInt
    · cases small with
      | false => trivial
      | true =>
          change Mxx.maxCanonicalCoefficient normalized < base.natAbs
          exact gadgetDecomposeSmall_canonical_range samplers contract paramsId params base
            digitCount inputEvidence.matrix rawOutput decompositionResult
  have relationHolds : List.Forall₂
      (MatrixRelationEvidence.Holds samplers parameters states)
      [.decomposition {
        producer := .value address.template outputWire
        publicIdentity := .gadget paramsId params inputEvidence.matrix.rows base small digitCount
        inputOrigin := inputSummary.origin
        inputSummary
        base
        small
        digitCount
        status
      }]
      [.decomposition {
        producerKey := .executionValue address outputWire normalized
        publicKey := .gadget paramsId params inputEvidence.matrix.rows base small digitCount
        input := inputEvidence
        publicMatrix
        decomposition := normalized
      }] := by
    apply List.Forall₂.cons
    · apply gadgetDecomposeRelation_operational_sound samplers contract parameters states paramsId
        params base small digitCount (.value address.template outputWire)
        (.executionValue address outputWire normalized) inputSummary inputEvidence publicMatrix
        rawOutput status
      · exact ⟨rfl, rfl, Mxx.MatrixModEq.refl normalized⟩
      · exact Mxx.MatrixModEq.refl normalized
      · exact inputSummaryHolds
      · exact publicResult
      · exact decompositionResult
      · exact statusSound
      · exact publicKeyHolds
    · exact List.Forall₂.nil
  have result := OperationalFact.HoldsEvidence.withMatrixRelations (fact := baseFact)
    (evidence := baseEvidence) baseHolds relationHolds
  simpa [baseFact, baseEvidence, normalized, maximum] using result

/-- A decomposed deterministic hash is exactly the backend decomposition of the matching plain
query and therefore carries the same operational decomposition relation as an explicit
`gadgetDecompose` node. -/
theorem decomposedHashFact_operational_sound
    (samplers : MxxSamplerFamily)
    (contract : MxxBoundedSamplerContract samplers)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (outputWire : WireRef)
    (matrixType plainMatrixType : MatrixTypeExpr)
    (paramsId : Mxx.SamplerParamsId)
    (plain decomposed : Mxx.HashQuery)
    (raw : DeterministicHashIdentity)
    (hashEvidence : DeterministicHashEvidence)
    (base : Int)
    (small : Bool)
    (digitCount : Nat)
    (publicMatrix : Mxx.Matrix)
    (status : ReconstructionStatus)
    (queryExact : instantiateHashQuery raw hashEvidence.address hashEvidence.keyOrigin
      hashEvidence.key hashEvidence.trailingOrigins hashEvidence.trailingValues = some plain)
    (queriesMatch : Mxx.HashQueriesMatchDecomposition plain decomposed base small digitCount)
    (typeHolds : matrixType.Holds parameters
      ((samplers.hashSample decomposed).withSamplerParams decomposed.params))
    (plainTypeHolds : plainMatrixType.Holds parameters
      ((samplers.hashSample plain).withSamplerParams plain.params))
    (modulusPositive : 0 < decomposed.params.modulus)
    (plainModulusPositive : 0 < plain.params.modulus)
    (layoutId : samplers.layoutId decomposed.params = some paramsId)
    (publicResult : samplers.gadgetPublicMatrix paramsId decomposed.params plain.params.rows
      base small digitCount = some publicMatrix)
    (statusSound : status = .available →
      (small = false ∨ ∃ limit,
        samplers.smallDecompositionInputLimit paramsId decomposed.params = some limit ∧
        Mxx.maxCanonicalCoefficient
          ((samplers.hashSample plain).withSamplerParams plain.params) < limit))
    (publicKeyHolds : (SemanticMatrixKey.gadget paramsId decomposed.params plain.params.rows
      base small digitCount).Holds samplers publicMatrix) :
    OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
      subject := outputWire
      origin := .value address.template outputWire
      matrixType
      matrixParams := decomposed.params
      totalHardBound := (.minimum
        (.closedInt (.constant (decomposed.params.modulus / 2)))
        (.closedInt (.constant (Int.ofNat (Mxx.gadgetDecompositionBound base small)))))
      canonicalRange := if small then .below base.natAbs else .unknown
      relations := [.decomposition {
        producer := .value address.template outputWire
        publicIdentity := .gadget paramsId decomposed.params plain.params.rows base small digitCount
        inputOrigin := .deterministicHash raw
        inputSummary := {
          origin := .deterministicHash raw
          matrixType := plainMatrixType
          matrixParams := plain.params
          totalHardBound := (.closedInt (.constant (plain.params.modulus / 2)))
          canonicalRange := .unknown
        }
        base
        small
        digitCount
        status
      }]
    }) (.matrix ((samplers.hashSample decomposed).withSamplerParams decomposed.params)) (.matrix {
      originKey := .executionValue address outputWire
        ((samplers.hashSample decomposed).withSamplerParams decomposed.params)
      relations := [.decomposition {
        producerKey := .executionValue address outputWire
          ((samplers.hashSample decomposed).withSamplerParams decomposed.params)
        publicKey := .gadget paramsId decomposed.params plain.params.rows base small digitCount
        input := {
          key := .deterministicHash plain
          matrix := (samplers.hashSample plain).withSamplerParams plain.params
          hashOrigin := some hashEvidence
        }
        publicMatrix
        decomposition :=
          (samplers.hashSample decomposed).withSamplerParams decomposed.params
      }]
    }) := by
  let inputSummary : RelationTargetSummary := {
    origin := .deterministicHash raw
    matrixType := plainMatrixType
    matrixParams := plain.params
    totalHardBound := (.closedInt (.constant (plain.params.modulus / 2)))
    canonicalRange := .unknown
  }
  let inputEvidence : RelationTargetEvidence := {
    key := .deterministicHash plain
    matrix := (samplers.hashSample plain).withSamplerParams plain.params
    hashOrigin := some hashEvidence
  }
  have inputSummaryHolds : inputSummary.HoldsEvidence samplers parameters states
      inputEvidence := by
    exact deterministicHashTarget_operational_sound samplers parameters states raw hashEvidence
      plain plainMatrixType plainTypeHolds queryExact plainModulusPositive
  have decompositionResult := contract.decomposedHashConsistency paramsId plain decomposed base
    small digitCount layoutId queriesMatch
  have result := gadgetDecompositionFact_operational_sound samplers contract parameters states
    address outputWire matrixType paramsId decomposed.params base small digitCount inputSummary
    inputEvidence publicMatrix (samplers.hashSample decomposed) status typeHolds modulusPositive
    inputSummaryHolds publicResult decompositionResult statusSound publicKeyHolds
  simpa [inputSummary, inputEvidence, Mxx.Matrix.withSamplerParams] using result

theorem decomposedHashNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (contract : MxxBoundedSamplerContract samplers)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (keyRef outputWire : WireRef)
    (trailingRefs : List WireRef)
    (key : ByteArray)
    (trailingValues : List Int)
    (matrixType plainMatrixType : MatrixTypeExpr)
    (tagPrefix : List Nat)
    (tagExpressions tagDecimalExpressions tagU64LeExpressions : List IntExpr)
    (baseExpr digitCountExpr : IntExpr)
    (runtimeParams : Mxx.SamplerParams)
    (tagValues tagDecimalValues tagU64LeValues : List Int)
    (paramsId : Mxx.SamplerParamsId)
    (plain decomposed : Mxx.HashQuery)
    (raw : DeterministicHashIdentity)
    (hashEvidence : DeterministicHashEvidence)
    (base : Int)
    (small : Bool)
    (digitCount : Nat)
    (publicMatrix : Mxx.Matrix)
    (status : ReconstructionStatus)
    (outputCount : Nat)
    (decomposedExact : decomposed = {
      params := runtimeParams
      key
      variant := if small then .smallDecomposed else .decomposed
      tagPrefix
      tagValues
      tagDecimalValues
      tagU64LeValues
      trailingIntegerTagValues := trailingValues
      base := some base
      digitCount := some (Int.ofNat digitCount)
    })
    (queryExact : instantiateHashQuery raw hashEvidence.address hashEvidence.keyOrigin
      hashEvidence.key hashEvidence.trailingOrigins hashEvidence.trailingValues = some plain)
    (queriesMatch : Mxx.HashQueriesMatchDecomposition plain decomposed base small digitCount)
    (argumentsEvaluate : (keyRef :: trailingRefs).mapM (fun wire => lookupWire wire wires) =
      some (.bytes key :: trailingValues.map Value.integer))
    (matrixTypeEvaluate : matrixType.evaluate parameters (.constant 0) = some runtimeParams)
    (tagsEvaluate : tagExpressions.mapM (IntExpr.evaluate parameters) = some tagValues)
    (decimalTagsEvaluate :
      tagDecimalExpressions.mapM (IntExpr.evaluate parameters) = some tagDecimalValues)
    (u64TagsEvaluate :
      tagU64LeExpressions.mapM (IntExpr.evaluate parameters) = some tagU64LeValues)
    (baseEvaluate : evaluateOptionalIntExpr parameters (some baseExpr) = some (some base))
    (digitCountEvaluate : evaluateOptionalIntExpr parameters (some digitCountExpr) =
      some (some (Int.ofNat digitCount)))
    (typeHolds : matrixType.Holds parameters
      ((samplers.hashSample decomposed).withSamplerParams decomposed.params))
    (plainTypeHolds : plainMatrixType.Holds parameters
      ((samplers.hashSample plain).withSamplerParams plain.params))
    (modulusPositive : 0 < decomposed.params.modulus)
    (plainModulusPositive : 0 < plain.params.modulus)
    (layoutId : samplers.layoutId decomposed.params = some paramsId)
    (publicResult : samplers.gadgetPublicMatrix paramsId decomposed.params plain.params.rows
      base small digitCount = some publicMatrix)
    (statusSound : status = .available →
      (small = false ∨ ∃ limit,
        samplers.smallDecompositionInputLimit paramsId decomposed.params = some limit ∧
        Mxx.maxCanonicalCoefficient
          ((samplers.hashSample plain).withSamplerParams plain.params) < limit))
    (publicKeyHolds : (SemanticMatrixKey.gadget paramsId decomposed.params plain.params.rows
      base small digitCount).Holds samplers publicMatrix)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .hashSample matrixType (if small then .smallDecomposed else .decomposed)
        tagPrefix tagExpressions tagDecimalExpressions tagU64LeExpressions
        (some baseExpr) (some digitCountExpr)
      arguments := keyRef :: trailingRefs
      outputCount
    }) :
    values = [.matrix ((samplers.hashSample decomposed).withSamplerParams decomposed.params)] ∧
      OperationalFact.HoldsEvidence samplers parameters states address
        (decomposedHashOperationalFact address outputWire matrixType plainMatrixType paramsId
          plain decomposed raw base small digitCount status)
        (.matrix ((samplers.hashSample decomposed).withSamplerParams decomposed.params))
        (decomposedHashOperationalEvidence samplers address outputWire paramsId plain decomposed
          hashEvidence base small digitCount publicMatrix) := by
  have valuesExact :
      values = [.matrix ((samplers.hashSample decomposed).withSamplerParams decomposed.params)] := by
    have evaluated := mem_evaluateNode_hashSample_of_arguments runChild samplers parameters inputs
      wires keyRef trailingRefs key trailingValues matrixType
      (if small then .smallDecomposed else .decomposed) tagPrefix tagExpressions
      tagDecimalExpressions tagU64LeExpressions (some baseExpr) (some digitCountExpr)
      runtimeParams tagValues tagDecimalValues tagU64LeValues (some base)
      (some (Int.ofNat digitCount)) outputCount argumentsEvaluate matrixTypeEvaluate tagsEvaluate
      decimalTagsEvaluate u64TagsEvaluate baseEvaluate digitCountEvaluate member
    simpa [decomposedExact] using evaluated
  refine ⟨valuesExact, ?_⟩
  have factSound := decomposedHashFact_operational_sound samplers contract parameters states
    address outputWire matrixType plainMatrixType paramsId plain decomposed raw hashEvidence base
    small digitCount publicMatrix status queryExact queriesMatch typeHolds plainTypeHolds
    modulusPositive plainModulusPositive layoutId publicResult statusSound publicKeyHolds
  simpa [decomposedHashOperationalFact, decomposedHashOperationalEvidence] using factSound

theorem gadgetDecomposeNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (contract : MxxBoundedSamplerContract samplers)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (inputRef outputWire : WireRef)
    (matrixType : MatrixTypeExpr)
    (baseExpr digitCountExpr : IntExpr)
    (small : Bool)
    (paramsId : Mxx.SamplerParamsId)
    (params : Mxx.SamplerParams)
    (base : Int)
    (digitCount : Nat)
    (inputSummary : RelationTargetSummary)
    (inputEvidence : RelationTargetEvidence)
    (publicMatrix rawOutput : Mxx.Matrix)
    (status : ReconstructionStatus)
    (outputCount : Nat)
    (argumentsEvaluate : [inputRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix inputEvidence.matrix])
    (matrixTypeEvaluate : matrixType.evaluate parameters (.constant 0) = some params)
    (baseEvaluate : baseExpr.evaluate parameters = some base)
    (digitCountEvaluate : digitCountExpr.evaluate parameters = some (Int.ofNat digitCount))
    (baseValid : 1 < base)
    (digitCountPositive : 0 < digitCount)
    (layoutId : samplers.layoutId params = some paramsId)
    (decompositionResult : samplers.gadgetDecompose paramsId params base small digitCount
      inputEvidence.matrix = some rawOutput)
    (typeHolds : matrixType.Holds parameters (rawOutput.withSamplerParams params))
    (modulusPositive : 0 < params.modulus)
    (inputSummaryHolds : inputSummary.HoldsEvidence samplers parameters states inputEvidence)
    (publicResult : samplers.gadgetPublicMatrix paramsId params inputEvidence.matrix.rows base
      small digitCount = some publicMatrix)
    (statusSound : status = .available →
      (small = false ∨ ∃ limit,
        samplers.smallDecompositionInputLimit paramsId params = some limit ∧
        Mxx.maxCanonicalCoefficient inputEvidence.matrix < limit))
    (publicKeyHolds : (SemanticMatrixKey.gadget paramsId params inputEvidence.matrix.rows base
      small digitCount).Holds samplers publicMatrix)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .gadgetDecompose matrixType baseExpr small digitCountExpr
      arguments := [inputRef]
      outputCount
    }) :
    values = [.matrix (rawOutput.withSamplerParams params)] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
      subject := outputWire
      origin := .value address.template outputWire
      matrixType
      matrixParams := params
      totalHardBound := (.minimum
        (.closedInt (.constant (params.modulus / 2)))
        (.closedInt (.constant (Int.ofNat (Mxx.gadgetDecompositionBound base small)))))
      canonicalRange := if small then .below base.natAbs else .unknown
      relations := [.decomposition {
        producer := .value address.template outputWire
        publicIdentity := .gadget paramsId params inputEvidence.matrix.rows base small digitCount
        inputOrigin := inputSummary.origin
        inputSummary
        base
        small
        digitCount
        status
      }]
    }) (.matrix (rawOutput.withSamplerParams params)) (.matrix {
      originKey := .executionValue address outputWire (rawOutput.withSamplerParams params)
      relations := [.decomposition {
        producerKey := .executionValue address outputWire (rawOutput.withSamplerParams params)
        publicKey := .gadget paramsId params inputEvidence.matrix.rows base small digitCount
        input := inputEvidence
        publicMatrix
        decomposition := rawOutput.withSamplerParams params
      }]
    }) := by
  have valuesExact : values = [.matrix (rawOutput.withSamplerParams params)] := by
    have digitCountIntPositive : 0 < Int.ofNat digitCount := by
      cases digitCount with
      | zero => omega
      | succ count => simp
    simpa [evaluateNode, Mxx.Ir.arguments, argumentsEvaluate, matrixTypeEvaluate, baseEvaluate,
      digitCountEvaluate, show ¬base ≤ 1 by omega,
      show ¬(Int.ofNat digitCount) ≤ 0 from not_le.mpr digitCountIntPositive,
      show digitCount ≠ 0 by omega, layoutId, decompositionResult] using member
  refine ⟨valuesExact, ?_⟩
  exact gadgetDecompositionFact_operational_sound samplers contract parameters states address
    outputWire matrixType paramsId params base small digitCount inputSummary inputEvidence
    publicMatrix rawOutput status typeHolds modulusPositive inputSummaryHolds publicResult
    decompositionResult statusSound publicKeyHolds

theorem thresholdDecodeBoolNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (matrixRef : WireRef)
    (matrix : Mxx.Matrix)
    (ciphertextModulus plaintextModulus length : IntExpr)
    (evaluatedCiphertextModulus evaluatedPlaintextModulus evaluatedLength : Int)
    (outputCount : Nat)
    (argumentsEvaluate : [matrixRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix matrix])
    (ciphertextModulusEvaluate :
      ciphertextModulus.evaluate parameters = some evaluatedCiphertextModulus)
    (plaintextModulusEvaluate :
      plaintextModulus.evaluate parameters = some evaluatedPlaintextModulus)
    (lengthEvaluate : length.evaluate parameters = some evaluatedLength)
    (lengthNonnegative : 0 ≤ evaluatedLength)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .thresholdDecodeBool ciphertextModulus plaintextModulus length
      arguments := [matrixRef]
      outputCount
    }) :
    values = ((matrix.coefficients.take evaluatedLength.toNat).map fun coefficient =>
      .boolean (Mxx.Ir.thresholdDecodeBool evaluatedCiphertextModulus
        evaluatedPlaintextModulus coefficient)) ∧
      ∀ (port : Nat) (value : Bool),
        values[port]? = some (Value.boolean value) →
        OperationalFact.HoldsEvidence samplers parameters states address .boolean
          (.boolean value) (.boolean value) := by
  have valuesExact : values = ((matrix.coefficients.take evaluatedLength.toNat).map
      fun coefficient => .boolean (Mxx.Ir.thresholdDecodeBool evaluatedCiphertextModulus
        evaluatedPlaintextModulus coefficient)) := by
    simpa [evaluateNode, Mxx.Ir.arguments, argumentsEvaluate, ciphertextModulusEvaluate,
      plaintextModulusEvaluate, lengthEvaluate, if_neg (not_lt.mpr lengthNonnegative)] using member
  refine ⟨valuesExact, ?_⟩
  intro port value outputLookup
  exact booleanFact_holdsEvidence samplers parameters states address value

theorem matrixAddNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (leftRef rightRef outputWire : WireRef)
    (left right : Mxx.Matrix)
    (matrixType : MatrixTypeExpr)
    (layoutParams factParams : Mxx.SamplerParams)
    (leftMaximum rightMaximum : OperationalBoundExpr)
    (outputCount : Nat)
    (typeEvaluates : matrixType.evaluate parameters = some layoutParams)
    (leftTypeHolds : matrixType.Holds parameters left)
    (rightWellFormed : right.WellFormed)
    (sameModulus : left.modulus = right.modulus)
    (sameRingDimension : left.ringDimension = right.ringDimension)
    (sameRows : left.rows = right.rows)
    (sameColumns : left.columns = right.columns)
    (factModulus : left.modulus = factParams.modulus)
    (factRingDimension : left.ringDimension = factParams.ringDimension)
    (factRows : left.rows = factParams.rows)
    (factColumns : left.columns = factParams.columns)
    (modulusPositive : 0 < left.modulus)
    (leftMagnitude :
      OperationalBoundExpr.Holds parameters states (.finite leftMaximum) left)
    (rightMagnitude :
      OperationalBoundExpr.Holds parameters states (.finite rightMaximum) right)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix left, .matrix right])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .matrixAdd
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    ∃ output,
      values = [.matrix output] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := outputWire
        origin := .value address.template outputWire
        matrixType
        matrixParams := factParams
        totalHardBound := (.add
          (.add (.closedInt (.constant 0)) leftMaximum) rightMaximum)
      }) (.matrix output) (.matrix {
        originKey := .executionValue address outputWire output
    }) := by
  rcases leftTypeHolds with
    ⟨leftLayoutParams, leftTypeEvaluates, leftLayoutModulus, leftLayoutRingDimension,
      leftLayoutRows, leftLayoutColumns, leftWellFormed⟩
  have layoutParamsEq : leftLayoutParams = layoutParams := by
    exact Option.some.inj (leftTypeEvaluates.symm.trans typeEvaluates)
  subst leftLayoutParams
  rcases leftMagnitude with ⟨leftBound, leftEvaluates, leftNonnegative, leftNorm⟩
  rcases rightMagnitude with ⟨rightBound, rightEvaluates, rightNonnegative, rightNorm⟩
  let q := left.modulus.toNat
  have qPositive : 0 < q := by simpa [q] using modulusPositive
  letI : NeZero q := ⟨Nat.ne_of_gt qPositive⟩
  have leftModulus : left.modulus = q := by
    exact (Int.toNat_of_nonneg modulusPositive.le).symm
  have rightModulus : right.modulus = q := sameModulus.symm.trans leftModulus
  obtain ⟨valuesExact, outputNorm⟩ := matrixAddNode_local_sound runChild samplers parameters
    inputs wires leftRef rightRef left right outputCount q leftBound.toNat rightBound.toNat
    argumentsEvaluate leftModulus rightModulus leftNorm rightNorm member
  let output := Mxx.matrixAdd left right
  refine ⟨output, valuesExact, ?_⟩
  apply localMatrixFact_holdsEvidence
  · refine ⟨layoutParams, typeEvaluates, ?_, ?_, ?_, ?_, ?_⟩
    · simpa [output, Mxx.matrixAdd] using leftLayoutModulus
    · simpa [output, Mxx.matrixAdd] using leftLayoutRingDimension
    · simpa [output, Mxx.matrixAdd] using leftLayoutRows
    · simpa [output, Mxx.matrixAdd] using leftLayoutColumns
    · exact Mxx.matrixAdd_wellFormed left right leftWellFormed rightWellFormed
        sameRows sameColumns sameRingDimension
  · simpa [output, Mxx.matrixAdd] using factModulus
  · simpa [output, Mxx.matrixAdd] using factRingDimension
  · simpa [output, Mxx.matrixAdd] using factRows
  · simpa [output, Mxx.matrixAdd] using factColumns
  · refine ⟨leftBound + rightBound, ?_, by omega, ?_⟩
    · simp only [OperationalBoundExpr.evaluateWithStates,
        OperationalBoundExpr.evaluateWithStates_closedConstant]
      rw [leftEvaluates, rightEvaluates]
      change (pure (0 + leftBound + rightBound) : Except OperationalError Int) =
        pure (leftBound + rightBound)
      congr 2
      omega
    · rw [Int.toNat_add leftNonnegative rightNonnegative]
      exact outputNorm
  · trivial

theorem matrixSubtractNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (leftRef rightRef outputWire : WireRef)
    (left right : Mxx.Matrix)
    (matrixType : MatrixTypeExpr)
    (layoutParams factParams : Mxx.SamplerParams)
    (leftMaximum rightMaximum : OperationalBoundExpr)
    (outputCount : Nat)
    (typeEvaluates : matrixType.evaluate parameters = some layoutParams)
    (leftTypeHolds : matrixType.Holds parameters left)
    (rightWellFormed : right.WellFormed)
    (sameModulus : left.modulus = right.modulus)
    (sameRingDimension : left.ringDimension = right.ringDimension)
    (sameRows : left.rows = right.rows)
    (sameColumns : left.columns = right.columns)
    (factModulus : left.modulus = factParams.modulus)
    (factRingDimension : left.ringDimension = factParams.ringDimension)
    (factRows : left.rows = factParams.rows)
    (factColumns : left.columns = factParams.columns)
    (modulusPositive : 0 < left.modulus)
    (leftMagnitude :
      OperationalBoundExpr.Holds parameters states (.finite leftMaximum) left)
    (rightMagnitude :
      OperationalBoundExpr.Holds parameters states (.finite rightMaximum) right)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix left, .matrix right])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .matrixSubtract
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    ∃ output,
      values = [.matrix output] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := outputWire
        origin := .value address.template outputWire
        matrixType
        matrixParams := factParams
        totalHardBound := (.add
          (.add (.closedInt (.constant 0)) leftMaximum) rightMaximum)
      }) (.matrix output) (.matrix {
        originKey := .executionValue address outputWire output
      }) := by
  rcases leftTypeHolds with
    ⟨leftLayoutParams, leftTypeEvaluates, leftLayoutModulus, leftLayoutRingDimension,
      leftLayoutRows, leftLayoutColumns, leftWellFormed⟩
  have layoutParamsEq : leftLayoutParams = layoutParams := by
    exact Option.some.inj (leftTypeEvaluates.symm.trans typeEvaluates)
  subst leftLayoutParams
  rcases leftMagnitude with ⟨leftBound, leftEvaluates, leftNonnegative, leftNorm⟩
  rcases rightMagnitude with ⟨rightBound, rightEvaluates, rightNonnegative, rightNorm⟩
  let q := left.modulus.toNat
  have qPositive : 0 < q := by simpa [q] using modulusPositive
  letI : NeZero q := ⟨Nat.ne_of_gt qPositive⟩
  have leftModulus : left.modulus = q := by
    exact (Int.toNat_of_nonneg modulusPositive.le).symm
  have rightModulus : right.modulus = q := sameModulus.symm.trans leftModulus
  obtain ⟨valuesExact, outputNorm⟩ := matrixSubtractNode_local_sound runChild samplers parameters
    inputs wires leftRef rightRef left right outputCount q leftBound.toNat rightBound.toNat
    argumentsEvaluate leftModulus rightModulus leftNorm rightNorm member
  let output := Mxx.matrixSubtract left right
  refine ⟨output, valuesExact, ?_⟩
  apply localMatrixFact_holdsEvidence
  · refine ⟨layoutParams, typeEvaluates, ?_, ?_, ?_, ?_, ?_⟩
    · simpa [output, Mxx.matrixSubtract] using leftLayoutModulus
    · simpa [output, Mxx.matrixSubtract] using leftLayoutRingDimension
    · simpa [output, Mxx.matrixSubtract] using leftLayoutRows
    · simpa [output, Mxx.matrixSubtract] using leftLayoutColumns
    · exact Mxx.matrixSubtract_wellFormed left right leftWellFormed rightWellFormed
        sameRows sameColumns sameRingDimension
  · simpa [output, Mxx.matrixSubtract] using factModulus
  · simpa [output, Mxx.matrixSubtract] using factRingDimension
  · simpa [output, Mxx.matrixSubtract] using factRows
  · simpa [output, Mxx.matrixSubtract] using factColumns
  · refine ⟨leftBound + rightBound, ?_, by omega, ?_⟩
    · simp only [OperationalBoundExpr.evaluateWithStates,
        OperationalBoundExpr.evaluateWithStates_closedConstant]
      rw [leftEvaluates, rightEvaluates]
      change (pure (0 + leftBound + rightBound) : Except OperationalError Int) =
        pure (leftBound + rightBound)
      congr 2
      omega
    · rw [Int.toNat_add leftNonnegative rightNonnegative]
      exact outputNorm
  · trivial

theorem matrixNegateNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (inputRef outputWire : WireRef)
    (input : Mxx.Matrix)
    (matrixType : MatrixTypeExpr)
    (factParams : Mxx.SamplerParams)
    (magnitude : OperationalBoundExpr)
    (outputCount : Nat)
    (inputTypeHolds : matrixType.Holds parameters input)
    (factModulus : input.modulus = factParams.modulus)
    (factRingDimension : input.ringDimension = factParams.ringDimension)
    (factRows : input.rows = factParams.rows)
    (factColumns : input.columns = factParams.columns)
    (modulusPositive : 0 < input.modulus)
    (inputMagnitude : magnitude.Holds parameters states input)
    (argumentsEvaluate : [inputRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix input])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .matrixNegate
      arguments := [inputRef]
      outputCount
    }) :
    let output := Mxx.matrixNegate input
    values = [.matrix output] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := outputWire
        origin := .value address.template outputWire
        matrixType
        matrixParams := factParams
        magnitude
      }) (.matrix output) (.matrix {
        originKey := .executionValue address outputWire output
      }) := by
  dsimp
  rcases inputTypeHolds with
    ⟨layoutParams, typeEvaluates, layoutModulus, layoutRingDimension, layoutRows,
      layoutColumns, inputWellFormed⟩
  rcases inputMagnitude with ⟨bound, boundEvaluates, boundNonnegative, inputNorm⟩
  let q := input.modulus.toNat
  have qPositive : 0 < q := by simpa [q] using modulusPositive
  letI : NeZero q := ⟨Nat.ne_of_gt qPositive⟩
  have inputModulus : input.modulus = q :=
    (Int.toNat_of_nonneg modulusPositive.le).symm
  obtain ⟨valuesExact, outputNorm⟩ := matrixNegateNode_local_sound runChild samplers
    parameters inputs wires inputRef input outputCount q bound.toNat argumentsEvaluate
    inputModulus inputNorm member
  refine ⟨valuesExact, ?_⟩
  apply localMatrixFact_holdsEvidence
  · refine ⟨layoutParams, typeEvaluates, ?_, ?_, ?_, ?_, ?_⟩
    · simpa [Mxx.matrixNegate] using layoutModulus
    · simpa [Mxx.matrixNegate] using layoutRingDimension
    · simpa [Mxx.matrixNegate] using layoutRows
    · simpa [Mxx.matrixNegate] using layoutColumns
    · exact Mxx.matrixNegate_wellFormed input inputWellFormed
  · simpa [Mxx.matrixNegate] using factModulus
  · simpa [Mxx.matrixNegate] using factRingDimension
  · simpa [Mxx.matrixNegate] using factRows
  · simpa [Mxx.matrixNegate] using factColumns
  · exact ⟨bound, boundEvaluates, boundNonnegative, outputNorm⟩
  · trivial

theorem matrixScaleNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (inputRef outputWire : WireRef)
    (input : Mxx.Matrix)
    (scalarExpression : IntExpr)
    (scalar : Int)
    (matrixType : MatrixTypeExpr)
    (factParams : Mxx.SamplerParams)
    (inputMagnitude : OperationalBoundExpr)
    (scalarMaximum : OperationalBoundExpr)
    (outputNotSmall : Bool)
    (outputCount : Nat)
    (inputTypeHolds : matrixType.Holds parameters input)
    (factModulus : input.modulus = factParams.modulus)
    (factRingDimension : input.ringDimension = factParams.ringDimension)
    (factRows : input.rows = factParams.rows)
    (factColumns : input.columns = factParams.columns)
    (modulusPositive : 0 < input.modulus)
    (inputMagnitudeHolds : inputMagnitude.Holds parameters states input)
    (scalarMaximumValue : Int)
    (scalarMaximumEvaluates :
      scalarMaximum.evaluateWithStates parameters states = .ok scalarMaximumValue)
    (scalarMaximumNonnegative : 0 ≤ scalarMaximumValue)
    (scalarNatBound : scalar.natAbs ≤ scalarMaximumValue.toNat)
    (scalarEvaluate : scalarExpression.evaluate parameters = some scalar)
    (argumentsEvaluate : [inputRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix input])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .matrixScale scalarExpression
      arguments := [inputRef]
      outputCount
    }) :
    let output := Mxx.matrixScale scalar input
    let maximum := .multiply scalarMaximum inputMagnitude.boundExpression
    let outputMagnitude := if outputNotSmall then .notSmall maximum else .finite maximum
    values = [.matrix output] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := outputWire
        origin := .value address.template outputWire
        matrixType
        matrixParams := factParams
        magnitude := outputMagnitude
      }) (.matrix output) (.matrix {
        originKey := .executionValue address outputWire output
      }) := by
  dsimp
  rcases inputTypeHolds with
    ⟨layoutParams, typeEvaluates, layoutModulus, layoutRingDimension, layoutRows,
      layoutColumns, inputWellFormed⟩
  rcases inputMagnitudeHolds with
    ⟨inputBound, inputBoundEvaluates, inputBoundNonnegative, inputNorm⟩
  have valuesExact := mem_evaluateNode_matrixScale_of_arguments runChild samplers parameters
    inputs wires inputRef input scalarExpression scalar outputCount argumentsEvaluate scalarEvaluate member
  let output := Mxx.matrixScale scalar input
  let maximum := OperationalBoundExpr.multiply scalarMaximum inputMagnitude.boundExpression
  let outputMagnitude := if outputNotSmall then
    OperationalBoundExpr.notSmall maximum else .finite maximum
  have outputNorm : Mxx.maxCenteredCoefficientNorm output ≤
      (scalarMaximumValue * inputBound).toNat := by
    let q := input.modulus.toNat
    have qPositive : 0 < q := by simpa [q] using modulusPositive
    letI : NeZero q := ⟨Nat.ne_of_gt qPositive⟩
    have inputModulus : input.modulus = q :=
      (Int.toNat_of_nonneg modulusPositive.le).symm
    have scaled := Mxx.Toolkit.matrixScale_norm_le q scalar input inputModulus
    rw [Int.toNat_mul scalarMaximumNonnegative inputBoundNonnegative]
    exact le_trans scaled (Nat.mul_le_mul scalarNatBound inputNorm)
  have outputMagnitudeHolds : outputMagnitude.Holds parameters states output := by
    refine ⟨scalarMaximumValue * inputBound, ?_,
      mul_nonneg scalarMaximumNonnegative inputBoundNonnegative, outputNorm⟩
    cases outputNotSmall <;>
      simp [outputMagnitude, maximum, OperationalBoundExpr.boundExpression,
        OperationalBoundExpr.evaluateWithStates, scalarMaximumEvaluates, inputBoundEvaluates] <;>
      rfl
  refine ⟨valuesExact, ?_⟩
  apply localMatrixFact_holdsEvidence
  · refine ⟨layoutParams, typeEvaluates, ?_, ?_, ?_, ?_, ?_⟩
    · simpa [output, Mxx.matrixScale] using layoutModulus
    · simpa [output, Mxx.matrixScale] using layoutRingDimension
    · simpa [output, Mxx.matrixScale] using layoutRows
    · simpa [output, Mxx.matrixScale] using layoutColumns
    · exact Mxx.matrixScale_wellFormed scalar input inputWellFormed
  · simpa [output, Mxx.matrixScale] using factModulus
  · simpa [output, Mxx.matrixScale] using factRingDimension
  · simpa [output, Mxx.matrixScale] using factRows
  · simpa [output, Mxx.matrixScale] using factColumns
  · exact outputMagnitudeHolds
  · trivial

theorem matrixProductFact_operational_sound
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (outputWire : WireRef)
    (output : Mxx.Matrix)
    (matrixType : MatrixTypeExpr)
    (factParams : Mxx.SamplerParams)
    (ringDimension innerDimension : Nat)
    (leftMaximum rightMaximum : OperationalBoundExpr)
    (outputNotSmall : Bool)
    (leftBound rightBound : Int)
    (leftEvaluates : leftMaximum.evaluateWithStates parameters states = .ok leftBound)
    (rightEvaluates : rightMaximum.evaluateWithStates parameters states = .ok rightBound)
    (leftNonnegative : 0 ≤ leftBound)
    (rightNonnegative : 0 ≤ rightBound)
    (typeHolds : matrixType.Holds parameters output)
    (factModulus : output.modulus = factParams.modulus)
    (factRingDimension : output.ringDimension = factParams.ringDimension)
    (factRows : output.rows = factParams.rows)
    (factColumns : output.columns = factParams.columns)
    (outputNorm : Mxx.maxCenteredCoefficientNorm output ≤
      ringDimension * innerDimension * leftBound.toNat * rightBound.toNat) :
    let maximum := .matrixProduct
      (.closedInt (.constant (Int.ofNat ringDimension)))
      (.closedInt (.constant (Int.ofNat innerDimension))) leftMaximum rightMaximum
    let magnitude := if outputNotSmall then .notSmall maximum else .finite maximum
    OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
      subject := outputWire
      origin := .value address.template outputWire
      matrixType
      matrixParams := factParams
      magnitude
    }) (.matrix output) (.matrix {
      originKey := .executionValue address outputWire output
    }) := by
  dsimp
  let maximum : OperationalBoundExpr := .matrixProduct
    (.closedInt (.constant (Int.ofNat ringDimension)))
    (.closedInt (.constant (Int.ofNat innerDimension))) leftMaximum rightMaximum
  let magnitude := if outputNotSmall then OperationalBoundExpr.notSmall maximum else
    .finite maximum
  have magnitudeHolds : magnitude.Holds parameters states output := by
    let bound := Int.ofNat ringDimension * Int.ofNat innerDimension * leftBound * rightBound
    have boundNonnegative : 0 ≤ bound := by
      exact mul_nonneg (mul_nonneg
        (mul_nonneg (Int.natCast_nonneg ringDimension) (Int.natCast_nonneg innerDimension))
        leftNonnegative) rightNonnegative
    refine ⟨bound, ?_, boundNonnegative, ?_⟩
    · cases outputNotSmall <;>
        simp [magnitude, maximum, bound, OperationalBoundExpr.evaluateWithStates,
          leftEvaluates, rightEvaluates] <;>
        rfl
    · have boundToNat : bound.toNat =
          ringDimension * innerDimension * leftBound.toNat * rightBound.toNat := by
        dsimp [bound]
        rw [Int.toNat_mul (mul_nonneg
          (mul_nonneg (Int.natCast_nonneg ringDimension) (Int.natCast_nonneg innerDimension))
          leftNonnegative) rightNonnegative]
        rw [Int.toNat_mul
          (mul_nonneg (Int.natCast_nonneg ringDimension) (Int.natCast_nonneg innerDimension))
          leftNonnegative]
        rw [Int.toNat_mul (Int.natCast_nonneg ringDimension)
          (Int.natCast_nonneg innerDimension)]
        simp
      rw [boundToNat]
      exact outputNorm
  exact ⟨typeHolds, factModulus, factRingDimension, factRows, factColumns,
    ⟨rfl, rfl, Mxx.MatrixModEq.refl output⟩, magnitudeHolds, trivial, trivial,
    List.Forall₂.nil⟩

theorem matrixMultiplyOrdinaryNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (leftRef rightRef outputWire : WireRef)
    (left right : Mxx.Matrix)
    (matrixType : MatrixTypeExpr)
    (factParams : Mxx.SamplerParams)
    (q ringDimension innerDimension : Nat)
    [NeZero q]
    (leftMaximum rightMaximum : OperationalBoundExpr)
    (outputNotSmall : Bool)
    (leftBound rightBound : Int)
    (leftEvaluates : leftMaximum.evaluateWithStates parameters states = .ok leftBound)
    (rightEvaluates : rightMaximum.evaluateWithStates parameters states = .ok rightBound)
    (leftNonnegative : 0 ≤ leftBound)
    (rightNonnegative : 0 ≤ rightBound)
    (runtimeBranch : Mxx.matrixMultiply left right = Mxx.matrixMul left right)
    (leftModulus : left.modulus = q)
    (rightModulus : right.modulus = q)
    (leftRing : left.ringDimension = ringDimension)
    (rightRing : right.ringDimension = ringDimension)
    (leftColumns : left.columns = innerDimension)
    (rightRows : right.rows = innerDimension)
    (leftNorm : Mxx.maxCenteredCoefficientNorm left ≤ leftBound.toNat)
    (rightNorm : Mxx.maxCenteredCoefficientNorm right ≤ rightBound.toNat)
    (outputTypeHolds : matrixType.Holds parameters (Mxx.matrixMultiply left right))
    (factModulus : (Mxx.matrixMultiply left right).modulus = factParams.modulus)
    (factRingDimension :
      (Mxx.matrixMultiply left right).ringDimension = factParams.ringDimension)
    (factRows : (Mxx.matrixMultiply left right).rows = factParams.rows)
    (factColumns : (Mxx.matrixMultiply left right).columns = factParams.columns)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix left, .matrix right])
    (outputCount : Nat)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .matrixMultiply
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    let output := Mxx.matrixMultiply left right
    let maximum := .matrixProduct
      (.closedInt (.constant (Int.ofNat ringDimension)))
      (.closedInt (.constant (Int.ofNat innerDimension))) leftMaximum rightMaximum
    let magnitude := if outputNotSmall then .notSmall maximum else .finite maximum
    values = [.matrix output] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := outputWire
        origin := .value address.template outputWire
        matrixType
        matrixParams := factParams
        magnitude
      }) (.matrix output) (.matrix {
        originKey := .executionValue address outputWire output
      }) := by
  dsimp
  have valuesExact := mem_evaluateNode_matrixMultiply_of_arguments runChild samplers parameters
    inputs wires leftRef rightRef left right outputCount argumentsEvaluate member
  have outputNorm : Mxx.maxCenteredCoefficientNorm (Mxx.matrixMultiply left right) ≤
      ringDimension * innerDimension * leftBound.toNat * rightBound.toNat := by
    rw [runtimeBranch]
    exact Mxx.Toolkit.matrixMul_norm_le q ringDimension innerDimension leftBound.toNat
      rightBound.toNat left right leftModulus rightModulus leftRing rightRing leftColumns rightRows
      leftNorm rightNorm
  refine ⟨valuesExact, ?_⟩
  exact matrixProductFact_operational_sound samplers parameters states address outputWire
    (Mxx.matrixMultiply left right) matrixType factParams ringDimension innerDimension leftMaximum
    rightMaximum outputNotSmall leftBound rightBound leftEvaluates rightEvaluates leftNonnegative
    rightNonnegative outputTypeHolds factModulus factRingDimension factRows factColumns outputNorm

theorem matrixMultiplySwappedNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (leftRef rightRef outputWire : WireRef)
    (left right : Mxx.Matrix)
    (matrixType : MatrixTypeExpr)
    (factParams : Mxx.SamplerParams)
    (q ringDimension : Nat)
    [NeZero q]
    (leftMaximum rightMaximum : OperationalBoundExpr)
    (outputNotSmall : Bool)
    (leftBound rightBound : Int)
    (leftEvaluates : leftMaximum.evaluateWithStates parameters states = .ok leftBound)
    (rightEvaluates : rightMaximum.evaluateWithStates parameters states = .ok rightBound)
    (leftNonnegative : 0 ≤ leftBound)
    (rightNonnegative : 0 ≤ rightBound)
    (runtimeBranch : Mxx.matrixMultiply left right = Mxx.matrixMul right left)
    (leftModulus : left.modulus = q)
    (rightModulus : right.modulus = q)
    (leftRing : left.ringDimension = ringDimension)
    (rightRing : right.ringDimension = ringDimension)
    (rightColumns : right.columns = 1)
    (leftRows : left.rows = 1)
    (leftNorm : Mxx.maxCenteredCoefficientNorm left ≤ leftBound.toNat)
    (rightNorm : Mxx.maxCenteredCoefficientNorm right ≤ rightBound.toNat)
    (outputTypeHolds : matrixType.Holds parameters (Mxx.matrixMultiply left right))
    (factModulus : (Mxx.matrixMultiply left right).modulus = factParams.modulus)
    (factRingDimension :
      (Mxx.matrixMultiply left right).ringDimension = factParams.ringDimension)
    (factRows : (Mxx.matrixMultiply left right).rows = factParams.rows)
    (factColumns : (Mxx.matrixMultiply left right).columns = factParams.columns)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix left, .matrix right])
    (outputCount : Nat)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .matrixMultiply
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    let output := Mxx.matrixMultiply left right
    let maximum := .matrixProduct
      (.closedInt (.constant (Int.ofNat ringDimension)))
      (.closedInt (.constant 1)) leftMaximum rightMaximum
    let magnitude := if outputNotSmall then .notSmall maximum else .finite maximum
    values = [.matrix output] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := outputWire
        origin := .value address.template outputWire
        matrixType
        matrixParams := factParams
        magnitude
      }) (.matrix output) (.matrix {
        originKey := .executionValue address outputWire output
      }) := by
  dsimp
  have valuesExact := mem_evaluateNode_matrixMultiply_of_arguments runChild samplers parameters
    inputs wires leftRef rightRef left right outputCount argumentsEvaluate member
  have outputNorm : Mxx.maxCenteredCoefficientNorm (Mxx.matrixMultiply left right) ≤
      ringDimension * 1 * leftBound.toNat * rightBound.toNat := by
    rw [runtimeBranch]
    rw [show ringDimension * 1 * leftBound.toNat * rightBound.toNat =
      ringDimension * 1 * rightBound.toNat * leftBound.toNat by ac_rfl]
    exact Mxx.Toolkit.matrixMul_norm_le q ringDimension 1 rightBound.toNat leftBound.toNat
      right left rightModulus leftModulus rightRing leftRing rightColumns leftRows rightNorm leftNorm
  refine ⟨valuesExact, ?_⟩
  exact matrixProductFact_operational_sound samplers parameters states address outputWire
    (Mxx.matrixMultiply left right) matrixType factParams ringDimension 1 leftMaximum rightMaximum
    outputNotSmall leftBound rightBound leftEvaluates rightEvaluates leftNonnegative rightNonnegative
    outputTypeHolds factModulus factRingDimension factRows factColumns outputNorm

theorem matrixPolynomialScale_operational_norm_le
    (q ringDimension scalarBound matrixBound : Nat)
    [NeZero q]
    (scalar matrix : Mxx.Matrix)
    (scalarModulus : scalar.modulus = q)
    (matrixModulus : matrix.modulus = q)
    (matrixRing : matrix.ringDimension = ringDimension)
    (scalarNorm : Mxx.maxCenteredCoefficientNorm scalar ≤ scalarBound)
    (matrixNorm : Mxx.maxCenteredCoefficientNorm matrix ≤ matrixBound) :
    Mxx.maxCenteredCoefficientNorm (Mxx.matrixPolynomialScale scalar matrix) ≤
      ringDimension * scalarBound * matrixBound := by
  unfold Mxx.maxCenteredCoefficientNorm
  apply Mxx.Toolkit.coefficientNorm_le
  intro outputCoefficient outputMember
  simp only [List.mem_map] at outputMember
  obtain ⟨reduced, reducedMember, rfl⟩ := outputMember
  obtain ⟨linear, rfl⟩ := List.mem_ofFn.mp reducedMember
  let coefficient := linear.val % matrix.ringDimension
  let entry := linear.val / matrix.ringDimension
  let column := entry % matrix.columns
  let row := entry / matrix.columns
  have ringPositive : 0 < matrix.ringDimension := by
    by_contra nonpositive
    have zero : matrix.ringDimension = 0 := Nat.eq_zero_of_not_pos nonpositive
    have linearLt := linear.isLt
    simp [zero] at linearLt
  have coefficientLt : coefficient < matrix.ringDimension :=
    Nat.mod_lt _ ringPositive
  change
    (Mxx.centeredCoefficient matrix.modulus
      (Mxx.reduceCoefficient matrix.modulus
        (Mxx.negacyclicCoefficient matrix.ringDimension
          (scalar.coefficient 0 0) (matrix.coefficient row column) coefficient))).natAbs ≤
      ringDimension * scalarBound * matrixBound
  rw [matrixModulus, Mxx.Toolkit.centeredCoefficient_reduce]
  rw [matrixRing]
  apply le_trans
    (Mxx.Toolkit.negacyclicCoefficient_natAbs_le q ringDimension
      (scalar.coefficient 0 0) (matrix.coefficient row column) coefficient
      scalarBound matrixBound (by simpa [matrixRing] using coefficientLt) _ _)
  · rfl
  · intro index indexLt
    exact le_trans
      (by simpa [scalarModulus] using
        Mxx.Toolkit.centeredEntry_natAbs_le_norm scalar 0 0 index)
      scalarNorm
  · intro index indexLt
    exact le_trans
      (by simpa [matrixModulus, matrixRing] using
        Mxx.Toolkit.centeredEntry_natAbs_le_norm matrix row column index)
      matrixNorm

theorem matrixMultiplyBroadcastNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (leftRef rightRef outputWire : WireRef)
    (left right scalar matrix : Mxx.Matrix)
    (matrixType : MatrixTypeExpr)
    (factParams : Mxx.SamplerParams)
    (q ringDimension : Nat)
    [NeZero q]
    (leftMaximum rightMaximum : OperationalBoundExpr)
    (outputNotSmall : Bool)
    (leftBound rightBound : Int)
    (leftEvaluates : leftMaximum.evaluateWithStates parameters states = .ok leftBound)
    (rightEvaluates : rightMaximum.evaluateWithStates parameters states = .ok rightBound)
    (leftNonnegative : 0 ≤ leftBound)
    (rightNonnegative : 0 ≤ rightBound)
    (runtimeBranch : Mxx.matrixMultiply left right = Mxx.matrixPolynomialScale scalar matrix)
    (matrixIsOtherOperand :
      (scalar = left ∧ matrix = right) ∨ (scalar = right ∧ matrix = left))
    (scalarModulus : scalar.modulus = q)
    (matrixModulus : matrix.modulus = q)
    (matrixRing : matrix.ringDimension = ringDimension)
    (leftNorm : Mxx.maxCenteredCoefficientNorm left ≤ leftBound.toNat)
    (rightNorm : Mxx.maxCenteredCoefficientNorm right ≤ rightBound.toNat)
    (outputTypeHolds : matrixType.Holds parameters (Mxx.matrixMultiply left right))
    (factModulus : (Mxx.matrixMultiply left right).modulus = factParams.modulus)
    (factRingDimension :
      (Mxx.matrixMultiply left right).ringDimension = factParams.ringDimension)
    (factRows : (Mxx.matrixMultiply left right).rows = factParams.rows)
    (factColumns : (Mxx.matrixMultiply left right).columns = factParams.columns)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix left, .matrix right])
    (outputCount : Nat)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .matrixMultiply
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    let output := Mxx.matrixMultiply left right
    let maximum := .matrixProduct
      (.closedInt (.constant (Int.ofNat ringDimension)))
      (.closedInt (.constant 1)) leftMaximum rightMaximum
    let magnitude := if outputNotSmall then .notSmall maximum else .finite maximum
    values = [.matrix output] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := outputWire
        origin := .value address.template outputWire
        matrixType
        matrixParams := factParams
        magnitude
      }) (.matrix output) (.matrix {
        originKey := .executionValue address outputWire output
      }) := by
  dsimp
  have valuesExact := mem_evaluateNode_matrixMultiply_of_arguments runChild samplers parameters
    inputs wires leftRef rightRef left right outputCount argumentsEvaluate member
  have outputNorm : Mxx.maxCenteredCoefficientNorm (Mxx.matrixMultiply left right) ≤
      ringDimension * 1 * leftBound.toNat * rightBound.toNat := by
    rw [runtimeBranch]
    rcases matrixIsOtherOperand with ⟨scalarIsLeft, matrixIsRight⟩ |
      ⟨scalarIsRight, matrixIsLeft⟩
    · subst scalar
      subst matrix
      simpa [Nat.mul_assoc] using
        matrixPolynomialScale_operational_norm_le q ringDimension leftBound.toNat
          rightBound.toNat left right scalarModulus matrixModulus matrixRing leftNorm rightNorm
    · subst scalar
      subst matrix
      rw [show ringDimension * 1 * leftBound.toNat * rightBound.toNat =
        ringDimension * rightBound.toNat * leftBound.toNat by ac_rfl]
      exact matrixPolynomialScale_operational_norm_le q ringDimension rightBound.toNat
        leftBound.toNat right left scalarModulus matrixModulus matrixRing rightNorm leftNorm
  refine ⟨valuesExact, ?_⟩
  exact matrixProductFact_operational_sound samplers parameters states address outputWire
    (Mxx.matrixMultiply left right) matrixType factParams ringDimension 1 leftMaximum rightMaximum
    outputNotSmall leftBound rightBound leftEvaluates rightEvaluates leftNonnegative rightNonnegative
    outputTypeHolds factModulus factRingDimension factRows factColumns outputNorm

theorem matrixAddNode_operational_sound_capped
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (leftRef rightRef outputWire : WireRef)
    (left right : Mxx.Matrix)
    (matrixType : MatrixTypeExpr)
    (layoutParams factParams : Mxx.SamplerParams)
    (leftMaximum rightMaximum : OperationalBoundExpr)
    (outputCount : Nat)
    (typeEvaluates : matrixType.evaluate parameters = some layoutParams)
    (leftTypeHolds : matrixType.Holds parameters left)
    (rightWellFormed : right.WellFormed)
    (sameModulus : left.modulus = right.modulus)
    (sameRingDimension : left.ringDimension = right.ringDimension)
    (sameRows : left.rows = right.rows)
    (sameColumns : left.columns = right.columns)
    (factModulus : left.modulus = factParams.modulus)
    (factRingDimension : left.ringDimension = factParams.ringDimension)
    (factRows : left.rows = factParams.rows)
    (factColumns : left.columns = factParams.columns)
    (modulusPositive : 0 < left.modulus)
    (leftMagnitude :
      OperationalBoundExpr.Holds parameters states (.finite leftMaximum) left)
    (rightMagnitude :
      OperationalBoundExpr.Holds parameters states (.finite rightMaximum) right)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix left, .matrix right])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .matrixAdd
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    ∃ output,
      values = [.matrix output] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := outputWire
        origin := .value address.template outputWire
        matrixType
        matrixParams := factParams
        totalHardBound := (.minimum
          (.closedInt (.constant (left.modulus / 2)))
          (.add (.add (.closedInt (.constant 0)) leftMaximum) rightMaximum))
      }) (.matrix output) (.matrix {
        originKey := .executionValue address outputWire output
      }) := by
  have leftMagnitudeCopy := leftMagnitude
  have rightMagnitudeCopy := rightMagnitude
  obtain ⟨leftBound, leftEvaluates, _, _⟩ := leftMagnitude
  obtain ⟨rightBound, rightEvaluates, _, _⟩ := rightMagnitude
  obtain ⟨output, valuesExact, holds⟩ := matrixAddNode_operational_sound runChild samplers
    parameters inputs wires states address leftRef rightRef outputWire left right matrixType
    layoutParams factParams leftMaximum rightMaximum outputCount typeEvaluates leftTypeHolds
    rightWellFormed sameModulus sameRingDimension sameRows sameColumns factModulus
    factRingDimension factRows factColumns modulusPositive leftMagnitudeCopy rightMagnitudeCopy
    argumentsEvaluate member
  let maximum : OperationalBoundExpr :=
    .add (.add (.closedInt (.constant 0)) leftMaximum) rightMaximum
  have maximumEvaluates : maximum.evaluateWithStates parameters states =
      .ok (leftBound + rightBound) := by
    simp only [maximum, OperationalBoundExpr.evaluateWithStates,
      OperationalBoundExpr.evaluateWithStates_closedConstant]
    rw [leftEvaluates, rightEvaluates]
    change Except.ok (0 + leftBound + rightBound) = Except.ok (leftBound + rightBound)
    congr 2
    omega
  have outputModulus : output.modulus = left.modulus := by
    exact holds.2.1.trans factModulus.symm
  have capped := holds.capFiniteMatrix rfl maximumEvaluates (by
    rw [outputModulus]
    exact modulusPositive)
  refine ⟨output, valuesExact, ?_⟩
  rw [outputModulus] at capped
  simpa [maximum] using capped

theorem matrixMultiplyRelationNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (leftRef rightRef outputWire : WireRef)
    (q ringDimension rows inner columns : Nat)
    [Fact (1 < q)] [NeZero q] [NeZero ringDimension]
    (leftKey rightKey : SemanticMatrixKey)
    (left right publicMatrix subject : Mxx.Matrix)
    (matrixType : MatrixTypeExpr)
    (targetSummary : RelationTargetSummary)
    (targetEvidence : RelationTargetEvidence)
    (leftLayout : Mxx.Toolkit.MatrixLayout left q ringDimension rows inner)
    (publicLayout : Mxx.Toolkit.MatrixLayout publicMatrix q ringDimension rows inner)
    (rightLayout : Mxx.Toolkit.MatrixLayout right q ringDimension inner columns)
    (subjectLayout : Mxx.Toolkit.MatrixLayout subject q ringDimension inner columns)
    (targetLayout : Mxx.Toolkit.MatrixLayout targetEvidence.matrix q ringDimension rows columns)
    (outputTypeHolds : matrixType.Holds parameters (Mxx.matrixMultiply left right))
    (leftHolds : leftKey.Holds samplers left)
    (publicHolds : leftKey.Holds samplers publicMatrix)
    (rightHolds : rightKey.Holds samplers right)
    (subjectHolds : rightKey.Holds samplers subject)
    (targetHolds : targetSummary.HoldsEvidence samplers parameters states targetEvidence)
    (storedRelation : Mxx.MatrixModEq (Mxx.matrixMul publicMatrix subject)
      targetEvidence.matrix)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix left, .matrix right])
    (outputCount : Nat)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .matrixMultiply
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixMultiply left right)] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := outputWire
        origin := targetSummary.origin
        matrixType
        matrixParams := targetSummary.matrixParams
        totalHardBound := targetSummary.totalHardBound
        canonicalRange := targetSummary.canonicalRange
      }) (.matrix (Mxx.matrixMultiply left right)) (.matrix {
        originKey := targetEvidence.key
        hashOrigin := targetEvidence.hashOrigin
      }) := by
  refine ⟨mem_evaluateNode_matrixMultiply_of_arguments runChild samplers parameters inputs wires
    leftRef rightRef left right outputCount argumentsEvaluate member, ?_⟩
  exact relationProductFact_operational_sound samplers parameters states address outputWire
    matrixType q ringDimension rows inner columns leftKey rightKey left right publicMatrix subject
    targetSummary targetEvidence leftLayout publicLayout rightLayout subjectLayout targetLayout
    outputTypeHolds leftHolds publicHolds rightHolds subjectHolds targetHolds storedRelation

theorem matrixSelectNode_operational_sound
    (runChild : ChildRunner)
    (samplers : MxxSamplerFamily)
    (parameters : ParamEnvironment)
    (inputs : Environment)
    (wires : WireEnvironment)
    (states : List OperationalNumericState)
    (address : ExecutionAddress)
    (indexRef outputWire : WireRef)
    (branchRefs : List WireRef)
    (index : Int)
    (branches : List Mxx.Matrix)
    (output : Mxx.Matrix)
    (matrixType : MatrixTypeExpr)
    (factParams : Mxx.SamplerParams)
    (maximum : OperationalBoundExpr)
    (outputCount : Nat)
    (selected : branches[index.toNat]? = some output)
    (typeHolds : matrixType.Holds parameters output)
    (factModulus : output.modulus = factParams.modulus)
    (factRingDimension : output.ringDimension = factParams.ringDimension)
    (factRows : output.rows = factParams.rows)
    (factColumns : output.columns = factParams.columns)
    (magnitudeHolds :
      OperationalBoundExpr.Holds parameters states (.finite maximum) output)
    (argumentsEvaluate : (indexRef :: branchRefs).mapM (fun wire => lookupWire wire wires) =
      some (.integer index :: branches.map Value.matrix))
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .select
      arguments := indexRef :: branchRefs
      outputCount
    }) :
    values = [.matrix output] ∧
      OperationalFact.HoldsEvidence samplers parameters states address (.matrix {
        subject := outputWire
        origin := .value address.template outputWire
        matrixType
        matrixParams := factParams
        magnitude := .finite maximum
      }) (.matrix output) (.matrix {
        originKey := .executionValue address outputWire output
      }) := by
  have selectedValue : (branches.map Value.matrix)[index.toNat]? = some (.matrix output) := by
    rw [List.getElem?_map, selected]
    rfl
  have valuesExact := selectNode_execution runChild samplers parameters inputs wires indexRef
    branchRefs index (branches.map Value.matrix) outputCount argumentsEvaluate member
  have valuesExact' : values = [.matrix output] := by
    rw [valuesExact, selectedValue]
    simp
  exact ⟨valuesExact', localMatrixFact_holdsEvidence samplers parameters states address outputWire
    matrixType factParams (.finite maximum) .unknown output typeHolds factModulus
    factRingDimension factRows factColumns magnitudeHolds trivial⟩

theorem OperationalScopeAlignment.stepConstantInt
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {inputs : Environment}
    {states : List OperationalNumericState}
    {facts : OperationalScopeFacts}
    {wires : WireEnvironment}
    {entries : List AlignedWireEvidence}
    (alignment : OperationalScopeAlignment samplers parameters states facts wires entries)
    (address : ExecutionAddress)
    (value : Int)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .constantInt value
      arguments := []
      outputCount := 1
    }) :
    ∃ outputEntries,
      values = [.integer value] ∧
      OperationalScopeAlignment samplers parameters states (facts.push #[.integer {
        subject := { node := facts.size, port := 0 }
        origin := .local address.template { node := facts.size, port := 0 }
        lower := value
        upper := value
      }]) (wires ++ bindOutputs facts.size values) (entries ++ outputEntries) := by
  let wire : WireRef := { node := facts.size, port := 0 }
  let fact : OperationalFact := .integer {
    subject := wire
    origin := .local address.template wire
    lower := value
    upper := value
  }
  have sound := constantIntNode_operational_sound runChild samplers parameters inputs wires
    states address wire (.local address.template wire) value 1 member
  let entry : AlignedWireEvidence := {
    wire
    fact
    value := .integer value
    evidence := .integer value
    address
  }
  refine ⟨[entry], sound.1, ?_⟩
  rw [sound.1]
  apply alignment.appendOutputs #[fact] (bindOutputs facts.size [.integer value]) [entry]
  simpa [fact, entry, wire] using singletonOutputAligned samplers parameters states facts address
    fact (.integer value) (.integer value) sound.2

theorem OperationalScopeAlignment.stepBooleanInput
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {inputs : Environment}
    {states : List OperationalNumericState}
    {facts : OperationalScopeFacts}
    {wires : WireEnvironment}
    {entries : List AlignedWireEvidence}
    (alignment : OperationalScopeAlignment samplers parameters states facts wires entries)
    (address : ExecutionAddress)
    (name : String)
    (value : Bool)
    (inputLookup : lookupEnvironment name inputs = some (.boolean value))
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .input name
      arguments := []
      outputCount := 1
    }) :
    ∃ outputEntries,
      values = [.boolean value] ∧
      OperationalScopeAlignment samplers parameters states (facts.push #[.boolean])
        (wires ++ bindOutputs facts.size values) (entries ++ outputEntries) := by
  have valuesExact : values = [.boolean value] := by
    simpa [evaluateNode, inputLookup] using member
  let wire : WireRef := { node := facts.size, port := 0 }
  let entry : AlignedWireEvidence := {
    wire
    fact := .boolean
    value := .boolean value
    evidence := .boolean value
    address
  }
  refine ⟨[entry], valuesExact, ?_⟩
  rw [valuesExact]
  apply alignment.appendOutputs #[.boolean] (bindOutputs facts.size [.boolean value]) [entry]
  simpa [entry, wire] using singletonOutputAligned samplers parameters states facts address
    (.boolean) (.boolean value) (.boolean value)
      (booleanFact_holdsEvidence samplers parameters states address value)

theorem OperationalScopeAlignment.stepBoolToInt
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {inputs : Environment}
    {states : List OperationalNumericState}
    {facts : OperationalScopeFacts}
    {wires : WireEnvironment}
    {entries : List AlignedWireEvidence}
    (alignment : OperationalScopeAlignment samplers parameters states facts wires entries)
    (address : ExecutionAddress)
    (inputRef : WireRef)
    (input : Bool)
    (argumentsEvaluate : [inputRef].mapM (fun wire => lookupWire wire wires) =
      some [.boolean input])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .boolToInt
      arguments := [inputRef]
      outputCount := 1
    }) :
    ∃ outputEntries,
      values = [.integer (if input then 1 else 0)] ∧
      OperationalScopeAlignment samplers parameters states (facts.push #[.integer {
        subject := { node := facts.size, port := 0 }
        origin := .local address.template { node := facts.size, port := 0 }
        lower := 0
        upper := 1
      }]) (wires ++ bindOutputs facts.size values) (entries ++ outputEntries) := by
  let wire : WireRef := { node := facts.size, port := 0 }
  let fact : OperationalFact := .integer {
    subject := wire
    origin := .local address.template wire
    lower := 0
    upper := 1
  }
  have sound := boolToIntNode_operational_sound runChild samplers parameters inputs wires states
    address inputRef wire (.local address.template wire) input 1 argumentsEvaluate member
  let entry : AlignedWireEvidence := {
    wire
    fact
    value := .integer (if input then 1 else 0)
    evidence := .integer (if input then 1 else 0)
    address
  }
  refine ⟨[entry], sound.1, ?_⟩
  rw [sound.1]
  apply alignment.appendOutputs #[fact]
    (bindOutputs facts.size [.integer (if input then 1 else 0)]) [entry]
  simpa [fact, entry, wire] using singletonOutputAligned samplers parameters states facts address
    fact (.integer (if input then 1 else 0)) (.integer (if input then 1 else 0)) sound.2

theorem OperationalScopeAlignment.stepZeroMatrix
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {inputs : Environment}
    {states : List OperationalNumericState}
    {facts : OperationalScopeFacts}
    {wires : WireEnvironment}
    {entries : List AlignedWireEvidence}
    (alignment : OperationalScopeAlignment samplers parameters states facts wires entries)
    (address : ExecutionAddress)
    (matrixType : MatrixTypeExpr)
    (matrixParams : Mxx.SamplerParams)
    (typeEvaluates : matrixType.evaluate parameters = some matrixParams)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .zeroMatrix matrixType
      arguments := []
      outputCount := 1
    }) :
    let wire : WireRef := { node := facts.size, port := 0 }
    let fact : OperationalFact := .matrix {
      subject := wire
      origin := .value address.template wire
      matrixType
      matrixParams
      totalHardBound := (.closedInt (.constant 0))
      canonicalRange := .below 1
    }
    let output := zeroConstantOutput matrixParams
    ∃ outputEntries,
      values = [.matrix output] ∧
      OperationalScopeAlignment samplers parameters states (facts.push #[fact])
        (wires ++ bindOutputs facts.size values) (entries ++ outputEntries) := by
  dsimp
  let wire : WireRef := { node := facts.size, port := 0 }
  let fact : OperationalFact := .matrix {
    subject := wire
    origin := .value address.template wire
    matrixType
    matrixParams
    totalHardBound := (.closedInt (.constant 0))
    canonicalRange := .below 1
  }
  let output := zeroConstantOutput matrixParams
  have sound := zeroMatrixNode_operational_sound runChild samplers parameters inputs wires states
    address wire matrixType matrixParams 1 typeEvaluates member
  let evidence : RuntimeFactEvidence := .matrix {
    originKey := .executionValue address wire output
  }
  let entry : AlignedWireEvidence := { wire, fact, value := .matrix output, evidence, address }
  refine ⟨[entry], sound.1, ?_⟩
  rw [sound.1]
  apply alignment.appendOutputs #[fact] (bindOutputs facts.size [.matrix output]) [entry]
  simpa [fact, entry, evidence, wire, output] using
    singletonOutputAligned samplers parameters states facts address fact (.matrix output) evidence sound.2

theorem OperationalScopeAlignment.stepIdentityMatrix
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {inputs : Environment}
    {states : List OperationalNumericState}
    {facts : OperationalScopeFacts}
    {wires : WireEnvironment}
    {entries : List AlignedWireEvidence}
    (alignment : OperationalScopeAlignment samplers parameters states facts wires entries)
    (address : ExecutionAddress)
    (matrixType : MatrixTypeExpr)
    (matrixParams : Mxx.SamplerParams)
    (typeEvaluates : matrixType.evaluate parameters = some matrixParams)
    (modulusPositive : 0 < matrixParams.modulus)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .identityMatrix matrixType
      arguments := []
      outputCount := 1
    }) :
    let wire : WireRef := { node := facts.size, port := 0 }
    let fact : OperationalFact := .matrix {
      subject := wire
      origin := .value address.template wire
      matrixType
      matrixParams
      totalHardBound := (.closedInt (.constant 1))
      canonicalRange := .below 2
    }
    let output := identityConstantOutput matrixParams
    ∃ outputEntries,
      values = [.matrix output] ∧
      OperationalScopeAlignment samplers parameters states (facts.push #[fact])
        (wires ++ bindOutputs facts.size values) (entries ++ outputEntries) := by
  dsimp
  let wire : WireRef := { node := facts.size, port := 0 }
  let fact : OperationalFact := .matrix {
    subject := wire
    origin := .value address.template wire
    matrixType
    matrixParams
    totalHardBound := (.closedInt (.constant 1))
    canonicalRange := .below 2
  }
  let output := identityConstantOutput matrixParams
  have sound := identityMatrixNode_operational_sound runChild samplers parameters inputs wires
    states address wire matrixType matrixParams 1 typeEvaluates modulusPositive member
  let evidence : RuntimeFactEvidence := .matrix {
    originKey := .executionValue address wire output
  }
  let entry : AlignedWireEvidence := { wire, fact, value := .matrix output, evidence, address }
  refine ⟨[entry], sound.1, ?_⟩
  rw [sound.1]
  apply alignment.appendOutputs #[fact] (bindOutputs facts.size [.matrix output]) [entry]
  simpa [fact, entry, evidence, wire, output] using
    singletonOutputAligned samplers parameters states facts address fact (.matrix output) evidence sound.2

theorem OperationalScopeAlignment.stepConstantMatrix
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {inputs : Environment}
    {states : List OperationalNumericState}
    {facts : OperationalScopeFacts}
    {wires : WireEnvironment}
    {entries : List AlignedWireEvidence}
    (alignment : OperationalScopeAlignment samplers parameters states facts wires entries)
    (address : ExecutionAddress)
    (matrixType : MatrixTypeExpr)
    (coefficients : List IntExpr)
    (runtimeParams factParams : Mxx.SamplerParams)
    (evaluated : List Int)
    (typeEvaluates : matrixType.evaluate parameters = some runtimeParams)
    (coefficientsEvaluate : coefficients.mapM (IntExpr.evaluate parameters) = some evaluated)
    (modulusPositive : 0 < runtimeParams.modulus)
    (modulus : runtimeParams.modulus = factParams.modulus)
    (ringDimension : runtimeParams.ringDimension = factParams.ringDimension)
    (rows : runtimeParams.rows = factParams.rows)
    (columns : runtimeParams.columns = factParams.columns)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .constantMatrix matrixType coefficients
      arguments := []
      outputCount := 1
    }) :
    let wire : WireRef := { node := facts.size, port := 0 }
    let bound := evaluated.foldl (fun maximum value => max maximum (absolute value)) 0
    let canonicalMaximum := evaluated.foldl (fun maximum value =>
      max maximum (Mxx.reduceCoefficient runtimeParams.modulus value).toNat) 0
    let fact : OperationalFact := .matrix {
      subject := wire
      origin := .value address.template wire
      matrixType
      matrixParams := factParams
      totalHardBound := (.closedInt (.constant bound))
      canonicalRange := .below (canonicalMaximum + 1)
    }
    let output := Mxx.Matrix.withSamplerParams {
      coefficients := evaluated.map (Mxx.reduceCoefficient runtimeParams.modulus)
    } runtimeParams
    ∃ outputEntries,
      values = [.matrix output] ∧
      OperationalScopeAlignment samplers parameters states (facts.push #[fact])
        (wires ++ bindOutputs facts.size values) (entries ++ outputEntries) := by
  dsimp
  let wire : WireRef := { node := facts.size, port := 0 }
  let bound := evaluated.foldl (fun maximum value => max maximum (absolute value)) 0
  let canonicalMaximum := evaluated.foldl (fun maximum value =>
    max maximum (Mxx.reduceCoefficient runtimeParams.modulus value).toNat) 0
  let fact : OperationalFact := .matrix {
    subject := wire
    origin := .value address.template wire
    matrixType
    matrixParams := factParams
    totalHardBound := (.closedInt (.constant bound))
    canonicalRange := .below (canonicalMaximum + 1)
  }
  let output := Mxx.Matrix.withSamplerParams {
    coefficients := evaluated.map (Mxx.reduceCoefficient runtimeParams.modulus)
  } runtimeParams
  have sound := constantMatrixNode_operational_sound runChild samplers parameters inputs wires
    states address wire matrixType coefficients runtimeParams factParams evaluated 1 typeEvaluates
    coefficientsEvaluate modulusPositive modulus ringDimension rows columns member
  let evidence : RuntimeFactEvidence := .matrix {
    originKey := .executionValue address wire output
  }
  let entry : AlignedWireEvidence := { wire, fact, value := .matrix output, evidence, address }
  refine ⟨[entry], sound.1, ?_⟩
  rw [sound.1]
  apply alignment.appendOutputs #[fact] (bindOutputs facts.size [.matrix output]) [entry]
  simpa [fact, entry, evidence, wire, bound, canonicalMaximum, output] using
    singletonOutputAligned samplers parameters states facts address fact (.matrix output) evidence sound.2

theorem OperationalScopeAlignment.stepGaussian
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    (contract : MxxBoundedSamplerContract samplers)
    {parameters : ParamEnvironment}
    {inputs : Environment}
    {states : List OperationalNumericState}
    {facts : OperationalScopeFacts}
    {wires : WireEnvironment}
    {entries : List AlignedWireEvidence}
    (alignment : OperationalScopeAlignment samplers parameters states facts wires entries)
    (address : ExecutionAddress)
    (matrixType : MatrixTypeExpr)
    (cutoff : IntExpr)
    (runtimeParams layoutParams factParams : Mxx.SamplerParams)
    (bound : Int)
    (typeEvaluates : matrixType.evaluate parameters cutoff = some runtimeParams)
    (layoutEvaluates : matrixType.evaluate parameters = some layoutParams)
    (cutoffEvaluates : cutoff.evaluate parameters = some bound)
    (boundNonnegative : 0 ≤ bound)
    (modulusPositive : 0 < runtimeParams.modulus)
    (runtimeBound : runtimeParams.maxCoefficientBound = bound.toNat)
    (layoutModulus : runtimeParams.modulus = layoutParams.modulus)
    (layoutRingDimension : runtimeParams.ringDimension = layoutParams.ringDimension)
    (layoutRows : runtimeParams.rows = layoutParams.rows)
    (layoutColumns : runtimeParams.columns = layoutParams.columns)
    (modulus : runtimeParams.modulus = factParams.modulus)
    (ringDimension : runtimeParams.ringDimension = factParams.ringDimension)
    (rows : runtimeParams.rows = factParams.rows)
    (columns : runtimeParams.columns = factParams.columns)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .gaussianSample matrixType cutoff
      arguments := []
      outputCount := 1
    }) :
    let wire : WireRef := { node := facts.size, port := 0 }
    let fact : OperationalFact := .matrix {
      subject := wire
      origin := .value address.template wire
      matrixType
      matrixParams := factParams
      totalHardBound := (.minimum
        (.closedInt (.constant (runtimeParams.modulus / 2)))
        (.contextual .maximum parameters [] cutoff))
    }
    ∃ output outputEntries,
      values = [.matrix output] ∧
      OperationalScopeAlignment samplers parameters states (facts.push #[fact])
        (wires ++ bindOutputs facts.size values) (entries ++ outputEntries) := by
  dsimp
  let wire : WireRef := { node := facts.size, port := 0 }
  let fact : OperationalFact := .matrix {
    subject := wire
    origin := .value address.template wire
    matrixType
    matrixParams := factParams
    totalHardBound := (.minimum
      (.closedInt (.constant (runtimeParams.modulus / 2)))
      (.contextual .maximum parameters [] cutoff))
  }
  obtain ⟨output, valuesExact, holds⟩ := gaussianNode_operational_sound runChild samplers contract
    parameters inputs wires states address wire matrixType cutoff runtimeParams layoutParams
    factParams bound 1 typeEvaluates layoutEvaluates cutoffEvaluates boundNonnegative
    modulusPositive runtimeBound layoutModulus layoutRingDimension layoutRows layoutColumns modulus
    ringDimension rows columns member
  let evidence : RuntimeFactEvidence := .matrix {
    originKey := .executionValue address wire output
  }
  let entry : AlignedWireEvidence := { wire, fact, value := .matrix output, evidence, address }
  refine ⟨output, [entry], valuesExact, ?_⟩
  rw [valuesExact]
  apply alignment.appendOutputs #[fact] (bindOutputs facts.size [.matrix output]) [entry]
  simpa [fact, entry, evidence, wire] using
    singletonOutputAligned samplers parameters states facts address fact (.matrix output) evidence holds

theorem OperationalScopeAlignment.stepSelectTwoFinite
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {inputs : Environment}
    {states : List OperationalNumericState}
    {facts : OperationalScopeFacts}
    {wires : WireEnvironment}
    {entries : List AlignedWireEvidence}
    (alignment : OperationalScopeAlignment samplers parameters states facts wires entries)
    (address : ExecutionAddress)
    (indexRef leftRef rightRef : WireRef)
    (index : Int)
    (left right output : Mxx.Matrix)
    (matrixType : MatrixTypeExpr)
    (factParams : Mxx.SamplerParams)
    (leftMaximum rightMaximum : OperationalBoundExpr)
    (leftMaximumValue rightMaximumValue : Int)
    (leftHolds : OperationalBoundExpr.Holds parameters states
      (.finite leftMaximum) left)
    (rightHolds : OperationalBoundExpr.Holds parameters states
      (.finite rightMaximum) right)
    (leftMaximumEvaluates : leftMaximum.evaluateWithStates parameters states = .ok leftMaximumValue)
    (rightMaximumEvaluates : rightMaximum.evaluateWithStates parameters states = .ok rightMaximumValue)
    (selected : [left, right][index.toNat]? = some output)
    (typeHolds : matrixType.Holds parameters output)
    (factModulus : output.modulus = factParams.modulus)
    (factRingDimension : output.ringDimension = factParams.ringDimension)
    (factRows : output.rows = factParams.rows)
    (factColumns : output.columns = factParams.columns)
    (modulusPositive : 0 < output.modulus)
    (argumentsEvaluate : [indexRef, leftRef, rightRef].mapM
      (fun wire => lookupWire wire wires) =
        some [.integer index, .matrix left, .matrix right])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .select
      arguments := [indexRef, leftRef, rightRef]
      outputCount := 1
    }) :
    let wire : WireRef := { node := facts.size, port := 0 }
    let branchMaximum := .maximum (.maximum (.closedInt (.constant 0)) leftMaximum) rightMaximum
    let cappedMaximum := .minimum
      (.closedInt (.constant (output.modulus / 2))) branchMaximum
    let fact : OperationalFact := .matrix {
      subject := wire
      origin := .value address.template wire
      matrixType
      matrixParams := factParams
      magnitude := .finite cappedMaximum
    }
    ∃ outputEntries,
      values = [.matrix output] ∧
      OperationalScopeAlignment samplers parameters states (facts.push #[fact])
        (wires ++ bindOutputs facts.size values) (entries ++ outputEntries) := by
  dsimp
  let wire : WireRef := { node := facts.size, port := 0 }
  let branchMaximum : OperationalBoundExpr :=
    .maximum (.maximum (.closedInt (.constant 0)) leftMaximum) rightMaximum
  let cappedMaximum : OperationalBoundExpr :=
    .minimum (.closedInt (.constant (output.modulus / 2))) branchMaximum
  let fact : OperationalFact := .matrix {
    subject := wire
    origin := .value address.template wire
    matrixType
    matrixParams := factParams
    magnitude := .finite cappedMaximum
  }
  have branchHolds : OperationalBoundExpr.Holds parameters states
      (.finite branchMaximum) output := by
    simpa [branchMaximum] using selectedTwoMagnitudeHolds leftHolds rightHolds selected
  have branchEvaluates : branchMaximum.evaluateWithStates parameters states =
      .ok (max (max 0 leftMaximumValue) rightMaximumValue) := by
    simp only [branchMaximum, OperationalBoundExpr.evaluateWithStates,
      OperationalBoundExpr.evaluateWithStates_closedConstant]
    rw [leftMaximumEvaluates, rightMaximumEvaluates]
    rfl
  have cappedHolds : OperationalBoundExpr.Holds parameters states
      (.finite cappedMaximum) output := by
    simpa [cappedMaximum] using branchHolds.minimumCentered branchEvaluates rfl modulusPositive
  have sound := matrixSelectNode_operational_sound runChild samplers parameters inputs wires states
    address indexRef wire [leftRef, rightRef] index [left, right] output matrixType factParams
    cappedMaximum 1 selected typeHolds factModulus factRingDimension factRows factColumns
    cappedHolds argumentsEvaluate member
  let evidence : RuntimeFactEvidence := .matrix {
    originKey := .executionValue address wire output
  }
  let entry : AlignedWireEvidence := { wire, fact, value := .matrix output, evidence, address }
  refine ⟨[entry], sound.1, ?_⟩
  rw [sound.1]
  apply alignment.appendOutputs #[fact] (bindOutputs facts.size [.matrix output]) [entry]
  simpa [fact, entry, evidence, wire] using singletonOutputAligned samplers parameters states facts
    address fact (.matrix output) evidence sound.2

theorem OperationalScopeAlignment.stepMatrixAddFinite
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {inputs : Environment}
    {states : List OperationalNumericState}
    {facts : OperationalScopeFacts}
    {wires : WireEnvironment}
    {entries : List AlignedWireEvidence}
    (alignment : OperationalScopeAlignment samplers parameters states facts wires entries)
    (address : ExecutionAddress)
    (leftRef rightRef : WireRef)
    (left right : Mxx.Matrix)
    (matrixType : MatrixTypeExpr)
    (layoutParams factParams : Mxx.SamplerParams)
    (leftMaximum rightMaximum : OperationalBoundExpr)
    (typeEvaluates : matrixType.evaluate parameters = some layoutParams)
    (leftTypeHolds : matrixType.Holds parameters left)
    (rightWellFormed : right.WellFormed)
    (sameModulus : left.modulus = right.modulus)
    (sameRingDimension : left.ringDimension = right.ringDimension)
    (sameRows : left.rows = right.rows)
    (sameColumns : left.columns = right.columns)
    (factModulus : left.modulus = factParams.modulus)
    (factRingDimension : left.ringDimension = factParams.ringDimension)
    (factRows : left.rows = factParams.rows)
    (factColumns : left.columns = factParams.columns)
    (modulusPositive : 0 < left.modulus)
    (leftMagnitude : OperationalBoundExpr.Holds parameters states
      (.finite leftMaximum) left)
    (rightMagnitude : OperationalBoundExpr.Holds parameters states
      (.finite rightMaximum) right)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix left, .matrix right])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .matrixAdd
      arguments := [leftRef, rightRef]
      outputCount := 1
    }) :
    let wire : WireRef := { node := facts.size, port := 0 }
    let maximum := .add (.add (.closedInt (.constant 0)) leftMaximum) rightMaximum
    let cappedMaximum := .minimum (.closedInt (.constant (left.modulus / 2))) maximum
    let fact : OperationalFact := .matrix {
      subject := wire
      origin := .value address.template wire
      matrixType
      matrixParams := factParams
      magnitude := .finite cappedMaximum
    }
    ∃ output outputEntries,
      values = [.matrix output] ∧
      OperationalScopeAlignment samplers parameters states (facts.push #[fact])
        (wires ++ bindOutputs facts.size values) (entries ++ outputEntries) := by
  dsimp
  let wire : WireRef := { node := facts.size, port := 0 }
  let maximum : OperationalBoundExpr :=
    .add (.add (.closedInt (.constant 0)) leftMaximum) rightMaximum
  let cappedMaximum : OperationalBoundExpr :=
    .minimum (.closedInt (.constant (left.modulus / 2))) maximum
  let fact : OperationalFact := .matrix {
    subject := wire
    origin := .value address.template wire
    matrixType
    matrixParams := factParams
    magnitude := .finite cappedMaximum
  }
  obtain ⟨output, valuesExact, holds⟩ := matrixAddNode_operational_sound_capped runChild samplers
    parameters inputs wires states address leftRef rightRef wire left right matrixType layoutParams
    factParams leftMaximum rightMaximum 1 typeEvaluates leftTypeHolds rightWellFormed sameModulus
    sameRingDimension sameRows sameColumns factModulus factRingDimension factRows factColumns
    modulusPositive leftMagnitude rightMagnitude argumentsEvaluate member
  let evidence : RuntimeFactEvidence := .matrix {
    originKey := .executionValue address wire output
  }
  let entry : AlignedWireEvidence := { wire, fact, value := .matrix output, evidence, address }
  refine ⟨output, [entry], valuesExact, ?_⟩
  rw [valuesExact]
  apply alignment.appendOutputs #[fact] (bindOutputs facts.size [.matrix output]) [entry]
  simpa [fact, entry, evidence, wire, maximum, cappedMaximum] using
    singletonOutputAligned samplers parameters states facts address fact (.matrix output) evidence holds

theorem OperationalScopeAlignment.stepMatrixSubtractFinite
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {inputs : Environment}
    {states : List OperationalNumericState}
    {facts : OperationalScopeFacts}
    {wires : WireEnvironment}
    {entries : List AlignedWireEvidence}
    (alignment : OperationalScopeAlignment samplers parameters states facts wires entries)
    (address : ExecutionAddress)
    (leftRef rightRef : WireRef)
    (left right : Mxx.Matrix)
    (matrixType : MatrixTypeExpr)
    (layoutParams factParams : Mxx.SamplerParams)
    (leftMaximum rightMaximum : OperationalBoundExpr)
    (typeEvaluates : matrixType.evaluate parameters = some layoutParams)
    (leftTypeHolds : matrixType.Holds parameters left)
    (rightWellFormed : right.WellFormed)
    (sameModulus : left.modulus = right.modulus)
    (sameRingDimension : left.ringDimension = right.ringDimension)
    (sameRows : left.rows = right.rows)
    (sameColumns : left.columns = right.columns)
    (factModulus : left.modulus = factParams.modulus)
    (factRingDimension : left.ringDimension = factParams.ringDimension)
    (factRows : left.rows = factParams.rows)
    (factColumns : left.columns = factParams.columns)
    (modulusPositive : 0 < left.modulus)
    (leftMagnitude : OperationalBoundExpr.Holds parameters states
      (.finite leftMaximum) left)
    (rightMagnitude : OperationalBoundExpr.Holds parameters states
      (.finite rightMaximum) right)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix left, .matrix right])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .matrixSubtract
      arguments := [leftRef, rightRef]
      outputCount := 1
    }) :
    let wire : WireRef := { node := facts.size, port := 0 }
    let maximum := .add (.add (.closedInt (.constant 0)) leftMaximum) rightMaximum
    let cappedMaximum := .minimum (.closedInt (.constant (left.modulus / 2))) maximum
    let fact : OperationalFact := .matrix {
      subject := wire
      origin := .value address.template wire
      matrixType
      matrixParams := factParams
      magnitude := .finite cappedMaximum
    }
    ∃ output outputEntries,
      values = [.matrix output] ∧
      OperationalScopeAlignment samplers parameters states (facts.push #[fact])
        (wires ++ bindOutputs facts.size values) (entries ++ outputEntries) := by
  dsimp
  let wire : WireRef := { node := facts.size, port := 0 }
  let maximum : OperationalBoundExpr :=
    .add (.add (.closedInt (.constant 0)) leftMaximum) rightMaximum
  let cappedMaximum : OperationalBoundExpr :=
    .minimum (.closedInt (.constant (left.modulus / 2))) maximum
  let baseFact : OperationalMatrixFact := {
    subject := wire
    origin := .value address.template wire
    matrixType
    matrixParams := factParams
    magnitude := .finite maximum
  }
  have leftMagnitudeCopy := leftMagnitude
  have rightMagnitudeCopy := rightMagnitude
  rcases leftMagnitude with ⟨leftBound, leftEvaluates, _, _⟩
  rcases rightMagnitude with ⟨rightBound, rightEvaluates, _, _⟩
  obtain ⟨output, valuesExact, baseHolds⟩ := matrixSubtractNode_operational_sound runChild
    samplers parameters inputs wires states address leftRef rightRef wire left right matrixType
    layoutParams factParams leftMaximum rightMaximum 1 typeEvaluates leftTypeHolds
    rightWellFormed sameModulus sameRingDimension sameRows sameColumns factModulus
    factRingDimension factRows factColumns modulusPositive leftMagnitudeCopy rightMagnitudeCopy
    argumentsEvaluate member
  have maximumEvaluates : maximum.evaluateWithStates parameters states =
      .ok (leftBound + rightBound) := by
    simp only [maximum, OperationalBoundExpr.evaluateWithStates,
      OperationalBoundExpr.evaluateWithStates_closedConstant]
    rw [leftEvaluates, rightEvaluates]
    change Except.ok (0 + leftBound + rightBound) = Except.ok (leftBound + rightBound)
    congr 2
    omega
  have outputModulus : output.modulus = left.modulus := by
    exact baseHolds.2.1.trans factModulus.symm
  have capped := baseHolds.capMatrix maximumEvaluates
    (outputModulus.trans_gt modulusPositive)
  let fact : OperationalFact := .matrix { baseFact with magnitude := .finite cappedMaximum }
  let evidence : RuntimeFactEvidence := .matrix {
    originKey := .executionValue address wire output
  }
  let entry : AlignedWireEvidence := { wire, fact, value := .matrix output, evidence, address }
  have holds : OperationalFact.HoldsEvidence samplers parameters states address fact
      (.matrix output) evidence := by
    simpa [fact, baseFact, cappedMaximum, maximum, outputModulus,
      OperationalBoundExpr.centeredCap] using capped
  refine ⟨output, [entry], valuesExact, ?_⟩
  rw [valuesExact]
  apply alignment.appendOutputs #[fact] (bindOutputs facts.size [.matrix output]) [entry]
  simpa [entry, wire] using singletonOutputAligned samplers parameters states facts address fact
    (.matrix output) evidence holds

theorem OperationalScopeAlignment.stepMatrixNegate
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {inputs : Environment}
    {states : List OperationalNumericState}
    {facts : OperationalScopeFacts}
    {wires : WireEnvironment}
    {entries : List AlignedWireEvidence}
    (alignment : OperationalScopeAlignment samplers parameters states facts wires entries)
    (address : ExecutionAddress)
    (inputRef : WireRef)
    (input : Mxx.Matrix)
    (matrixType : MatrixTypeExpr)
    (factParams : Mxx.SamplerParams)
    (magnitude : OperationalBoundExpr)
    (inputTypeHolds : matrixType.Holds parameters input)
    (factModulus : input.modulus = factParams.modulus)
    (factRingDimension : input.ringDimension = factParams.ringDimension)
    (factRows : input.rows = factParams.rows)
    (factColumns : input.columns = factParams.columns)
    (modulusPositive : 0 < input.modulus)
    (inputMagnitude : magnitude.Holds parameters states input)
    (bound : Int)
    (maximumEvaluates : magnitude.boundExpression.evaluateWithStates parameters states = .ok bound)
    (argumentsEvaluate : [inputRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix input])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .matrixNegate
      arguments := [inputRef]
      outputCount := 1
    }) :
    let wire : WireRef := { node := facts.size, port := 0 }
    let cappedMagnitude := magnitude.centeredCap input.modulus
    let fact : OperationalFact := .matrix {
      subject := wire
      origin := .value address.template wire
      matrixType
      matrixParams := factParams
      magnitude := cappedMagnitude
    }
    let output := Mxx.matrixNegate input
    ∃ outputEntries,
      values = [.matrix output] ∧
      OperationalScopeAlignment samplers parameters states (facts.push #[fact])
        (wires ++ bindOutputs facts.size values) (entries ++ outputEntries) := by
  dsimp
  let wire : WireRef := { node := facts.size, port := 0 }
  let baseFact : OperationalMatrixFact := {
    subject := wire
    origin := .value address.template wire
    matrixType
    matrixParams := factParams
    magnitude
  }
  have sound := matrixNegateNode_operational_sound runChild samplers parameters inputs wires states
    address inputRef wire input matrixType factParams magnitude 1 inputTypeHolds factModulus
    factRingDimension factRows factColumns modulusPositive inputMagnitude argumentsEvaluate member
  let output := Mxx.matrixNegate input
  have outputModulus : output.modulus = input.modulus := by rfl
  have capped := sound.2.capMatrix maximumEvaluates (outputModulus.trans_gt modulusPositive)
  let cappedMagnitude := magnitude.centeredCap input.modulus
  let fact : OperationalFact := .matrix { baseFact with magnitude := cappedMagnitude }
  let evidence : RuntimeFactEvidence := .matrix {
    originKey := .executionValue address wire output
  }
  let entry : AlignedWireEvidence := { wire, fact, value := .matrix output, evidence, address }
  refine ⟨[entry], sound.1, ?_⟩
  rw [sound.1]
  apply alignment.appendOutputs #[fact] (bindOutputs facts.size [.matrix output]) [entry]
  have capped' : OperationalFact.HoldsEvidence samplers parameters states address fact
      (.matrix output) evidence := by
    simpa [fact, baseFact, cappedMagnitude, output, outputModulus] using capped
  simpa [entry, wire] using singletonOutputAligned samplers parameters states facts address fact
    (.matrix output) evidence capped'

theorem OperationalScopeAlignment.stepThresholdDecodeBoolOne
    {runChild : ChildRunner}
    {samplers : MxxSamplerFamily}
    {parameters : ParamEnvironment}
    {inputs : Environment}
    {states : List OperationalNumericState}
    {facts : OperationalScopeFacts}
    {wires : WireEnvironment}
    {entries : List AlignedWireEvidence}
    (alignment : OperationalScopeAlignment samplers parameters states facts wires entries)
    (address : ExecutionAddress)
    (matrixRef : WireRef)
    (matrix : Mxx.Matrix)
    (ciphertextModulus plaintextModulus length : IntExpr)
    (evaluatedCiphertextModulus evaluatedPlaintextModulus evaluatedLength : Int)
    (value : Bool)
    (argumentsEvaluate : [matrixRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix matrix])
    (ciphertextModulusEvaluate :
      ciphertextModulus.evaluate parameters = some evaluatedCiphertextModulus)
    (plaintextModulusEvaluate :
      plaintextModulus.evaluate parameters = some evaluatedPlaintextModulus)
    (lengthEvaluate : length.evaluate parameters = some evaluatedLength)
    (lengthNonnegative : 0 ≤ evaluatedLength)
    (decodedValue :
      ((matrix.coefficients.take evaluatedLength.toNat).map fun coefficient =>
        Value.boolean (Mxx.Ir.thresholdDecodeBool evaluatedCiphertextModulus
          evaluatedPlaintextModulus coefficient)) = [.boolean value])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers parameters inputs wires {
      kind := .thresholdDecodeBool ciphertextModulus plaintextModulus length
      arguments := [matrixRef]
      outputCount := 1
    }) :
    ∃ outputEntries,
      values = [.boolean value] ∧
      OperationalScopeAlignment samplers parameters states (facts.push #[.boolean])
        (wires ++ bindOutputs facts.size values) (entries ++ outputEntries) := by
  obtain ⟨valuesDecoded, portHolds⟩ := thresholdDecodeBoolNode_operational_sound runChild
    samplers parameters inputs wires states address matrixRef matrix ciphertextModulus
    plaintextModulus length evaluatedCiphertextModulus evaluatedPlaintextModulus evaluatedLength
    1 argumentsEvaluate ciphertextModulusEvaluate plaintextModulusEvaluate lengthEvaluate
    lengthNonnegative member
  have valuesExact : values = [.boolean value] := valuesDecoded.trans decodedValue
  have holds := portHolds 0 value (by simp [valuesExact])
  let wire : WireRef := { node := facts.size, port := 0 }
  let entry : AlignedWireEvidence := {
    wire
    fact := .boolean
    value := .boolean value
    evidence := .boolean value
    address
  }
  refine ⟨[entry], valuesExact, ?_⟩
  rw [valuesExact]
  apply alignment.appendOutputs #[.boolean] (bindOutputs facts.size [.boolean value]) [entry]
  simpa [entry, wire] using singletonOutputAligned samplers parameters states facts address
    (.boolean) (.boolean value) (.boolean value) holds

end Mxx.Certificate
-/
