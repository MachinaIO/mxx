import Mxx.Certificate.LocalSoundness
import Mxx.Certificate.Bounds
import Mxx.Certificate.Workflow
import Mxx.Certificate.CanonicalResidues

namespace Mxx.Certificate

/-- Values assigned to stable certificate identities while interpreting derived facts. -/
structure FactEnvironment where
  parameters : Mxx.Ir.ParamEnvironment
  recurrenceBounds : CheckedRecurrenceBoundTable := {}
  analysis : Option AnalysisResult := none
  expressionArena : ExpressionArena := {}
  values : ValueInstanceRef → Option Mxx.Ir.Value

def FactEnvironment.bind
    (environment : FactEnvironment)
    (reference : ValueInstanceRef)
    (value : Mxx.Ir.Value) : FactEnvironment := {
  environment with
  values := fun candidate =>
    if candidate = reference then some value else environment.values candidate
}

/-- Attach the immutable analyzer-owned arena. This changes no runtime value binding. -/
def FactEnvironment.forAnalysis
    (environment : FactEnvironment)
    (analysis : AnalysisResult) : FactEnvironment := {
  environment with analysis := some analysis, expressionArena := analysis.expressionArena
}

@[simp] theorem FactEnvironment.bind_same
    (environment : FactEnvironment)
    (reference : ValueInstanceRef)
    (value : Mxx.Ir.Value) :
    (environment.bind reference value).values reference = some value := by
  simp [FactEnvironment.bind]

theorem FactEnvironment.bind_other
    (environment : FactEnvironment)
    (reference other : ValueInstanceRef)
    (value : Mxx.Ir.Value)
    (different : other ≠ reference) :
    (environment.bind reference value).values other = environment.values other := by
  simp [FactEnvironment.bind, different]

def FactEnvironment.ofWireEnvironment
    (parameters : Mxx.Ir.ParamEnvironment)
    (stage : StageId)
    (scope : StaticScopeId)
    (wires : Mxx.Ir.WireEnvironment) : FactEnvironment := {
  parameters
  values := fun reference =>
    match reference with
    | .concrete wire =>
        if wire.stage = stage && wire.scope = scope then
          Mxx.Ir.lookupWire ⟨wire.node.value, wire.port⟩ wires
        else none
    | _ => none
}

theorem FactEnvironment.ofWireEnvironment_lookup
    (parameters : Mxx.Ir.ParamEnvironment)
    (stage : StageId)
    (scope : StaticScopeId)
    (wires : Mxx.Ir.WireEnvironment)
    (wire : Mxx.Ir.WireRef) :
    (FactEnvironment.ofWireEnvironment parameters stage scope wires).values
        (.concrete {
          stage
          scope
          node := ⟨wire.node⟩
          port := wire.port
        }) =
      Mxx.Ir.lookupWire wire wires := by
  simp [FactEnvironment.ofWireEnvironment]

mutual
/-- Denotation for the closed scalar fragment currently used by certificate rules. Constructors
are intentionally absent for unsupported syntax, so proofs cannot assign it an ad-hoc meaning. -/
inductive RuntimeIntExpr.Denotes (environment : FactEnvironment) :
    RuntimeExpr .integer → Int → Prop where
  | intWire {wire : ValueInstanceRef} {value : Int}
      (lookup : environment.values wire = some (.integer value)) :
      RuntimeIntExpr.Denotes environment (.intWire wire) value
  | intConstant (value : Int) : RuntimeIntExpr.Denotes environment (.intConstant value) value
  | parameter {expression value}
      (evaluates : evaluateIntExpr environment.parameters expression = .ok value) :
      RuntimeIntExpr.Denotes environment (.parameter expression) value
  | boolToInt {expression : RuntimeExpr .boolean} {value : Bool}
      (input : RuntimeBoolExpr.Denotes environment expression value) :
      RuntimeIntExpr.Denotes environment (.boolToInt expression) (if value then 1 else 0)
  | intBinary {operation left right leftValue rightValue value}
      (leftDenotes : RuntimeIntExpr.Denotes environment left leftValue)
      (rightDenotes : RuntimeIntExpr.Denotes environment right rightValue)
      (evaluates : Mxx.Ir.evaluateIntBinary operation leftValue rightValue = some value) :
      RuntimeIntExpr.Denotes environment (.intBinary operation left right) value
  | familyElement {aggregate indexRef index indexValue value}
      (arenaLookup : environment.expressionArena.lookupInteger indexRef = some index)
      (indexDenotes : RuntimeIntExpr.Denotes environment index indexValue)
      (lookup : environment.values (.familyElement aggregate indexRef) = some (.integer value)) :
      RuntimeIntExpr.Denotes environment
        (.familyElement .integer aggregate indexRef index) value
  | select {index branches indexValue branchRef branch value}
      (indexDenotes : RuntimeIntExpr.Denotes environment index indexValue)
      (nonnegative : 0 ≤ indexValue)
      (selected : branches[indexValue.toNat]? = some branchRef)
      (arenaLookup : environment.expressionArena.lookupInteger branchRef = some branch)
      (branchDenotes : RuntimeIntExpr.Denotes environment branch value) :
      RuntimeIntExpr.Denotes environment (.select .integer index branches) value

inductive RuntimeBoolExpr.Denotes (environment : FactEnvironment) :
    RuntimeExpr .boolean → Bool → Prop where
  | boolWire {wire : ValueInstanceRef} {value : Bool}
      (lookup : environment.values wire = some (.boolean value)) :
      RuntimeBoolExpr.Denotes environment (.boolWire wire) value
  | boolConstant (value : Bool) : RuntimeBoolExpr.Denotes environment (.boolConstant value) value
  | compare {operation left right leftValue rightValue}
      (leftDenotes : RuntimeIntExpr.Denotes environment left leftValue)
      (rightDenotes : RuntimeIntExpr.Denotes environment right rightValue) :
      RuntimeBoolExpr.Denotes environment (.compare operation left right)
        (Mxx.Ir.evaluateIntCompare operation leftValue rightValue)
  | bitExtract {expression value bit bitValue}
      (input : RuntimeIntExpr.Denotes environment expression value)
      (bitEvaluates : evaluateIntExpr environment.parameters bit = .ok bitValue)
      (nonnegative : 0 ≤ bitValue) :
      RuntimeBoolExpr.Denotes environment (.bitExtract expression bit)
        (((value / (2 ^ bitValue.toNat)) % 2) ≠ 0)
  | thresholdDecodeBool {matrix ciphertextModulus plaintextModulus position value q p index}
      (matrixLookup : environment.values matrix = some (.matrix value))
      (qEvaluates : evaluateIntExpr environment.parameters ciphertextModulus = .ok q)
      (pEvaluates : evaluateIntExpr environment.parameters plaintextModulus = .ok p)
      (positionEvaluates : evaluateIntExpr environment.parameters position = .ok index)
      (nonnegative : 0 ≤ index)
      {valueCoefficient : Int}
      (coefficient : value.coefficients[index.toNat]? = some valueCoefficient) :
      RuntimeBoolExpr.Denotes environment
        (.thresholdDecodeBool matrix ciphertextModulus plaintextModulus position)
        (Mxx.Ir.thresholdDecodeBool q p valueCoefficient)
  | familyElement {aggregate indexRef index indexValue value}
      (arenaLookup : environment.expressionArena.lookupInteger indexRef = some index)
      (indexDenotes : RuntimeIntExpr.Denotes environment index indexValue)
      (lookup : environment.values (.familyElement aggregate indexRef) = some (.boolean value)) :
      RuntimeBoolExpr.Denotes environment
        (.familyElement .boolean aggregate indexRef index) value
  | select {index branches indexValue branchRef branch value}
      (indexDenotes : RuntimeIntExpr.Denotes environment index indexValue)
      (nonnegative : 0 ≤ indexValue)
      (selected : branches[indexValue.toNat]? = some branchRef)
      (arenaLookup : environment.expressionArena.lookupBoolean branchRef = some branch)
      (branchDenotes : RuntimeBoolExpr.Denotes environment branch value) :
      RuntimeBoolExpr.Denotes environment (.select .boolean index branches) value
end

private def rowEmbedValue
    (parts : List Mxx.SamplerParams) (part : Nat) (value : Mxx.Matrix) : Mxx.Matrix :=
  Mxx.matrixConcatRows <| parts.zipIdx.map fun (params, index) ↦
    if index = part then value
    else zeroConstantOutput { params with columns := value.columns }

private def columnEmbedValue
    (parts : List Mxx.SamplerParams) (part : Nat) (value : Mxx.Matrix) : Mxx.Matrix :=
  Mxx.matrixConcatColumns <| parts.zipIdx.map fun (params, index) ↦
    if index = part then value
    else zeroConstantOutput { params with rows := value.rows }

/-- Algebraic denotation of the supported matrix-expression fragment. -/
inductive MatrixExpr.Denotes (environment : FactEnvironment) :
    MatrixExpr → Mxx.Matrix → Prop where
  | wire {reference value}
      (lookup : environment.values reference.value = some (.matrix value)) :
      Denotes environment (.wire reference) value
  | zero {type params}
      (evaluates : type.evaluate environment.parameters = some params) :
      Denotes environment (.zero type) (zeroConstantOutput params)
  | identity {type params}
      (evaluates : type.evaluate environment.parameters = some params) :
      Denotes environment (.identity type) (identityConstantOutput params)
  | gadget {type base params evaluatedBase}
      (typeEvaluates : type.evaluate environment.parameters = some params)
      (baseEvaluates : base.evaluate environment.parameters = some evaluatedBase) :
      Denotes environment (.gadget type base)
        (Mxx.gadgetMatrix params evaluatedBase
          (if params.rows = 0 then 0 else params.columns / params.rows))
  | add {left right leftValue rightValue}
      (leftDenotes : Denotes environment left leftValue)
      (rightDenotes : Denotes environment right rightValue) :
      Denotes environment (.add left right) (Mxx.matrixAdd leftValue rightValue)
  | negate {expression value}
      (denotes : Denotes environment expression value) :
      Denotes environment (.negate expression) (Mxx.matrixNegate value)
  | multiply {left right leftValue rightValue}
      (leftDenotes : Denotes environment left leftValue)
      (rightDenotes : Denotes environment right rightValue) :
      Denotes environment (.multiply left right) (Mxx.matrixMultiply leftValue rightValue)
  | scalarMultiply {scalar expression scalarValue value}
      (scalarEvaluates : evaluateIntExpr environment.parameters scalar = .ok scalarValue)
      (denotes : Denotes environment expression value) :
      Denotes environment (.scalarMultiply scalar expression) (Mxx.matrixScale scalarValue value)
  | rowSlice {expression value start stop evaluatedStart evaluatedStop}
      (denotes : Denotes environment expression value)
      (startEvaluates : evaluateIntExpr environment.parameters start = .ok evaluatedStart)
      (stopEvaluates : evaluateIntExpr environment.parameters stop = .ok evaluatedStop)
      (startNonnegative : 0 ≤ evaluatedStart)
      (ordered : evaluatedStart ≤ evaluatedStop) :
      Denotes environment (.rowSlice expression start stop)
        (Mxx.matrixSlice value evaluatedStart.toNat evaluatedStop.toNat 0 value.columns)
  | columnSlice {expression value start stop evaluatedStart evaluatedStop}
      (denotes : Denotes environment expression value)
      (startEvaluates : evaluateIntExpr environment.parameters start = .ok evaluatedStart)
      (stopEvaluates : evaluateIntExpr environment.parameters stop = .ok evaluatedStop)
      (startNonnegative : 0 ≤ evaluatedStart)
      (ordered : evaluatedStart ≤ evaluatedStop) :
      Denotes environment (.columnSlice expression start stop)
        (Mxx.matrixSlice value 0 value.rows evaluatedStart.toNat evaluatedStop.toNat)
  | rowConcat {expressions values}
      (denotes : List.Forall₂ (Denotes environment) expressions values) :
      Denotes environment (.rowConcat expressions) (Mxx.matrixConcatRows values)
  | columnConcat {expressions values}
      (denotes : List.Forall₂ (Denotes environment) expressions values) :
      Denotes environment (.columnConcat expressions) (Mxx.matrixConcatColumns values)
  | diagonalConcat {expressions values}
      (denotes : List.Forall₂ (Denotes environment) expressions values) :
      Denotes environment (.diagonalConcat expressions) (Mxx.matrixConcatDiagonal values)
  | rowCoefficientEmbed {layout part expression value params}
      (denotes : Denotes environment expression value)
      (partsEvaluate :
        layout.parts.mapM (fun entry ↦ entry.matrixType.evaluate environment.parameters) =
          some params) :
      Denotes environment (.rowCoefficientEmbed layout part expression)
        (rowEmbedValue params part value)
  | columnBasisEmbed {layout part expression value params}
      (denotes : Denotes environment expression value)
      (partsEvaluate :
        layout.parts.mapM (fun entry ↦ entry.matrixType.evaluate environment.parameters) =
          some params) :
      Denotes environment (.columnBasisEmbed layout part expression)
        (columnEmbedValue params part value)
  | diagonalCoefficientEmbed {layout part expression value params}
      (denotes : Denotes environment expression value)
      (partsEvaluate :
        layout.parts.mapM (fun entry ↦ entry.matrixType.evaluate environment.parameters) =
          some params) :
      Denotes environment (.diagonalCoefficientEmbed layout part expression)
        (rowEmbedValue params part value)
  | diagonalBasisEmbed {layout part expression value params}
      (denotes : Denotes environment expression value)
      (partsEvaluate :
        layout.parts.mapM (fun entry ↦ entry.matrixType.evaluate environment.parameters) =
          some params) :
      Denotes environment (.diagonalBasisEmbed layout part expression)
        (columnEmbedValue params part value)
  | select {index branches indexValue branch value}
      (indexDenotes : RuntimeIntExpr.Denotes environment index indexValue)
      (nonnegative : 0 ≤ indexValue)
      (selected : branches[indexValue.toNat]? = some branch)
      (branchDenotes : Denotes environment branch value) :
      Denotes environment (.select index branches) value

def BoundedMatrixExpr.Holds
    (environment : FactEnvironment)
    (expression : BoundedMatrixExpr)
    (value : Mxx.Matrix) : Prop :=
  MatrixExpr.Denotes environment expression.expression value ∧
    ∃ bound, expression.normBound.evaluate environment.parameters = .ok bound ∧
      Mxx.maxCenteredCoefficientNorm value ≤ bound

/-- A signal term denotes its basis directly when its coefficient is the canonical symbolic
identity. Other supported terms denote executable matrix multiplication. -/
inductive SignalTerm.Denotes (environment : FactEnvironment) :
    SignalTerm → Mxx.Matrix → Prop where
  | identityCoefficient {identityType basis mode value}
      (basisDenotes : MatrixExpr.Denotes environment basis value) :
      Denotes environment {
        coefficient := { expression := .identity identityType, normBound := .constant 1 }
        basis
        mode
      } value
  | product {coefficient basis mode coefficientValue basisValue}
      (coefficientHolds : coefficient.Holds environment coefficientValue)
      (basisDenotes : MatrixExpr.Denotes environment basis basisValue) :
      Denotes environment { coefficient, basis, mode }
        (Mxx.matrixMultiply coefficientValue basisValue)

def AffineForm.Holds
    (environment : FactEnvironment)
    (form : AffineForm)
    (value : Mxx.Matrix) : Prop :=
  ∃ termValues noise noiseBound,
    List.Forall₂ (SignalTerm.Denotes environment) form.terms termValues ∧
    form.noiseBound.evaluate environment.parameters = .ok noiseBound ∧
    Mxx.maxCenteredCoefficientNorm noise ≤ noiseBound ∧
    Mxx.MatrixModEq value (termValues.foldr Mxx.matrixAdd noise)

/-- Affine semantics depends only on the typed value in `R_q`, not on the chosen stored integer
representatives. -/
theorem AffineForm.Holds.of_modEq
    {environment : FactEnvironment}
    {form : AffineForm}
    {left right : Mxx.Matrix}
    (relation : Mxx.MatrixModEq left right)
    (holds : form.Holds environment right) : form.Holds environment left := by
  obtain ⟨termValues, noise, noiseBound, termsDenote, boundEvaluates, noiseNorm,
    reconstruction⟩ := holds
  exact ⟨termValues, noise, noiseBound, termsDenote, boundEvaluates, noiseNorm,
    relation.trans reconstruction⟩

def MatrixPrimaryForm.Holds
    (environment : FactEnvironment)
    (form : MatrixPrimaryForm)
    (value : Mxx.Matrix) : Prop :=
  match form with
  | .exact expression => MatrixExpr.Denotes environment expression value
  | MatrixPrimaryForm.affine affineForm => AffineForm.Holds environment affineForm value

def MatrixRelation.Holds
    (environment : FactEnvironment) : MatrixRelation → Prop
  | .preimage subject source target trapdoor =>
      ∃ subjectValue sourceValue targetValue sourceParams targetParams,
        environment.values subject = some (.matrix subjectValue) ∧
        environment.values source.value = some (.matrix sourceValue) ∧
        environment.values target.value = some (.matrix targetValue) ∧
        environment.values trapdoor = some (.trapdoor sourceValue) ∧
        source.type.evaluate environment.parameters (.constant 0) = some sourceParams ∧
        target.type.evaluate environment.parameters (.constant 0) = some targetParams ∧
        Mxx.Toolkit.MatrixLayout sourceValue sourceParams.modulus sourceParams.ringDimension
          sourceParams.rows sourceParams.columns ∧
        Mxx.Toolkit.MatrixLayout subjectValue sourceParams.modulus sourceParams.ringDimension
          sourceParams.columns targetParams.columns ∧
        Mxx.Toolkit.MatrixLayout targetValue targetParams.modulus targetParams.ringDimension
          targetParams.rows targetParams.columns ∧
        sourceParams.modulus = targetParams.modulus ∧
        sourceParams.ringDimension = targetParams.ringDimension ∧
        sourceParams.rows = targetParams.rows ∧
        Mxx.Toolkit.matrixValue sourceParams.modulus.toNat sourceParams.ringDimension
            sourceParams.rows targetParams.columns (Mxx.matrixMul sourceValue subjectValue) =
          Mxx.Toolkit.matrixValue sourceParams.modulus.toNat sourceParams.ringDimension
            sourceParams.rows targetParams.columns targetValue
  | .gadgetDecomposition subject target base digitCount =>
      ∃ subjectValue targetValue matrixParams evaluatedBase evaluatedDigitCount,
        environment.values subject = some (.matrix subjectValue) ∧
        environment.values target.value = some (.matrix targetValue) ∧
        target.type.evaluate environment.parameters (.constant 0) = some matrixParams ∧
        evaluateIntExpr environment.parameters base = .ok evaluatedBase ∧
        evaluateIntExpr environment.parameters digitCount = .ok evaluatedDigitCount ∧
        Mxx.Toolkit.MatrixLayout targetValue matrixParams.modulus matrixParams.ringDimension
          matrixParams.rows matrixParams.columns ∧
        Mxx.Toolkit.MatrixLayout subjectValue matrixParams.modulus matrixParams.ringDimension
          (matrixParams.rows * evaluatedDigitCount.toNat) matrixParams.columns ∧
        Mxx.Toolkit.matrixValue matrixParams.modulus.toNat matrixParams.ringDimension
            matrixParams.rows matrixParams.columns
            (Mxx.matrixMul
              (Mxx.gadgetMatrix {
                matrixParams with
                columns := matrixParams.rows * evaluatedDigitCount.toNat
              } evaluatedBase evaluatedDigitCount.toNat)
              subjectValue) =
          Mxx.Toolkit.matrixValue matrixParams.modulus.toNat matrixParams.ringDimension
            matrixParams.rows matrixParams.columns targetValue

def MatrixFact.Holds
    (environment : FactEnvironment)
    (fact : MatrixFact) : Prop :=
  ∃ value totalBound,
    environment.values fact.subject = some (.matrix value) ∧
    fact.primary.Holds environment value ∧
    (∀ relation ∈ fact.relations, relation.Holds environment) ∧
    fact.totalNormBound.evaluate environment.parameters = .ok totalBound ∧
    Mxx.maxCenteredCoefficientNorm value ≤ totalBound ∧
    match fact.coefficientRepresentation with
    | .unknown => True
    | .canonicalResidues modulus =>
        ∃ evaluatedModulus,
          evaluateIntExpr environment.parameters modulus = .ok evaluatedModulus ∧
          0 < evaluatedModulus ∧ MatrixHasCanonicalResidues evaluatedModulus value

theorem exactMatrixFact_holds
    (environment : FactEnvironment)
    (subject : ValueInstanceRef)
    (expression : MatrixExpr)
    (value : Mxx.Matrix)
    (bound : BoundExpr)
    (boundValue : Nat)
    (lookup : environment.values subject = some (.matrix value))
    (denotes : MatrixExpr.Denotes environment expression value)
    (boundEvaluates : bound.evaluate environment.parameters = .ok boundValue)
    (normBound : Mxx.maxCenteredCoefficientNorm value ≤ boundValue) :
    MatrixFact.Holds environment {
      subject
      primary := .exact expression
      relations := []
      totalNormBound := bound
    } := by
  exact ⟨value, boundValue, lookup, denotes, by simp, boundEvaluates, normBound, trivial⟩

theorem boundedMatrixFact_holds
    (environment : FactEnvironment)
    (subject : ValueInstanceRef)
    (value : Mxx.Matrix)
    (bound : BoundExpr)
    (boundValue : Nat)
    (lookup : environment.values subject = some (.matrix value))
    (boundEvaluates : bound.evaluate environment.parameters = .ok boundValue)
    (normBound : Mxx.maxCenteredCoefficientNorm value ≤ boundValue) :
    MatrixFact.Holds environment {
      subject
      primary := .affine { terms := [], noiseBound := bound }
      relations := []
      totalNormBound := bound
    } := by
  refine ⟨value, boundValue, lookup, ?_, by simp, boundEvaluates, normBound, trivial⟩
  exact ⟨[], value, boundValue, .nil, boundEvaluates, normBound, .refl value⟩

theorem identitySignalNoiseMatrixFact_holds
    (environment : FactEnvironment)
    (subject : ValueInstanceRef)
    (identityType : MatrixTypeExpr)
    (signalExpression : MatrixExpr)
    (signal noise : Mxx.Matrix)
    (noiseBound totalBound : BoundExpr)
    (noiseBoundValue totalBoundValue : Nat)
    (lookup : environment.values subject = some (.matrix (Mxx.matrixAdd signal noise)))
    (signalDenotes : MatrixExpr.Denotes environment signalExpression signal)
    (noiseBoundEvaluates : noiseBound.evaluate environment.parameters = .ok noiseBoundValue)
    (noiseNorm : Mxx.maxCenteredCoefficientNorm noise ≤ noiseBoundValue)
    (totalBoundEvaluates : totalBound.evaluate environment.parameters = .ok totalBoundValue)
    (totalNorm :
      Mxx.maxCenteredCoefficientNorm (Mxx.matrixAdd signal noise) ≤ totalBoundValue) :
    MatrixFact.Holds environment {
      subject
      primary := .affine {
        terms := [{
          coefficient := { expression := .identity identityType, normBound := .constant 1 }
          basis := signalExpression
          mode := .ordinaryMatrixProduct
        }]
        noiseBound
      }
      relations := []
      totalNormBound := totalBound
    } := by
  refine ⟨Mxx.matrixAdd signal noise, totalBoundValue, lookup, ?_, by simp,
    totalBoundEvaluates, totalNorm, trivial⟩
  refine ⟨[signal], noise, noiseBoundValue, ?_, noiseBoundEvaluates, noiseNorm, ?_⟩
  exact .cons (.identityCoefficient signalDenotes) .nil
  exact .refl (Mxx.matrixAdd signal noise)

def ScopedWireFact.Holds
    (environment : FactEnvironment)
    (fact : ScopedWireFact) : Prop :=
  match fact.fact with
  | .matrix matrix => matrix.Holds environment
  | .integer integer => ∃ value lower upper,
      environment.values (.ofCoreWire fact.wire) = some (.integer value) ∧
        RuntimeIntExpr.Denotes environment integer.expression value ∧
        integer.lower.evaluate environment.parameters environment.recurrenceBounds = .ok lower ∧
        integer.upper.evaluate environment.parameters environment.recurrenceBounds = .ok upper ∧
        lower ≤ value ∧ value ≤ upper
  | .boolean boolean => ∃ value,
      environment.values (.ofCoreWire fact.wire) = some (.boolean value) ∧
        RuntimeBoolExpr.Denotes environment boolean.expression value
  | .bytes wire => ∃ bytes, environment.values wire = some (.bytes bytes)
  | .trapdoor trapdoor => ∃ publicMatrix,
      environment.values trapdoor.publicPort = some (.matrix publicMatrix)
  | .family _ => ∃ values,
      environment.values (.ofCoreWire fact.wire) = some (.family values)

def AnalysisHolds (environment : FactEnvironment) (analysis : AnalysisResult) : Prop :=
  environment.analysis = some analysis ∧
    environment.expressionArena = analysis.expressionArena ∧
    analysis.expressionArena.WF = true ∧
    ∀ fact ∈ analysis.facts, fact.Holds environment

end Mxx.Certificate
