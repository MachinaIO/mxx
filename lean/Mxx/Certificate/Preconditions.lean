import Mxx.Certificate.Bounds
import Mxx.Certificate.Execution
import Mxx.Certificate.ValueSemantics

namespace Mxx.Certificate

/-- A concrete parameter value satisfies its declaration. Dimensions are nonnegative integers;
ordinary integers and rationals retain their exact IR representation. -/
def ParameterDecl.ValueWF (declaration : ParameterDecl) : Mxx.Ir.ParamValue → Prop
  | .integer value =>
      match declaration.kind with
      | .dimension => 0 ≤ value
      | .integer => True
      | .rational => False
  | .rational _ => declaration.kind = .rational

/-- Parameter well-formedness is derived solely from the closed declaration and the concrete IR
parameter environment. A caller cannot substitute an unrelated predicate. -/
def ClosedProtocolDecl.ParamsWF
    (protocol : ClosedProtocolDecl)
    (parameters : Mxx.Ir.ParamEnvironment) : Prop :=
  ∀ declaration ∈ protocol.parameters,
    ∃ value, Mxx.Ir.lookupParam declaration.name parameters = some value ∧
      declaration.ValueWF value

/-- Every coefficient above degree zero in every matrix entry is zero in the coefficient ring. -/
def MatrixIsConstantPolynomial (matrix : Mxx.Matrix) : Prop :=
  ∀ row column coefficient,
    row < matrix.rows → column < matrix.columns → coefficient < matrix.ringDimension →
      coefficient ≠ 0 →
      Mxx.reduceCoefficient matrix.modulus
        (matrix.coefficient row column coefficient) = 0

/-- Runtime meaning of an input contract. Every numerical expression is evaluated in the same
parameter environment used by protocol execution. -/
def InputValueContract.Holds
    (parameters : Mxx.Ir.ParamEnvironment) : InputValueContract → Mxx.Ir.Value → Prop
  | .matrixExact matrixType canonicalExclusiveUpper isConstantPolynomial, .matrix matrix =>
      matrixType.Holds parameters matrix ∧
        (match canonicalExclusiveUpper with
        | none => True
        | some upper =>
            ∃ evaluatedUpper,
              evaluateIntExpr parameters upper = .ok evaluatedUpper ∧
              0 < evaluatedUpper ∧
              Mxx.maxCanonicalCoefficient matrix < evaluatedUpper.toNat) ∧
        (isConstantPolynomial = true → MatrixIsConstantPolynomial matrix)
  | .matrixBounded matrixType declaredBound, .matrix matrix =>
      matrixType.Holds parameters matrix ∧
        ∃ bound, declaredBound.toBoundExpr.evaluate parameters = .ok bound ∧
          Mxx.maxCenteredCoefficientNorm matrix ≤ bound
  | .integerRange lower upper, .integer value =>
      ∃ lowerValue upperValue,
        evaluateIntExpr parameters lower = .ok lowerValue ∧
        evaluateIntExpr parameters upper = .ok upperValue ∧
        lowerValue ≤ value ∧ value ≤ upperValue
  | .boolean, .boolean _ => True
  | .bytes length, .bytes value =>
      ∃ expected,
        evaluateIntExpr parameters length = .ok expected ∧
        0 ≤ expected ∧ value.size = expected.toNat
  | .family count element, .family values =>
      ∃ expected,
        evaluateIntExpr parameters count = .ok expected ∧
        0 ≤ expected ∧ values.length = expected.toNat ∧
        ∀ value ∈ values, element.Holds parameters value
  | _, _ => False

private def constantPolynomialContractFixtureType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 2
  rows := .constant 1
  columns := .constant 1

private def constantPolynomialContractFixtureMatrix : Mxx.Matrix where
  coefficients := [3, 0]
  modulus := 17
  ringDimension := 2
  rows := 1
  columns := 1

private def nonconstantPolynomialContractFixtureMatrix : Mxx.Matrix where
  coefficients := [3, 1]
  modulus := 17
  ringDimension := 2
  rows := 1
  columns := 1

example :
    (InputValueContract.matrixExact constantPolynomialContractFixtureType
      (some (.constant 4)) true).Holds [] (.matrix constantPolynomialContractFixtureMatrix) := by
  constructor
  · refine ⟨{
      maxCoefficientBound := 0
      modulus := 17
      ringDimension := 2
      rows := 1
      columns := 1
    }, ?_⟩
    norm_num [constantPolynomialContractFixtureType, constantPolynomialContractFixtureMatrix,
      Mxx.Ir.MatrixTypeExpr.evaluate, Mxx.Ir.IntExpr.evaluate, Mxx.Matrix.WellFormed]
    decide
  constructor
  · norm_num [evaluateIntExpr, Mxx.Ir.IntExpr.evaluate, Mxx.maxCanonicalCoefficient,
      constantPolynomialContractFixtureMatrix, Mxx.canonicalCoefficient,
      Mxx.reduceCoefficient]
  · intro _ row column coefficient rowBound columnBound coefficientBound coefficientNonzero
    norm_num [constantPolynomialContractFixtureMatrix] at rowBound columnBound coefficientBound
    have rowZero : row = 0 := by omega
    have columnZero : column = 0 := by omega
    have coefficientOne : coefficient = 1 := by omega
    subst row
    subst column
    subst coefficient
    decide

example :
    ¬(InputValueContract.matrixExact constantPolynomialContractFixtureType
      (some (.constant 3)) true).Holds [] (.matrix constantPolynomialContractFixtureMatrix) := by
  norm_num [InputValueContract.Holds, MatrixTypeExpr.Holds,
    constantPolynomialContractFixtureType, constantPolynomialContractFixtureMatrix,
    evaluateIntExpr, Mxx.Ir.IntExpr.evaluate, Mxx.maxCanonicalCoefficient,
    Mxx.canonicalCoefficient, Mxx.reduceCoefficient, Mxx.Matrix.WellFormed]

example :
    ¬(InputValueContract.matrixExact constantPolynomialContractFixtureType
      (some (.constant 4)) true).Holds [] (.matrix nonconstantPolynomialContractFixtureMatrix) := by
  intro holds
  have constant := holds.2.2 (by decide)
  have degreeOneZero := constant 0 0 1 (by decide) (by decide) (by decide) (by decide)
  norm_num [nonconstantPolynomialContractFixtureMatrix, Mxx.Matrix.coefficient,
    Mxx.reduceCoefficient] at degreeOneZero

/-- Input well-formedness is the direct interpretation of every named contract entry against the
concrete protocol input environment. -/
def ClosedProtocolDecl.InputsWF
    (protocol : ClosedProtocolDecl)
    (parameters : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment) : Prop :=
  ∀ entry ∈ protocol.bundle.inputContract.inputs,
    ∃ value, Mxx.Ir.lookupEnvironment entry.2.1 inputs = some value ∧
      entry.2.2.Holds parameters value

/-- One requirement succeeds exactly when the existing pure IR denotation terminates and its
declared output is the boolean value `true`. -/
def requirementHolds
    (bundle : ClosedProtocolBundle)
    (parameters : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (index : Nat)
    (program : Mxx.Ir.Prog)
    (outputName : String) : Prop :=
  ∃ output,
    Mxx.Ir.denotePure program parameters
        (requirementInputEnvironment bundle inputs index program) = some output ∧
      Mxx.Ir.lookupEnvironment outputName output = some (.boolean true)

/-- Protocol preconditions are not certificate or caller assumptions. They are precisely the
boolean results of the closed bundle's requirement programs under their declared bindings. -/
def ClosedProtocolDecl.ProtocolPreconditions
    (protocol : ClosedProtocolDecl)
    (parameters : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment) : Prop :=
  ∀ index program outputName,
    protocol.bundle.requirements[index]? = some program →
    protocol.bundle.preconditionSpec.requirementOutputs[index]? = some outputName →
    requirementHolds protocol.bundle parameters inputs index program outputName

end Mxx.Certificate
