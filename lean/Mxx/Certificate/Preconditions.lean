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

/-- Runtime meaning of an input contract. Every numerical expression is evaluated in the same
parameter environment used by protocol execution. -/
def InputValueContract.Holds
    (parameters : Mxx.Ir.ParamEnvironment) : InputValueContract → Mxx.Ir.Value → Prop
  | .matrixExact matrixType, .matrix matrix => matrixType.Holds parameters matrix
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
