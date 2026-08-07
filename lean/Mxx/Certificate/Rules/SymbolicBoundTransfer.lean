import Mxx.Certificate.SymbolicForm

namespace Mxx.Certificate

/-!
# Closed hard-bound transfers for symbolic matrix summaries

These constructors encode the deterministic transfer table.  The analyzer may select an
applicable constructor after type checking, but it cannot supply a bound formula.  The
Dimension arguments are the effective contraction dimensions established by the executable
matrix-product typing rule (including its broadcast convention).  The right signal coefficient
product is separate because its coefficient matrix need not have the value matrix's shape.
-/

namespace SymbolicBoundTransfer

/-- `floor (|q| / 2)`, the maximum centered representative norm. -/
def centeredRepresentativeBound (modulus : IntExpr) : BoundExpr :=
  .floorDivide (.absolute modulus) 2

/-- Cap a stored-value bound by the centered representative range. -/
def centeredCap (modulus : IntExpr) (bound : BoundExpr) : BoundExpr :=
  .minimum (centeredRepresentativeBound modulus) bound

/-- Negation preserves every hard-bound role. -/
def negate (input : MatrixBoundSummary) : MatrixBoundSummary := input

/-- Addition combines role bounds additively and caps the stored representative at `q / 2`.
The cap never replaces the more precise noise bound. -/
def add
    (modulus : IntExpr)
    (left right : MatrixBoundSummary) : MatrixBoundSummary :=
  {
    signal := left.signal.combine right.signal
    coefficientL1Bound := .add left.coefficientL1Bound right.coefficientL1Bound
    noiseBound := .add left.noiseBound right.noiseBound
    totalBound := centeredCap modulus (.add left.totalBound right.totalBound)
  }

inductive MultiplyError where
  | typing (error : TypingError)
  | signalTimesSignal

private def effectiveInnerDimension
    (leftType : MatrixTypeExpr)
    (product : MatrixProductType) : IntExpr :=
  match product.mode with
  | .ordinaryMatrixProduct => leftType.columns
  | .leftPolynomialScalarBroadcast | .rightPolynomialScalarBroadcast |
      .swappedRowVectorScalarProduct => .constant 1

private def checkedProductBound
    (leftType rightType : MatrixTypeExpr)
    (leftBound rightBound : BoundExpr) :
    Except MultiplyError (MatrixProductType × BoundExpr) := do
  let product ← inferMatrixProductType leftType rightType |>.mapError .typing
  return (product, .matrixProduct product.output.ringDimension
    (effectiveInnerDimension leftType product) leftBound rightBound)

/-- Closed initial multiplication table.

* bounded × bounded: the complete values are noise;
* signal × bounded: only the left signal coefficients survive;
* bounded × signal: the bounded left value becomes part of every signal coefficient;
* signal × signal: rejected until a proved protocol use requires a rule.
-/
def multiply
    (leftType rightType rightSignalCoefficientType : MatrixTypeExpr)
    (left right : MatrixBoundSummary) : Except MultiplyError MatrixBoundSummary :=
  do
  let valueProduct ← inferMatrixProductType leftType rightType |>.mapError .typing
  let centeredBound := centeredRepresentativeBound valueProduct.output.modulus
  match left.signal, right.signal with
  | .none, .none =>
      let (_, noiseBound) ← checkedProductBound leftType rightType
        left.noiseBound right.noiseBound
      let (_, totalBound) ← checkedProductBound leftType rightType
        left.totalBound right.totalBound
      .ok {
        signal := .none
        coefficientL1Bound := .constant 0
        noiseBound
        totalBound
      }
  | .present, .none =>
      let (_, noiseBound) ← checkedProductBound leftType rightType
        left.noiseBound right.totalBound
      .ok {
        signal := .present
        coefficientL1Bound := left.coefficientL1Bound
        noiseBound
        totalBound := centeredBound
      }
  | .none, .present =>
      let (_, coefficientL1Bound) ← checkedProductBound leftType rightSignalCoefficientType
        left.totalBound right.coefficientL1Bound
      let (_, noiseBound) ← checkedProductBound leftType rightType
        left.totalBound right.noiseBound
      .ok {
        signal := .present
        coefficientL1Bound
        noiseBound
        totalBound := centeredBound
      }
  | .present, .present => .error .signalTimesSignal

/-! Structural golden examples.  These intentionally compare `BoundExpr` syntax, not only its
numeric evaluation, so changing a deterministic formula is visible. -/

private abbrev q : IntExpr := .parameter "q"
private abbrev n : IntExpr := .parameter "n"
private abbrev k : IntExpr := .parameter "k"

private def leftType : MatrixTypeExpr where
  modulus := q
  ringDimension := n
  rows := .constant 2
  columns := k

private def rightType : MatrixTypeExpr where
  modulus := q
  ringDimension := n
  rows := k
  columns := .constant 4

/-- A right signal coefficient may be a polynomial scalar even though the represented right
matrix is not; its product therefore has effective inner dimension one. -/
private def rightSignalCoefficientType : MatrixTypeExpr where
  modulus := q
  ringDimension := n
  rows := .constant 1
  columns := .constant 1

private def boundedLeft : MatrixBoundSummary := .bounded (.parameter (.parameter "leftBound"))
private def boundedRight : MatrixBoundSummary := .bounded (.parameter (.parameter "rightBound"))
private def signalLeft : MatrixBoundSummary where
  signal := .present
  coefficientL1Bound := .parameter (.parameter "leftCoefficient")
  noiseBound := .parameter (.parameter "leftNoise")
  totalBound := centeredRepresentativeBound q

private def signalRight : MatrixBoundSummary where
  signal := .present
  coefficientL1Bound := .parameter (.parameter "rightCoefficient")
  noiseBound := .parameter (.parameter "rightNoise")
  totalBound := centeredRepresentativeBound q

example : negate signalLeft = signalLeft := rfl

example : add q signalLeft boundedRight = {
    signal := .present
    coefficientL1Bound := .add signalLeft.coefficientL1Bound
      boundedRight.coefficientL1Bound
    noiseBound := .add signalLeft.noiseBound boundedRight.noiseBound
    totalBound := .minimum (.floorDivide (.absolute q) 2)
      (.add signalLeft.totalBound boundedRight.totalBound)
  } := rfl

example : multiply leftType rightType rightSignalCoefficientType boundedLeft boundedRight = .ok {
    signal := .none
    coefficientL1Bound := .constant 0
    noiseBound := .matrixProduct n k boundedLeft.noiseBound boundedRight.noiseBound
    totalBound := .matrixProduct n k boundedLeft.totalBound boundedRight.totalBound
  } := rfl

example : multiply leftType rightType rightSignalCoefficientType signalLeft boundedRight = .ok {
    signal := .present
    coefficientL1Bound := signalLeft.coefficientL1Bound
    noiseBound := .matrixProduct n k signalLeft.noiseBound boundedRight.totalBound
    totalBound := .floorDivide (.absolute q) 2
  } := rfl

example : multiply leftType rightType rightSignalCoefficientType boundedLeft signalRight = .ok {
    signal := .present
    coefficientL1Bound := .matrixProduct n (.constant 1) boundedLeft.totalBound
      signalRight.coefficientL1Bound
    noiseBound := .matrixProduct n k boundedLeft.totalBound signalRight.noiseBound
    totalBound := .floorDivide (.absolute q) 2
  } := rfl

example : multiply leftType rightType rightSignalCoefficientType signalLeft signalRight =
    .error .signalTimesSignal := rfl

private def incompatibleRightType : MatrixTypeExpr :=
  { rightType with modulus := .constant 19 }

example : multiply leftType incompatibleRightType rightSignalCoefficientType
    boundedLeft boundedRight =
    .error (.typing (.incompatibleMatrixProduct leftType incompatibleRightType)) := rfl

end SymbolicBoundTransfer

end Mxx.Certificate
