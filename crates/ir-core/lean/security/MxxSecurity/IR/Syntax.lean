import Mathlib.Data.Rat.Defs

/-!
# The IR as Lean data

A deep embedding of the mxx IR (`mxx_ir_core::Graph`): every operation of `NodeKind` has one
constructor of `Op`, and a `Program` is the list of scopes of a frozen graph. A program is
exported after its compile parameters are bound, so every integer, real, and ring in it is a
constant except the expressions that depend on a loop index, which stay as `IExpr`, `RExpr`, and
`RingE` terms.

The syntax carries no meaning by itself: `MxxSecurity.IR.eval` gives each operation a function as
its semantics and `MxxSecurity.IR.Program.cost` its cost. A statement that quantifies over
programs therefore restricts a computation to the operations of the IR at the cost the table
assigns them.
-/

namespace MxxSecurity.IR

/-- An integer expression: a constant, a loop index, or an operation on them, mirroring
`IntExpr` after its parameters are bound. -/
inductive IExpr where
  | const (value : Int)
  | loopIndex (slot : Nat)
  | add (left right : IExpr)
  | sub (left right : IExpr)
  | mul (left right : IExpr)
  | div (left right : IExpr)
  | floorDiv (left right : IExpr)
  | rem (left right : IExpr)
  | roundDiv (left right : IExpr)
  | log2Ceil (value : IExpr)
  | select (selector : IExpr) (branches : List IExpr)
  deriving Repr, Inhabited

/-- A real expression, mirroring `RealExpr` after its parameters are bound. -/
inductive RExpr where
  | rational (value : Rat)
  | ofInt (value : IExpr)
  | add (left right : RExpr)
  | sub (left right : RExpr)
  | mul (left right : RExpr)
  | div (left right : RExpr)
  | sqrt (value : RExpr)
  deriving Repr, Inhabited

/-- An ordered CRT basis of the negacyclic ring of dimension `dimension`, mirroring `RingExpr`. -/
inductive RingE where
  | explicit (moduli : List IExpr) (dimension : Nat)
  | slice (source : RingE) (start stop : IExpr)
  | select (source : RingE) (indices : List IExpr)
  | concat (left right : RingE)
  deriving Repr, Inhabited

/-- The type of a matrix over a ring. -/
structure MatrixTy where
  ring : RingE
  rows : IExpr
  columns : IExpr
  deriving Repr, Inhabited

/-- How a bound on compact coefficients is read. -/
inductive BoundDomain where
  | global
  | perCrtLimb
  deriving Repr, Inhabited, DecidableEq

/-- The type of a wire, mirroring `WireType`. Compile-time and runtime scalars share a type. -/
inductive Ty where
  | int
  | real
  | bool
  | bytes (length : IExpr)
  | blob (name : String)
  | matrix (matrix : MatrixTy)
  | trapdoor (matrix : MatrixTy) (sigma : RExpr) (gadgetBase digitCount bound : IExpr)
      (statisticalRows : Option Nat) (preimageSigma : Option RExpr)
  | small (matrix : MatrixTy) (bound : IExpr) (domain : BoundDomain)
  | preimage (matrix : MatrixTy) (bound : IExpr) (domain : BoundDomain)
  | family (element : Ty) (count : IExpr)
  deriving Repr, Inhabited

/-- A constant matrix, mirroring `ConstantMatrix`. -/
inductive ConstMatrix where
  | zero
  | identity
  | unitRow (index : IExpr)
  | unitColumn (index : IExpr)
  | gadget (base : IExpr) (small : Bool)
  | powerOfBase (base exponent : IExpr)
  | rotation (exponent : IExpr)
  | polynomial (coefficients : List IExpr)
  /-- `polynomial` with nonnegative coefficients packed `width` bits each into `table`, the
  first `count` of them nonzero: the form an export uses for long coefficient lists. -/
  | packed (width count table : Nat)
  | allCoefficients (value : IExpr)
  deriving Repr, Inhabited

inductive IntOp where
  | add | sub | mul | div | rem
  deriving Repr, Inhabited, DecidableEq

inductive CompareOp where
  | eq | lt | le
  deriving Repr, Inhabited, DecidableEq

inductive RealOp where
  | add | sub | mul | div
  deriving Repr, Inhabited, DecidableEq

inductive MatrixOp where
  | add | sub | mul
  deriving Repr, Inhabited, DecidableEq

inductive Axis where
  | rows | columns | diagonal
  deriving Repr, Inhabited, DecidableEq

inductive HashVariant where
  | plain | decomposed | smallDecomposed
  deriving Repr, Inhabited, DecidableEq

/-- A component of a hash tag, mirroring `HashTagComponent`. -/
inductive TagPart where
  | bytes (value : List UInt8)
  | integer (value : IExpr)
  | decimal (value : IExpr)
  | u64le (value : IExpr)
  | operand (index : Nat)
  /-- A runtime integer argument, encoded like `u64le` of its value. -/
  | operandU64le (index : Nat)
  deriving Repr, Inhabited

/-- How a parallel loop reads an argument. -/
inductive LoopMode where
  | broadcast
  | zip
  | zipOffset (offset : Nat)
  /-- The member whose index is the iteration's value of the integer family argument
  `indexArgument`, itself read by `zip` or `zipOffset`. -/
  | gather (indexArgument : Nat)
  /-- Member `⌊i / divisor⌋` of iteration `i`. -/
  | zipQuotient (divisor : Nat)
  deriving Repr, Inhabited, DecidableEq

/-- One operation, mirroring `NodeKind`. Loops and calls name their body by its scope index. -/
inductive Op where
  | input (index : Nat)
  | constInt (value : Int)
  | evaluateInt (value : IExpr)
  | sha256BitColumns (seed : List UInt8) (rows columns indexBits : Nat)
  | constReal (value : RExpr)
  | constBool (value : Bool)
  | constMatrix (type : MatrixTy) (value : ConstMatrix)
  | gadgetTrapdoor (type : MatrixTy) (base : IExpr)
  | trapdoorPublic
  | intBinary (op : IntOp)
  | intCompare (op : CompareOp)
  | bitExtract (bit : IExpr)
  | intToReal
  | boolToInt
  | realBinary (op : RealOp)
  | realSqrt
  | matrixBinary (op : MatrixOp)
  | matrixMulAccumulate (coefficients : List IExpr) (hasBias : Bool)
  | matrixMulSmallRhs
  | matrixNegate
  | matrixScale (scalar : IExpr)
  | ringAutomorphism (index : IExpr)
  | multiplyMonomial
  | modulusSwitch (destination : RingE)
  | modulusReduce (destination : RingE)
  | centeredRebase (destination : RingE)
  | centeredRoundDivide (divisor : IExpr)
  /-- The digits of a CRT residue: `source` is the ordered basis of the argument, which fixes the
  grouping of its limbs into digits. -/
  | rnsModUp (source destination : RingE) (digitSize : Nat) (normalize : Bool)
  | rnsModDown (destination : RingE) (plaintextModulus : IExpr)
  | blockModSwitch (destination : RingE) (plaintextModulus : IExpr)
  | transpose
  | slice (rows columns : Option (IExpr × IExpr))
  | tensor
  | blockDiagonal (count : Nat)
  | concat (axis : Axis)
  | uniformResidueSample (type : MatrixTy)
  | uniformIntervalSample (type : MatrixTy) (minimum maximum : IExpr)
  | gaussianSample (type : MatrixTy) (sigma : RExpr) (bound : IExpr) (compact : Bool)
  | hashSample (type : MatrixTy) (variant : HashVariant) (tagPrefix : List UInt8)
      (tag : List TagPart) (base digitCount : Option IExpr)
  | hashIntFamily (count modulus : IExpr) (tagPrefix : List UInt8) (tag : List TagPart)
  | trapdoorSample (type : MatrixTy) (sigma : RExpr) (gadgetBase digitCount bound : IExpr)
      (statisticalRows : Option Nat) (preimageSigma : Option RExpr)
  | preimageSample (type : MatrixTy) (bound : IExpr)
  | gadgetDecompose (base : IExpr) (small : Bool) (digitCount : IExpr)
  | extractCoefficient (position : IExpr)
  | liftIntegerToConstantPolynomial (type : MatrixTy)
  | thresholdDecode (plaintextModulus length : IExpr) (outputBool family : Bool)
  | crtRecompose (modulus : IExpr) (plaintextModuli coefficients : List IExpr)
  | packPolynomialCoefficients (type : MatrixTy) (coefficientBits : IExpr)
  | polynomialFromValues (type : MatrixTy) (evaluation : Bool)
  | polynomialValues (evaluation : Bool)
  | intMatrixVectorProduct (transpose : Bool)
  | call (body : Nat)
  | parallelLoop (body : Nat) (count : IExpr) (slot : Nat) (modes : List LoopMode)
  | sequentialLoop (body : Nat) (count : IExpr) (slot : Nat) (carried : Nat)
  | familyPack (count : IExpr)
  | familyGetStatic (index : IExpr)
  | familyGetDynamic
  | select (count : IExpr)
  deriving Repr, Inhabited

/-- An output port of an earlier node of the same scope. -/
structure Ref where
  node : Nat
  port : Nat
  deriving Repr, Inhabited, DecidableEq

/-- A node: its operation, arguments, and output types. -/
structure Node where
  op : Op
  args : List Ref
  outputs : List Ty
  deriving Repr, Inhabited

/-- A scope: its nodes in an order where every argument refers to an earlier node, and its
outputs. Its inputs are its `Op.input` nodes. -/
structure Scope where
  nodes : List Node
  outputs : List Ref
  deriving Repr, Inhabited

/-- A frozen graph: its scopes, the first being the root, and the names of the root's inputs and
outputs in order. -/
structure Program where
  scopes : List Scope
  inputs : List String
  outputs : List String
  deriving Repr, Inhabited

end MxxSecurity.IR
