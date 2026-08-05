import Mxx.Certificate.Syntax

namespace Mxx.Certificate

def InstanceFrame.refsBefore (limit : Nat) : InstanceFrame → Bool
  | .subgraphCall _ => true
  | .parallelLane _ index | .sequentialIteration _ index => index.id < limit

def ValueInstanceRef.refsBefore (limit : Nat) : ValueInstanceRef → Bool
  | .protocolInput _ | .concrete _ | .template _ | .recurrenceResult _ _ => true
  | .instantiatedTemplate _ path => path.all (InstanceFrame.refsBefore limit)
  | .familyElement _ index => index.id < limit

def MatrixInstanceRef.refsBefore (limit : Nat) (reference : MatrixInstanceRef) : Bool :=
  reference.value.refsBefore limit

def RuntimeExpr.refsBefore (limit : Nat) : {type : RuntimeScalarType} → RuntimeExpr type → Bool
  | _, .intWire wire | _, .boolWire wire => wire.refsBefore limit
  | _, .intConstant _ | _, .boolConstant _ | _, .parameter _ | _, .loopIndex _ |
      _, .carriedInput _ => true
  | _, .intBinary _ left right | _, .compare _ left right =>
      left.refsBefore limit && right.refsBefore limit
  | _, .bitExtract value _ | _, .boolToInt value => value.refsBefore limit
  | _, .thresholdDecodeBool matrix _ _ _ => matrix.refsBefore limit
  | _, .extractCoefficient matrix _ => matrix.id < limit
  | _, .familyElement _ _ indexRef index => indexRef.id < limit && index.refsBefore limit
  | _, .select _ index branches =>
      index.refsBefore limit && branches.all (fun branch => branch.id < limit)

def MatrixExpr.refsBefore (limit : Nat) (expression : MatrixExpr) : Bool :=
  match expression with
  | .wire reference => reference.refsBefore limit
  | .zero _ | .identity _ | .gadget _ _ | .loopResult _ _ _ | .carriedInput _ _ => true
  | .add left right | .multiply left right => left.refsBefore limit && right.refsBefore limit
  | .negate value | .scalarMultiply _ value | .rowSlice value _ _ |
      .columnSlice value _ _ | .rowCoefficientEmbed _ _ value |
      .columnBasisEmbed _ _ value | .diagonalCoefficientEmbed _ _ value |
      .diagonalBasisEmbed _ _ value => value.refsBefore limit
  | .rowConcat parts | .columnConcat parts | .diagonalConcat parts =>
      parts.isEmpty
  | .select _ _ => false

inductive SymbolicExprEntry where
  | integer (expression : RuntimeExpr .integer)
  | boolean (expression : RuntimeExpr .boolean)
  | matrix (expression : MatrixExpr)

abbrev SymbolicExprTable := List SymbolicExprEntry

def SymbolicExprEntry.refsBefore (limit : Nat) : SymbolicExprEntry → Bool
  | .integer expression => expression.refsBefore limit
  | .boolean expression => expression.refsBefore limit
  | .matrix expression => expression.refsBefore limit

/-- Analyzer-owned immutable arena.  An entry may only refer to older entries, making lookup and
recursive substitution well-founded without accepting any expression supplied by a certificate. -/
structure ExpressionArena where
  entries : SymbolicExprTable := []

def ExpressionArena.WF (arena : ExpressionArena) : Bool :=
  arena.entries.zipIdx.all fun (entry, index) => entry.refsBefore index

def ExpressionArena.internInteger
    (arena : ExpressionArena)
    (expression : RuntimeExpr .integer) : Option (ExpressionArena × RuntimeExprRef .integer) :=
  if expression.refsBefore arena.entries.length then
    some (⟨arena.entries ++ [.integer expression]⟩, ⟨arena.entries.length⟩)
  else none

def ExpressionArena.internBoolean
    (arena : ExpressionArena)
    (expression : RuntimeExpr .boolean) : Option (ExpressionArena × RuntimeExprRef .boolean) :=
  if expression.refsBefore arena.entries.length then
    some (⟨arena.entries ++ [.boolean expression]⟩, ⟨arena.entries.length⟩)
  else none

def ExpressionArena.internMatrix
    (arena : ExpressionArena)
    (expression : MatrixExpr) : Option (ExpressionArena × MatrixExprRef) :=
  if expression.refsBefore arena.entries.length then
    some (⟨arena.entries ++ [.matrix expression]⟩, ⟨arena.entries.length⟩)
  else none

def ExpressionArena.lookupInteger
    (arena : ExpressionArena) (reference : RuntimeExprRef .integer) : Option (RuntimeExpr .integer) :=
  match arena.entries[reference.id]? with
  | some (.integer expression) => some expression
  | _ => none

def ExpressionArena.lookupBoolean
    (arena : ExpressionArena) (reference : RuntimeExprRef .boolean) : Option (RuntimeExpr .boolean) :=
  match arena.entries[reference.id]? with
  | some (.boolean expression) => some expression
  | _ => none

def ExpressionArena.lookupMatrix
    (arena : ExpressionArena) (reference : MatrixExprRef) : Option MatrixExpr :=
  match arena.entries[reference.id]? with
  | some (.matrix expression) => some expression
  | _ => none

example :
    (ExpressionArena.internInteger { entries := [] } (.intConstant 3)).map
      (fun result => result.2.id) = some 0 :=
  rfl

example : ExpressionArena.WF { entries := [] } = true := rfl

end Mxx.Certificate
