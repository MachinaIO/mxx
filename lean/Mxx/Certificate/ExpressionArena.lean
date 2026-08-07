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

mutual

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
      MatrixExprList.refsBefore limit parts
  | .select index branches =>
      index.refsBefore limit && MatrixExprList.refsBefore limit branches

def MatrixExprList.refsBefore (limit : Nat) : List MatrixExpr → Bool
  | [] => true
  | expression :: tail => expression.refsBefore limit && MatrixExprList.refsBefore limit tail

end

inductive SymbolicExprEntry where
  | integer (expression : RuntimeExpr .integer)
  | boolean (expression : RuntimeExpr .boolean)
  | matrix (expression : MatrixExpr)

abbrev SymbolicExprTable := Array SymbolicExprEntry

def SymbolicExprEntry.refsBefore (limit : Nat) : SymbolicExprEntry → Bool
  | .integer expression => expression.refsBefore limit
  | .boolean expression => expression.refsBefore limit
  | .matrix expression => expression.refsBefore limit

/-- Analyzer-owned immutable arena.  An entry may only refer to older entries, making lookup and
recursive substitution well-founded without accepting any expression supplied by a certificate. -/
structure ExpressionArena where
  entries : SymbolicExprTable := #[]

/-- Append-only extension relation used to transport successful references across later analyzer
construction steps. -/
def ExpressionArena.Extends (older newer : ExpressionArena) : Prop :=
  ∃ suffix, newer.entries = older.entries ++ suffix

def ExpressionArena.WF (arena : ExpressionArena) : Bool :=
  arena.entries.toList.zipIdx.all fun (entry, index) => entry.refsBefore index

def ExpressionArena.internInteger
    (arena : ExpressionArena)
    (expression : RuntimeExpr .integer) : Option (ExpressionArena × RuntimeExprRef .integer) :=
  if expression.refsBefore arena.entries.size then
    some (⟨arena.entries.push (.integer expression)⟩, ⟨arena.entries.size⟩)
  else none

def ExpressionArena.internBoolean
    (arena : ExpressionArena)
    (expression : RuntimeExpr .boolean) : Option (ExpressionArena × RuntimeExprRef .boolean) :=
  if expression.refsBefore arena.entries.size then
    some (⟨arena.entries.push (.boolean expression)⟩, ⟨arena.entries.size⟩)
  else none

def ExpressionArena.internMatrix
    (arena : ExpressionArena)
    (expression : MatrixExpr) : Option (ExpressionArena × MatrixExprRef) :=
  if expression.refsBefore arena.entries.size then
    some (⟨arena.entries.push (.matrix expression)⟩, ⟨arena.entries.size⟩)
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

/-! ## Structural identity of arena-backed scalar references -/

/- Parallel template instantiation can reconstruct one frozen dynamic index in a later arena
slot.  This comparison follows only the analyzer-owned expression syntax and frozen value origins;
it never compares evaluated integers or treats numerically equal expressions as interchangeable. -/
namespace ExpressionArena

mutual

private def sameIntegerReferenceWithFuel
    (arena : ExpressionArena) : Nat → RuntimeExprRef .integer → RuntimeExprRef .integer → Bool
  | 0, _, _ => false
  | fuel + 1, left, right =>
      if left == right then true else
        match arena.lookupInteger left, arena.lookupInteger right with
        | some leftExpression, some rightExpression =>
            arena.sameIntegerExpressionWithFuel fuel leftExpression rightExpression
        | _, _ => false

private def sameBooleanReferenceWithFuel
    (arena : ExpressionArena) : Nat → RuntimeExprRef .boolean → RuntimeExprRef .boolean → Bool
  | 0, _, _ => false
  | fuel + 1, left, right =>
      if left == right then true else
        match arena.lookupBoolean left, arena.lookupBoolean right with
        | some leftExpression, some rightExpression =>
            arena.sameBooleanExpressionWithFuel fuel leftExpression rightExpression
        | _, _ => false

private def sameIntegerReferenceListWithFuel
    (arena : ExpressionArena) : Nat → List (RuntimeExprRef .integer) → List (RuntimeExprRef .integer) → Bool
  | 0, _, _ => false
  | _ + 1, [], [] => true
  | fuel + 1, left :: leftTail, right :: rightTail =>
      arena.sameIntegerReferenceWithFuel fuel left right &&
        arena.sameIntegerReferenceListWithFuel fuel leftTail rightTail
  | _, _, _ => false

private def sameBooleanReferenceListWithFuel
    (arena : ExpressionArena) : Nat → List (RuntimeExprRef .boolean) → List (RuntimeExprRef .boolean) → Bool
  | 0, _, _ => false
  | _ + 1, [], [] => true
  | fuel + 1, left :: leftTail, right :: rightTail =>
      arena.sameBooleanReferenceWithFuel fuel left right &&
        arena.sameBooleanReferenceListWithFuel fuel leftTail rightTail
  | _, _, _ => false

private def sameInstancePathWithFuel
    (arena : ExpressionArena) : Nat → InstancePathExpr → InstancePathExpr → Bool
  | 0, _, _ => false
  | _ + 1, [], [] => true
  | fuel + 1, .subgraphCall left :: leftTail, .subgraphCall right :: rightTail =>
      left == right && arena.sameInstancePathWithFuel fuel leftTail rightTail
  | fuel + 1, .parallelLane leftSite leftIndex :: leftTail,
      .parallelLane rightSite rightIndex :: rightTail =>
      leftSite == rightSite && arena.sameIntegerReferenceWithFuel fuel leftIndex rightIndex &&
        arena.sameInstancePathWithFuel fuel leftTail rightTail
  | fuel + 1, .sequentialIteration leftSite leftIndex :: leftTail,
      .sequentialIteration rightSite rightIndex :: rightTail =>
      leftSite == rightSite && arena.sameIntegerReferenceWithFuel fuel leftIndex rightIndex &&
        arena.sameInstancePathWithFuel fuel leftTail rightTail
  | _, _, _ => false

private def sameAggregateWithFuel
    (arena : ExpressionArena) : Nat → FamilyAggregateRef → FamilyAggregateRef → Bool
  | 0, _, _ => false
  | fuel + 1, .joint leftFamily leftSlot leftPath, .joint rightFamily rightSlot rightPath =>
      leftFamily == rightFamily && leftSlot == rightSlot &&
        arena.sameInstancePathWithFuel fuel leftPath rightPath
  | _ + 1, .carriedInput left, .carriedInput right => left == right
  | fuel + 1, .recurrenceResult leftRecurrence leftPath leftSlot,
      .recurrenceResult rightRecurrence rightPath rightSlot =>
      leftRecurrence == rightRecurrence && leftSlot == rightSlot &&
        arena.sameInstancePathWithFuel fuel leftPath rightPath
  | fuel + 1, .familyElement leftParent leftIndex, .familyElement rightParent rightIndex =>
      arena.sameAggregateWithFuel fuel leftParent rightParent &&
        arena.sameIntegerReferenceWithFuel fuel leftIndex rightIndex
  | _, _, _ => false

private def sameValueWithFuel
    (arena : ExpressionArena) : Nat → ValueInstanceRef → ValueInstanceRef → Bool
  | 0, _, _ => false
  | _ + 1, .protocolInput left, .protocolInput right => left == right
  | _ + 1, .concrete left, .concrete right => left == right
  | _ + 1, .template left, .template right => left == right
  | fuel + 1, .instantiatedTemplate leftWire leftPath, .instantiatedTemplate rightWire rightPath =>
      leftWire == rightWire && arena.sameInstancePathWithFuel fuel leftPath rightPath
  | fuel + 1, .familyElement leftAggregate leftIndex, .familyElement rightAggregate rightIndex =>
      arena.sameAggregateWithFuel fuel leftAggregate rightAggregate &&
        arena.sameIntegerReferenceWithFuel fuel leftIndex rightIndex
  | _ + 1, .recurrenceResult leftRecurrence leftSlot, .recurrenceResult rightRecurrence rightSlot =>
      leftRecurrence == rightRecurrence && leftSlot == rightSlot
  | _, _, _ => false

private def sameIntegerExpressionWithFuel
    (arena : ExpressionArena) : Nat → RuntimeExpr .integer → RuntimeExpr .integer → Bool
  | 0, _, _ => false
  | fuel + 1, .intWire left, .intWire right => arena.sameValueWithFuel fuel left right
  | _ + 1, .intConstant left, .intConstant right => left == right
  | _ + 1, .parameter left, .parameter right => left == right
  | fuel + 1, .intBinary leftOperation leftLeft leftRight,
      .intBinary rightOperation rightLeft rightRight =>
      leftOperation == rightOperation &&
        arena.sameIntegerExpressionWithFuel fuel leftLeft rightLeft &&
        arena.sameIntegerExpressionWithFuel fuel leftRight rightRight
  | fuel + 1, .boolToInt left, .boolToInt right =>
      arena.sameBooleanExpressionWithFuel fuel left right
  | _ + 1, .extractCoefficient leftMatrix leftPosition, .extractCoefficient rightMatrix rightPosition =>
      leftMatrix == rightMatrix && leftPosition == rightPosition
  | fuel + 1, .familyElement _ leftAggregate leftIndexReference leftIndex,
      .familyElement _ rightAggregate rightIndexReference rightIndex =>
      arena.sameAggregateWithFuel fuel leftAggregate rightAggregate &&
        arena.sameIntegerReferenceWithFuel fuel leftIndexReference rightIndexReference &&
        arena.sameIntegerExpressionWithFuel fuel leftIndex rightIndex
  | fuel + 1, .select _ leftIndex leftBranches, .select _ rightIndex rightBranches =>
      arena.sameIntegerExpressionWithFuel fuel leftIndex rightIndex &&
        arena.sameIntegerReferenceListWithFuel fuel leftBranches rightBranches
  | _ + 1, .loopIndex left, .loopIndex right => left == right
  | _ + 1, .carriedInput left, .carriedInput right => left == right
  | _, _, _ => false

private def sameBooleanExpressionWithFuel
    (arena : ExpressionArena) : Nat → RuntimeExpr .boolean → RuntimeExpr .boolean → Bool
  | 0, _, _ => false
  | fuel + 1, .boolWire left, .boolWire right => arena.sameValueWithFuel fuel left right
  | _ + 1, .boolConstant left, .boolConstant right => left == right
  | fuel + 1, .compare leftOperation leftLeft leftRight,
      .compare rightOperation rightLeft rightRight =>
      leftOperation == rightOperation &&
        arena.sameIntegerExpressionWithFuel fuel leftLeft rightLeft &&
        arena.sameIntegerExpressionWithFuel fuel leftRight rightRight
  | fuel + 1, .bitExtract leftValue leftPosition, .bitExtract rightValue rightPosition =>
      leftPosition == rightPosition && arena.sameIntegerExpressionWithFuel fuel leftValue rightValue
  | fuel + 1, .thresholdDecodeBool leftMatrix leftQ leftP leftPosition,
      .thresholdDecodeBool rightMatrix rightQ rightP rightPosition =>
      arena.sameValueWithFuel fuel leftMatrix rightMatrix && leftQ == rightQ && leftP == rightP &&
        leftPosition == rightPosition
  | fuel + 1, .familyElement _ leftAggregate leftIndexReference leftIndex,
      .familyElement _ rightAggregate rightIndexReference rightIndex =>
      arena.sameAggregateWithFuel fuel leftAggregate rightAggregate &&
        arena.sameIntegerReferenceWithFuel fuel leftIndexReference rightIndexReference &&
        arena.sameIntegerExpressionWithFuel fuel leftIndex rightIndex
  | fuel + 1, .select _ leftIndex leftBranches, .select _ rightIndex rightBranches =>
      arena.sameIntegerExpressionWithFuel fuel leftIndex rightIndex &&
        arena.sameBooleanReferenceListWithFuel fuel leftBranches rightBranches
  | _, _, _ => false

end

def sameIntegerReference
    (arena : ExpressionArena) (left right : RuntimeExprRef .integer) : Bool :=
  arena.sameIntegerReferenceWithFuel (arena.entries.size * 2 + 16) left right

def sameValue
    (arena : ExpressionArena) (left right : ValueInstanceRef) : Bool :=
  arena.sameValueWithFuel (arena.entries.size * 2 + 16) left right

end ExpressionArena

/-- The reference returned by matrix interning resolves to the appended expression. -/
theorem ExpressionArena.lookupMatrix_internMatrix_eq
    (arena next : ExpressionArena) (expression : MatrixExpr) (reference : MatrixExprRef)
    (interned : arena.internMatrix expression = some (next, reference)) :
    next.lookupMatrix reference = some expression := by
  simp [ExpressionArena.internMatrix] at interned
  rcases interned with ⟨_, rfl, rfl⟩
  simp [ExpressionArena.lookupMatrix]

/-- Matrix interning preserves every previously successful matrix lookup. -/
theorem ExpressionArena.lookupMatrix_internMatrix_preserved
    (arena next : ExpressionArena) (expression : MatrixExpr) (reference old : MatrixExprRef)
    (oldExpression : MatrixExpr)
    (oldLookup : arena.lookupMatrix old = some oldExpression)
    (interned : arena.internMatrix expression = some (next, reference)) :
    next.lookupMatrix old = arena.lookupMatrix old := by
  simp [ExpressionArena.internMatrix] at interned
  rcases interned with ⟨_, rfl, rfl⟩
  have oldInBounds : old.id < arena.entries.size := by
    unfold ExpressionArena.lookupMatrix at oldLookup
    cases lookup : arena.entries[old.id]? with
    | none => simp [lookup] at oldLookup
    | some entry => exact (Array.getElem?_eq_some_iff.mp lookup).1
  have oldLookupEntry : arena.entries[old.id]? = some arena.entries[old.id] := by
    simp [oldInBounds]
  simp only [ExpressionArena.lookupMatrix]
  rw [Array.getElem?_push_lt oldInBounds, oldLookupEntry]

theorem ExpressionArena.Extends.lookupMatrix
    {older newer : ExpressionArena}
    (extension : older.Extends newer)
    {reference : MatrixExprRef}
    {expression : MatrixExpr}
    (lookup : older.lookupMatrix reference = some expression) :
    newer.lookupMatrix reference = some expression := by
  obtain ⟨suffix, extensionEq⟩ := extension
  unfold ExpressionArena.lookupMatrix at lookup ⊢
  have inBounds : reference.id < older.entries.size := by
    cases entryLookup : older.entries[reference.id]? with
    | none => simp [entryLookup] at lookup
    | some entry => exact (Array.getElem?_eq_some_iff.mp entryLookup).1
  rw [extensionEq, Array.getElem?_append_left inBounds]
  exact lookup

example :
    (ExpressionArena.internInteger { entries := #[] } (.intConstant 3)).map
      (fun result => result.2.id) = some 0 :=
  rfl

example : ExpressionArena.WF { entries := #[] } = true := rfl

end Mxx.Certificate
