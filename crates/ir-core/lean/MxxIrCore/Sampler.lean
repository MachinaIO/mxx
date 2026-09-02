import MxxIrCore.Eval

namespace Mxx.IR

noncomputable section

def FamilyIndex.append : {left right : List Nat} →
    FamilyIndex left → FamilyIndex right → FamilyIndex (left ++ right)
  | [], _, (), rightIndex => rightIndex
  | _ :: _, _, (head, tail), rightIndex =>
      (head, FamilyIndex.append tail rightIndex)

/- A sampler argument executes in the same dynamic scope occurrence as the sampler node.  The
   static wire is read from the stored node rather than supplied by an application predicate. -/
def SampleRef.argumentOccurrence? {data : ProgramData} (sample : SampleRef data)
    (index : Nat) : Option WireOccurrence := do
  let stage ← data.stages[sample.occurrence.stage]?
  let scope ← scopeAt stage sample.occurrence.wire.scope
  let node ← nodeAt scope sample.occurrence.wire.node
  let wire ← node.arguments[index]?
  pure (occurrenceOf sample.occurrence.stage sample.occurrence.path wire)

def SampleRef.Reached {data : ProgramData} {backend : SemanticBackend}
    (trace : Trace backend) (sample : SampleRef data) : Prop :=
  ∃ value, traceValueAt trace sample.occurrence = some value

def closedNatural? (expression : StructuralIntExpr) : Option Nat :=
  match expression.eval {} with
  | .ok value => if 0 ≤ value then some value.toNat else none
  | .error _ => none

def closedIntervalCutoff? (range : IntRange) : Option Nat := do
  let start ← range.start.eval {} |>.toOption
  let stop ← range.stop.eval {} |>.toOption
  pure (max start.natAbs stop.natAbs)

/- This classification is a deterministic projection of the stored sampler payload.  A cutoff
   that is not a closed nonnegative integer is rejected instead of accepting a caller-selected
   value.  Samplers without a smallness obligation still retain output/trace identity. -/
inductive SamplerFactKind where
  | cutoff (matrixType : MatrixType) (bound : Nat)
  | preimage (matrixType : MatrixType) (bound : Nat)
  | familyPreimage (matrixType : MatrixType) (bound : Nat)
  | outputOnly
  | invalid

def SampleRef.factKind {data : ProgramData} (sample : SampleRef data) : SamplerFactKind :=
  match sample.payload with
  | .uniformIntervalSample matrixType range =>
      match closedIntervalCutoff? range with
      | some bound => .cutoff matrixType bound
      | none => .invalid
  | .gaussianSample matrixType _ bound =>
      match closedNatural? bound with
      | some value => .cutoff matrixType value
      | none => .invalid
  | .preimageSample matrixType bound =>
      match closedNatural? bound with
      | some value => .preimage matrixType value
      | none => .invalid
  | .familyPreimageSample matrixType bound =>
      match closedNatural? bound with
      | some value => .familyPreimage matrixType value
      | none => .invalid
  | _ => .outputOnly

example : closedNatural? (.literal 7) = some 7 := rfl

example : closedNatural? (.literal (-1)) = none := rfl

end

end Mxx.IR
