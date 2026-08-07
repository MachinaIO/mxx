import Mxx.Certificate.Rules.BggThreeTrace

namespace Mxx.Certificate

/-!
# Trace-bound six-way BGG gate selection

The executable Boolean and BGG bodies use the same six candidate positions.  This module turns
the integer found at the actual selector wire into the corresponding closed `BggBooleanGate`.
The conversion is fail-closed outside `0 .. 5`; it accepts neither a callback nor a
caller-provided equation describing the selected gate.
-/

/-- Positional encoding used by all checked six-way gate bodies. -/
def BggBooleanGate.index : BggBooleanGate → Nat
  | .zero => 0
  | .one => 1
  | .copyLeft => 2
  | .notLeft => 3
  | .and => 4
  | .xor => 5

/-- Decode only the closed six-way gate universe. -/
def BggBooleanGate.ofSelector? : Int → Option BggBooleanGate
  | 0 => some .zero
  | 1 => some .one
  | 2 => some .copyLeft
  | 3 => some .notLeft
  | 4 => some .and
  | 5 => some .xor
  | _ => none

/-- The scalar formula occupying each checked candidate position. -/
def CheckedSixWayBooleanSkeleton.formulaForGate
    (skeleton : CheckedSixWayBooleanSkeleton) : BggBooleanGate → FrozenPointwiseScalarFormula
  | .zero => .compare .equal (.boolToInt (.boolean false)) (.integer 1)
  | .one => .compare .equal (.boolToInt (.boolean true)) (.integer 1)
  | .copyLeft => skeleton.leftFormula
  | .notLeft => .compare .equal (.boolToInt skeleton.leftFormula) (.integer 0)
  | .and => .compare .equal
      (.intBinary .multiply (.boolToInt skeleton.leftFormula)
        (.boolToInt skeleton.rightFormula)) (.integer 1)
  | .xor => .compare .equal
      (.intBinary .add (.boolToInt skeleton.leftFormula)
        (.boolToInt skeleton.rightFormula)) (.integer 1)

/-- The checked skeleton stores the formula for a gate at exactly its closed selector index. -/
theorem CheckedSixWayBooleanSkeleton.formulaAtGate
    (skeleton : CheckedSixWayBooleanSkeleton)
    (gate : BggBooleanGate) :
    skeleton.formulas[gate.index]? = some (skeleton.formulaForGate gate) := by
  rw [skeleton.formulasMatch]
  cases gate <;> rfl

/-- Exact selector evidence tied to one actual executable wire environment.

`decoded` is computed by the closed decoder above.  Consequently this evidence cannot associate
an actual selector value with a different gate, and values outside the six supported positions
cannot inhabit the structure.
-/
structure TraceBoundSixWayGateSelector
    {program : Mxx.Ir.Prog}
    (selection : CheckedSixWayGateSelection program)
    (wires : Mxx.Ir.WireEnvironment) where
  selector : Int
  selectorFound :
    Mxx.Ir.lookupWire selection.gateSelector wires = some (.integer selector)
  gate : BggBooleanGate
  decoded : BggBooleanGate.ofSelector? selector = some gate
  position : selector.toNat = gate.index

/-- Construct trace-bound evidence by running the closed decoder on the actual wire value.
Unsupported selector integers return `none`. -/
def TraceBoundSixWayGateSelector.ofLookup?
    {program : Mxx.Ir.Prog}
    (selection : CheckedSixWayGateSelection program)
    (wires : Mxx.Ir.WireEnvironment)
    (selector : Int)
    (selectorFound :
      Mxx.Ir.lookupWire selection.gateSelector wires = some (.integer selector)) :
    Option (TraceBoundSixWayGateSelector selection wires) :=
  match selector with
  | 0 => some { selector := 0, selectorFound, gate := .zero, decoded := rfl, position := rfl }
  | 1 => some { selector := 1, selectorFound, gate := .one, decoded := rfl, position := rfl }
  | 2 => some { selector := 2, selectorFound, gate := .copyLeft, decoded := rfl, position := rfl }
  | 3 => some { selector := 3, selectorFound, gate := .notLeft, decoded := rfl, position := rfl }
  | 4 => some { selector := 4, selectorFound, gate := .and, decoded := rfl, position := rfl }
  | 5 => some { selector := 5, selectorFound, gate := .xor, decoded := rfl, position := rfl }
  | _ => none

/-- The selected gate fixes an in-bounds executable candidate wire. -/
theorem TraceBoundSixWayGateSelector.candidateAtGate
    {program : Mxx.Ir.Prog}
    {selection : CheckedSixWayGateSelection program}
    {wires : Mxx.Ir.WireEnvironment}
    (selected : TraceBoundSixWayGateSelector selection wires) :
    ∃ candidate,
      selection.candidates[selected.gate.index]? = some candidate := by
  have indexBound : selected.gate.index < selection.candidates.length := by
    rw [selection.sixCandidates]
    cases selected.gate <;> decide
  exact ⟨selection.candidates[selected.gate.index], List.getElem?_eq_getElem indexBound⟩

/-- The actual selector value and the candidate position used by the checked skeleton coincide. -/
theorem TraceBoundSixWayGateSelector.selectorPosition
    {program : Mxx.Ir.Prog}
    {selection : CheckedSixWayGateSelection program}
    {wires : Mxx.Ir.WireEnvironment}
    (selected : TraceBoundSixWayGateSelector selection wires) :
    selected.selector.toNat = selected.gate.index := selected.position

/-- Combined positional consequence used by BGG/Boolean trace coupling.  Both the executable
candidate wire and the normalized Boolean formula are selected by the same actual integer. -/
theorem TraceBoundSixWayGateSelector.candidateAndBooleanFormula
    {program : Mxx.Ir.Prog}
    {selection : CheckedSixWayGateSelection program}
    {wires : Mxx.Ir.WireEnvironment}
    (selected : TraceBoundSixWayGateSelector selection wires)
    (skeleton : CheckedSixWayBooleanSkeleton) :
    ∃ candidate,
      selection.candidates[selected.selector.toNat]? = some candidate ∧
      skeleton.formulas[selected.selector.toNat]? =
        some (skeleton.formulaForGate selected.gate) := by
  obtain ⟨candidate, candidateFound⟩ := selected.candidateAtGate
  rw [selected.selectorPosition]
  exact ⟨candidate, candidateFound, skeleton.formulaAtGate selected.gate⟩

end Mxx.Certificate
