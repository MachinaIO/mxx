import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard350

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

def shardIndex : Nat := 350
def shardStartEvent : Nat := 89600
def shardEndEvent : Nat := 89856
def rawSemanticCount : Nat := 145
def rawBoundTransferCount : Nat := 72
def rawResultCount : Nat := 73
def rawRelationCount : Nat := 0
def rawSurvivorFoldCount : Nat := 0
def rawPreFoldCount : Nat := 0
def rawInvocationEndCount : Nat := 0
def canonicalWork : Nat := 160

namespace Operation0
def selectedEvent : Nat := 89601
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨13991⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11385⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11385⟩⟩, ⟨.program ⟨214⟩, ⟨13990⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨13990⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11385⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 89599
def selectedLeftResultEvent : Nat := 89596
def selectedRightResultEvent : Nat := 89593
def selectedResultEvent : Nat := 89601
def leftScalar : Bool := false
def rightScalar : Bool := false

theorem resultAgreement : CanonicalAgreement output (productPoly left right leftScalar rightScalar) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by
  exact productCanonicalResultSound env left right output leftScalar rightScalar resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 89597 .coefficient) (.predecessor 1 89598 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation0

namespace Operation1
def selectedEvent : Nat := 89624
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨13774⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13773⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11301⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨13773⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11301⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 89622
def selectedLeftResultEvent : Nat := 89619
def selectedRightResultEvent : Nat := 89616
def selectedResultEvent : Nat := 89624
def leftScalar : Bool := false
def rightScalar : Bool := false

theorem resultAgreement : CanonicalAgreement output (productPoly left right leftScalar rightScalar) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by
  exact productCanonicalResultSound env left right output leftScalar rightScalar resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 89620 .coefficient) (.predecessor 1 89621 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation1

namespace Operation2
def selectedEvent : Nat := 89647
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨13557⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13556⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11217⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11217⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨13556⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11217⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 89645
def selectedLeftResultEvent : Nat := 89642
def selectedRightResultEvent : Nat := 89639
def selectedResultEvent : Nat := 89647
def leftScalar : Bool := false
def rightScalar : Bool := false

theorem resultAgreement : CanonicalAgreement output (productPoly left right leftScalar rightScalar) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by
  exact productCanonicalResultSound env left right output leftScalar rightScalar resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 89643 .coefficient) (.predecessor 1 89644 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation2

namespace Operation3
def selectedEvent : Nat := 89670
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨12164⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11133⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨12163⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11133⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 89668
def selectedLeftResultEvent : Nat := 89665
def selectedRightResultEvent : Nat := 89662
def selectedResultEvent : Nat := 89670
def leftScalar : Bool := false
def rightScalar : Bool := false

theorem resultAgreement : CanonicalAgreement output (productPoly left right leftScalar rightScalar) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by
  exact productCanonicalResultSound env left right output leftScalar rightScalar resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 89666 .coefficient) (.predecessor 1 89667 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation3

namespace Operation4
def selectedEvent : Nat := 89693
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨10978⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10842⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10977⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨10842⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨10977⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 89691
def selectedLeftResultEvent : Nat := 89688
def selectedRightResultEvent : Nat := 89685
def selectedResultEvent : Nat := 89693
def leftScalar : Bool := false
def rightScalar : Bool := false

theorem resultAgreement : CanonicalAgreement output (productPoly left right leftScalar rightScalar) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by
  exact productCanonicalResultSound env left right output leftScalar rightScalar resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 89689 .coefficient) (.predecessor 1 89690 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation4

namespace Operation5
def selectedEvent : Nat := 89716
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨10677⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9505⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10676⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9505⟩⟩, ⟨.program ⟨214⟩, ⟨10676⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨9505⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨10676⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 89714
def selectedLeftResultEvent : Nat := 89711
def selectedRightResultEvent : Nat := 89708
def selectedResultEvent : Nat := 89716
def leftScalar : Bool := false
def rightScalar : Bool := false

theorem resultAgreement : CanonicalAgreement output (productPoly left right leftScalar rightScalar) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by
  exact productCanonicalResultSound env left right output leftScalar rightScalar resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 89712 .coefficient) (.predecessor 1 89713 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation5

namespace Operation6
def selectedEvent : Nat := 89739
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨10481⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9400⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10480⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨9400⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨10480⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 89737
def selectedLeftResultEvent : Nat := 89734
def selectedRightResultEvent : Nat := 89731
def selectedResultEvent : Nat := 89739
def leftScalar : Bool := false
def rightScalar : Bool := false

theorem resultAgreement : CanonicalAgreement output (productPoly left right leftScalar rightScalar) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by
  exact productCanonicalResultSound env left right output leftScalar rightScalar resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 89735 .coefficient) (.predecessor 1 89736 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation6

namespace Operation7
def selectedEvent : Nat := 89755
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨15311⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨15265⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15310⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 89754
def selectedLeftResultEvent : Nat := 89751
def selectedRightResultEvent : Nat := 89728
def selectedResultEvent : Nat := 89755

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89752 .coefficient, .predecessor 1 89753 .coefficient])) := by
  rfl

end Operation7

namespace Operation8
def selectedEvent : Nat := 89759
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨15367⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨15311⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15366⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 2
def selectedSumRuleEvent : Nat := 89758
def selectedLeftResultEvent : Nat := 89755
def selectedRightResultEvent : Nat := 89705
def selectedResultEvent : Nat := 89759

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89756 .coefficient, .predecessor 1 89757 .coefficient])) := by
  rfl

end Operation8

namespace Operation9
def selectedEvent : Nat := 89763
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17328⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨15367⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17327⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 3
def selectedSumRuleEvent : Nat := 89762
def selectedLeftResultEvent : Nat := 89759
def selectedRightResultEvent : Nat := 89682
def selectedResultEvent : Nat := 89763

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89760 .coefficient, .predecessor 1 89761 .coefficient])) := by
  rfl

end Operation9

namespace Operation10
def selectedEvent : Nat := 89767
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17329⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17328⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15629⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 4
def selectedSumRuleEvent : Nat := 89766
def selectedLeftResultEvent : Nat := 89763
def selectedRightResultEvent : Nat := 89659
def selectedResultEvent : Nat := 89767

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89764 .coefficient, .predecessor 1 89765 .coefficient])) := by
  rfl

end Operation10

namespace Operation11
def selectedEvent : Nat := 89771
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17330⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17329⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15748⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 5
def selectedSumRuleEvent : Nat := 89770
def selectedLeftResultEvent : Nat := 89767
def selectedRightResultEvent : Nat := 89636
def selectedResultEvent : Nat := 89771

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89768 .coefficient, .predecessor 1 89769 .coefficient])) := by
  rfl

end Operation11

namespace Operation12
def selectedEvent : Nat := 89775
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17331⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17330⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15867⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 6
def selectedSumRuleEvent : Nat := 89774
def selectedLeftResultEvent : Nat := 89771
def selectedRightResultEvent : Nat := 89613
def selectedResultEvent : Nat := 89775

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89772 .coefficient, .predecessor 1 89773 .coefficient])) := by
  rfl

end Operation12

namespace Operation13
def selectedEvent : Nat := 89779
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17332⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17331⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15986⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 7
def selectedSumRuleEvent : Nat := 89778
def selectedLeftResultEvent : Nat := 89775
def selectedRightResultEvent : Nat := 89590
def selectedResultEvent : Nat := 89779

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89776 .coefficient, .predecessor 1 89777 .coefficient])) := by
  rfl

end Operation13

namespace Operation14
def selectedEvent : Nat := 89783
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17333⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17332⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16105⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 8
def selectedSumRuleEvent : Nat := 89782
def selectedLeftResultEvent : Nat := 89779
def selectedRightResultEvent : Nat := 89567
def selectedResultEvent : Nat := 89783

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89780 .coefficient, .predecessor 1 89781 .coefficient])) := by
  rfl

end Operation14

namespace Operation15
def selectedEvent : Nat := 89787
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18341⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17333⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18340⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 9
def selectedSumRuleEvent : Nat := 89786
def selectedLeftResultEvent : Nat := 89783
def selectedRightResultEvent : Nat := 89544
def selectedResultEvent : Nat := 89787

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89784 .coefficient, .predecessor 1 89785 .coefficient])) := by
  rfl

end Operation15

namespace Operation16
def selectedEvent : Nat := 89791
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18342⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18341⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16308⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 10
def selectedSumRuleEvent : Nat := 89790
def selectedLeftResultEvent : Nat := 89787
def selectedRightResultEvent : Nat := 89521
def selectedResultEvent : Nat := 89791

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89788 .coefficient, .predecessor 1 89789 .coefficient])) := by
  rfl

end Operation16

namespace Operation17
def selectedEvent : Nat := 89795
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18343⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18342⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17120⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 11
def selectedSumRuleEvent : Nat := 89794
def selectedLeftResultEvent : Nat := 89791
def selectedRightResultEvent : Nat := 89498
def selectedResultEvent : Nat := 89795

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89792 .coefficient, .predecessor 1 89793 .coefficient])) := by
  rfl

end Operation17

namespace Operation18
def selectedEvent : Nat := 89799
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18344⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18343⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17904⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 12
def selectedSumRuleEvent : Nat := 89798
def selectedLeftResultEvent : Nat := 89795
def selectedRightResultEvent : Nat := 89475
def selectedResultEvent : Nat := 89799

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89796 .coefficient, .predecessor 1 89797 .coefficient])) := by
  rfl

end Operation18

namespace Operation19
def selectedEvent : Nat := 89803
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18345⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18344⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18205⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 13
def selectedSumRuleEvent : Nat := 89802
def selectedLeftResultEvent : Nat := 89799
def selectedRightResultEvent : Nat := 89452
def selectedResultEvent : Nat := 89803

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89800 .coefficient, .predecessor 1 89801 .coefficient])) := by
  rfl

end Operation19

namespace Operation20
def selectedEvent : Nat := 89807
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18346⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16679⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16679⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18345⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16679⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 14
def selectedSumRuleEvent : Nat := 89806
def selectedLeftResultEvent : Nat := 89803
def selectedRightResultEvent : Nat := 89429
def selectedResultEvent : Nat := 89807

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89804 .coefficient, .predecessor 1 89805 .coefficient])) := by
  rfl

end Operation20

namespace Operation21
def selectedEvent : Nat := 89811
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18347⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16679⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16798⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16679⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16798⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18346⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16798⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 15
def selectedSumRuleEvent : Nat := 89810
def selectedLeftResultEvent : Nat := 89807
def selectedRightResultEvent : Nat := 89406
def selectedResultEvent : Nat := 89811

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89808 .coefficient, .predecessor 1 89809 .coefficient])) := by
  rfl

end Operation21

namespace Operation22
def selectedEvent : Nat := 89815
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18348⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16679⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16798⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17085⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16679⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16798⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17085⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18347⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17085⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 16
def selectedSumRuleEvent : Nat := 89814
def selectedLeftResultEvent : Nat := 89811
def selectedRightResultEvent : Nat := 89383
def selectedResultEvent : Nat := 89815

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89812 .coefficient, .predecessor 1 89813 .coefficient])) := by
  rfl

end Operation22

namespace Operation23
def selectedEvent : Nat := 89819
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18349⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16679⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16798⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17085⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18170⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16679⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16798⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17085⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18170⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18348⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18170⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 17
def selectedSumRuleEvent : Nat := 89818
def selectedLeftResultEvent : Nat := 89815
def selectedRightResultEvent : Nat := 89360
def selectedResultEvent : Nat := 89819

theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 89816 .coefficient, .predecessor 1 89817 .coefficient])) := by
  rfl

end Operation23

def theoremCount : Nat := 144

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard350
