import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard065

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

def shardIndex : Nat := 65
def shardStartEvent : Nat := 16640
def shardEndEvent : Nat := 16896
def rawSemanticCount : Nat := 134
def rawBoundTransferCount : Nat := 67
def rawResultCount : Nat := 67
def rawRelationCount : Nat := 0
def rawSurvivorFoldCount : Nat := 0
def rawPreFoldCount : Nat := 0
def rawInvocationEndCount : Nat := 0
def canonicalWork : Nat := 308

namespace Operation0
def selectedEvent : Nat := 16647
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨10513⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9420⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10512⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9420⟩⟩, ⟨.program ⟨214⟩, ⟨10512⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨9420⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨10512⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 16645
def selectedLeftResultEvent : Nat := 16642
def selectedRightResultEvent : Nat := 16639
def selectedResultEvent : Nat := 16647
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 16643 .coefficient) (.predecessor 1 16644 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation0

namespace Operation1
def selectedEvent : Nat := 16663
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨15327⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨15277⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15326⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 16662
def selectedLeftResultEvent : Nat := 16659
def selectedRightResultEvent : Nat := 16636
def selectedResultEvent : Nat := 16663

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16660 .coefficient, .predecessor 1 16661 .coefficient])) := by
  rfl

end Operation1

namespace Operation2
def selectedEvent : Nat := 16667
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨15383⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨15327⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15382⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 2
def selectedSumRuleEvent : Nat := 16666
def selectedLeftResultEvent : Nat := 16663
def selectedRightResultEvent : Nat := 16613
def selectedResultEvent : Nat := 16667

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16664 .coefficient, .predecessor 1 16665 .coefficient])) := by
  rfl

end Operation2

namespace Operation3
def selectedEvent : Nat := 16671
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17364⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨15383⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17363⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 3
def selectedSumRuleEvent : Nat := 16670
def selectedLeftResultEvent : Nat := 16667
def selectedRightResultEvent : Nat := 16590
def selectedResultEvent : Nat := 16671

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16668 .coefficient, .predecessor 1 16669 .coefficient])) := by
  rfl

end Operation3

namespace Operation4
def selectedEvent : Nat := 16675
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17365⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17364⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15641⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 4
def selectedSumRuleEvent : Nat := 16674
def selectedLeftResultEvent : Nat := 16671
def selectedRightResultEvent : Nat := 16567
def selectedResultEvent : Nat := 16675

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16672 .coefficient, .predecessor 1 16673 .coefficient])) := by
  rfl

end Operation4

namespace Operation5
def selectedEvent : Nat := 16679
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17366⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17365⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15760⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 5
def selectedSumRuleEvent : Nat := 16678
def selectedLeftResultEvent : Nat := 16675
def selectedRightResultEvent : Nat := 16544
def selectedResultEvent : Nat := 16679

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16676 .coefficient, .predecessor 1 16677 .coefficient])) := by
  rfl

end Operation5

namespace Operation6
def selectedEvent : Nat := 16683
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17367⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17366⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15879⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 6
def selectedSumRuleEvent : Nat := 16682
def selectedLeftResultEvent : Nat := 16679
def selectedRightResultEvent : Nat := 16521
def selectedResultEvent : Nat := 16683

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16680 .coefficient, .predecessor 1 16681 .coefficient])) := by
  rfl

end Operation6

namespace Operation7
def selectedEvent : Nat := 16687
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17368⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17367⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15998⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 7
def selectedSumRuleEvent : Nat := 16686
def selectedLeftResultEvent : Nat := 16683
def selectedRightResultEvent : Nat := 16498
def selectedResultEvent : Nat := 16687

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16684 .coefficient, .predecessor 1 16685 .coefficient])) := by
  rfl

end Operation7

namespace Operation8
def selectedEvent : Nat := 16691
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17369⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17368⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16117⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 8
def selectedSumRuleEvent : Nat := 16690
def selectedLeftResultEvent : Nat := 16687
def selectedRightResultEvent : Nat := 16475
def selectedResultEvent : Nat := 16691

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16688 .coefficient, .predecessor 1 16689 .coefficient])) := by
  rfl

end Operation8

namespace Operation9
def selectedEvent : Nat := 16695
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18393⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17369⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18392⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 9
def selectedSumRuleEvent : Nat := 16694
def selectedLeftResultEvent : Nat := 16691
def selectedRightResultEvent : Nat := 16452
def selectedResultEvent : Nat := 16695

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16692 .coefficient, .predecessor 1 16693 .coefficient])) := by
  rfl

end Operation9

namespace Operation10
def selectedEvent : Nat := 16699
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18394⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18393⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16320⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 10
def selectedSumRuleEvent : Nat := 16698
def selectedLeftResultEvent : Nat := 16695
def selectedRightResultEvent : Nat := 16429
def selectedResultEvent : Nat := 16699

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16696 .coefficient, .predecessor 1 16697 .coefficient])) := by
  rfl

end Operation10

namespace Operation11
def selectedEvent : Nat := 16703
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18395⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18394⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17132⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 11
def selectedSumRuleEvent : Nat := 16702
def selectedLeftResultEvent : Nat := 16699
def selectedRightResultEvent : Nat := 16406
def selectedResultEvent : Nat := 16703

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16700 .coefficient, .predecessor 1 16701 .coefficient])) := by
  rfl

end Operation11

namespace Operation12
def selectedEvent : Nat := 16707
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18396⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17916⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17916⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18395⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17916⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 12
def selectedSumRuleEvent : Nat := 16706
def selectedLeftResultEvent : Nat := 16703
def selectedRightResultEvent : Nat := 16383
def selectedResultEvent : Nat := 16707

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16704 .coefficient, .predecessor 1 16705 .coefficient])) := by
  rfl

end Operation12

namespace Operation13
def selectedEvent : Nat := 16711
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18397⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17916⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18217⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17916⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18217⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18396⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18217⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 13
def selectedSumRuleEvent : Nat := 16710
def selectedLeftResultEvent : Nat := 16707
def selectedRightResultEvent : Nat := 16360
def selectedResultEvent : Nat := 16711

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16708 .coefficient, .predecessor 1 16709 .coefficient])) := by
  rfl

end Operation13

namespace Operation14
def selectedEvent : Nat := 16715
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18398⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17916⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18217⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17916⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18217⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18397⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16691⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 14
def selectedSumRuleEvent : Nat := 16714
def selectedLeftResultEvent : Nat := 16711
def selectedRightResultEvent : Nat := 16337
def selectedResultEvent : Nat := 16715

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16712 .coefficient, .predecessor 1 16713 .coefficient])) := by
  rfl

end Operation14

namespace Operation15
def selectedEvent : Nat := 16719
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18399⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17916⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18217⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16810⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16810⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17916⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18217⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18398⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16810⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 15
def selectedSumRuleEvent : Nat := 16718
def selectedLeftResultEvent : Nat := 16715
def selectedRightResultEvent : Nat := 16314
def selectedResultEvent : Nat := 16719

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16716 .coefficient, .predecessor 1 16717 .coefficient])) := by
  rfl

end Operation15

namespace Operation16
def selectedEvent : Nat := 16723
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18400⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16810⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17916⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18217⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17097⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16810⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17097⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17916⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18217⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18399⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17097⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 16
def selectedSumRuleEvent : Nat := 16722
def selectedLeftResultEvent : Nat := 16719
def selectedRightResultEvent : Nat := 16291
def selectedResultEvent : Nat := 16723

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16720 .coefficient, .predecessor 1 16721 .coefficient])) := by
  rfl

end Operation16

namespace Operation17
def selectedEvent : Nat := 16727
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18401⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16810⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17097⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17916⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18217⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18182⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16810⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17097⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17916⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18182⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18217⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18400⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18182⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 17
def selectedSumRuleEvent : Nat := 16726
def selectedLeftResultEvent : Nat := 16723
def selectedRightResultEvent : Nat := 16268
def selectedResultEvent : Nat := 16727

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16724 .coefficient, .predecessor 1 16725 .coefficient])) := by
  rfl

end Operation17

namespace Operation18
def selectedEvent : Nat := 16774
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18665⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16810⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17097⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17916⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18182⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18217⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15998⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16117⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16320⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16810⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17097⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17132⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17916⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18217⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18392⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18664⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 18
def selectedSumRuleEvent : Nat := 16755
def selectedLeftResultEvent : Nat := 16752
def selectedRightResultEvent : Nat := 16750
def selectedResultEvent : Nat := 16774
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 16753 .coefficient) (.predecessor 1 16754 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation18

namespace Operation19
def selectedEvent : Nat := 16832
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6795⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6709⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6711⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 16831
def selectedLeftResultEvent : Nat := 16828
def selectedRightResultEvent : Nat := 16825
def selectedResultEvent : Nat := 16832

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16829 .coefficient, .predecessor 1 16830 .coefficient])) := by
  rfl

end Operation19

namespace Operation20
def selectedEvent : Nat := 16836
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6796⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6795⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6713⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 2
def selectedSumRuleEvent : Nat := 16835
def selectedLeftResultEvent : Nat := 16832
def selectedRightResultEvent : Nat := 16822
def selectedResultEvent : Nat := 16836

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16833 .coefficient, .predecessor 1 16834 .coefficient])) := by
  rfl

end Operation20

namespace Operation21
def selectedEvent : Nat := 16840
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6797⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6796⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6715⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 3
def selectedSumRuleEvent : Nat := 16839
def selectedLeftResultEvent : Nat := 16836
def selectedRightResultEvent : Nat := 16819
def selectedResultEvent : Nat := 16840

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16837 .coefficient, .predecessor 1 16838 .coefficient])) := by
  rfl

end Operation21

namespace Operation22
def selectedEvent : Nat := 16844
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6798⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6797⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6717⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 4
def selectedSumRuleEvent : Nat := 16843
def selectedLeftResultEvent : Nat := 16840
def selectedRightResultEvent : Nat := 16816
def selectedResultEvent : Nat := 16844

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16841 .coefficient, .predecessor 1 16842 .coefficient])) := by
  rfl

end Operation22

namespace Operation23
def selectedEvent : Nat := 16848
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6799⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6798⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6719⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 5
def selectedSumRuleEvent : Nat := 16847
def selectedLeftResultEvent : Nat := 16844
def selectedRightResultEvent : Nat := 16813
def selectedResultEvent : Nat := 16848

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16845 .coefficient, .predecessor 1 16846 .coefficient])) := by
  rfl

end Operation23

namespace Operation24
def selectedEvent : Nat := 16852
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6800⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6799⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6721⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 6
def selectedSumRuleEvent : Nat := 16851
def selectedLeftResultEvent : Nat := 16848
def selectedRightResultEvent : Nat := 16810
def selectedResultEvent : Nat := 16852

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16849 .coefficient, .predecessor 1 16850 .coefficient])) := by
  rfl

end Operation24

namespace Operation25
def selectedEvent : Nat := 16856
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6801⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6800⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6723⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 7
def selectedSumRuleEvent : Nat := 16855
def selectedLeftResultEvent : Nat := 16852
def selectedRightResultEvent : Nat := 16807
def selectedResultEvent : Nat := 16856

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16853 .coefficient, .predecessor 1 16854 .coefficient])) := by
  rfl

end Operation25

namespace Operation26
def selectedEvent : Nat := 16860
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6802⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6801⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6725⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 8
def selectedSumRuleEvent : Nat := 16859
def selectedLeftResultEvent : Nat := 16856
def selectedRightResultEvent : Nat := 16804
def selectedResultEvent : Nat := 16860

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16857 .coefficient, .predecessor 1 16858 .coefficient])) := by
  rfl

end Operation26

namespace Operation27
def selectedEvent : Nat := 16864
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6803⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6802⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6727⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 9
def selectedSumRuleEvent : Nat := 16863
def selectedLeftResultEvent : Nat := 16860
def selectedRightResultEvent : Nat := 16801
def selectedResultEvent : Nat := 16864

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16861 .coefficient, .predecessor 1 16862 .coefficient])) := by
  rfl

end Operation27

namespace Operation28
def selectedEvent : Nat := 16868
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6804⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6803⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6729⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 10
def selectedSumRuleEvent : Nat := 16867
def selectedLeftResultEvent : Nat := 16864
def selectedRightResultEvent : Nat := 16798
def selectedResultEvent : Nat := 16868

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16865 .coefficient, .predecessor 1 16866 .coefficient])) := by
  rfl

end Operation28

namespace Operation29
def selectedEvent : Nat := 16872
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6805⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6804⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6731⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 11
def selectedSumRuleEvent : Nat := 16871
def selectedLeftResultEvent : Nat := 16868
def selectedRightResultEvent : Nat := 16795
def selectedResultEvent : Nat := 16872

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16869 .coefficient, .predecessor 1 16870 .coefficient])) := by
  rfl

end Operation29

namespace Operation30
def selectedEvent : Nat := 16876
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6806⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6805⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6733⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 12
def selectedSumRuleEvent : Nat := 16875
def selectedLeftResultEvent : Nat := 16872
def selectedRightResultEvent : Nat := 16792
def selectedResultEvent : Nat := 16876

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16873 .coefficient, .predecessor 1 16874 .coefficient])) := by
  rfl

end Operation30

namespace Operation31
def selectedEvent : Nat := 16880
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6807⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6806⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6735⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 13
def selectedSumRuleEvent : Nat := 16879
def selectedLeftResultEvent : Nat := 16876
def selectedRightResultEvent : Nat := 16789
def selectedResultEvent : Nat := 16880

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16877 .coefficient, .predecessor 1 16878 .coefficient])) := by
  rfl

end Operation31

namespace Operation32
def selectedEvent : Nat := 16884
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6808⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6807⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6737⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 14
def selectedSumRuleEvent : Nat := 16883
def selectedLeftResultEvent : Nat := 16880
def selectedRightResultEvent : Nat := 16786
def selectedResultEvent : Nat := 16884

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16881 .coefficient, .predecessor 1 16882 .coefficient])) := by
  rfl

end Operation32

namespace Operation33
def selectedEvent : Nat := 16888
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6809⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6808⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6739⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 15
def selectedSumRuleEvent : Nat := 16887
def selectedLeftResultEvent : Nat := 16884
def selectedRightResultEvent : Nat := 16783
def selectedResultEvent : Nat := 16888

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16885 .coefficient, .predecessor 1 16886 .coefficient])) := by
  rfl

end Operation33

namespace Operation34
def selectedEvent : Nat := 16892
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨6810⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6809⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6741⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 16
def selectedSumRuleEvent : Nat := 16891
def selectedLeftResultEvent : Nat := 16888
def selectedRightResultEvent : Nat := 16780
def selectedResultEvent : Nat := 16892

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 16889 .coefficient, .predecessor 1 16890 .coefficient])) := by
  rfl

end Operation34

def theoremCount : Nat := 210

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard065
