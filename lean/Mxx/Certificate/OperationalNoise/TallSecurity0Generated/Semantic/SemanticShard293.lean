import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard293

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

def shardIndex : Nat := 293
def shardStartEvent : Nat := 75008
def shardEndEvent : Nat := 75264
def rawSemanticCount : Nat := 146
def rawBoundTransferCount : Nat := 73
def rawResultCount : Nat := 73
def rawRelationCount : Nat := 0
def rawSurvivorFoldCount : Nat := 0
def rawPreFoldCount : Nat := 0
def rawInvocationEndCount : Nat := 0
def canonicalWork : Nat := 160

namespace Operation0
def selectedEvent : Nat := 75012
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨13982⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13981⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11381⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨13981⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11381⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 75010
def selectedLeftResultEvent : Nat := 75007
def selectedRightResultEvent : Nat := 75004
def selectedResultEvent : Nat := 75012
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 75008 .coefficient) (.predecessor 1 75009 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation0

namespace Operation1
def selectedEvent : Nat := 75035
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨13765⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13764⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11297⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨13764⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11297⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 75033
def selectedLeftResultEvent : Nat := 75030
def selectedRightResultEvent : Nat := 75027
def selectedResultEvent : Nat := 75035
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 75031 .coefficient) (.predecessor 1 75032 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation1

namespace Operation2
def selectedEvent : Nat := 75058
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨13548⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13547⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11213⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨13547⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11213⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 75056
def selectedLeftResultEvent : Nat := 75053
def selectedRightResultEvent : Nat := 75050
def selectedResultEvent : Nat := 75058
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 75054 .coefficient) (.predecessor 1 75055 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation2

namespace Operation3
def selectedEvent : Nat := 75081
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨12155⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12154⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11129⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨12154⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11129⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 75079
def selectedLeftResultEvent : Nat := 75076
def selectedRightResultEvent : Nat := 75073
def selectedResultEvent : Nat := 75081
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 75077 .coefficient) (.predecessor 1 75078 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation3

namespace Operation4
def selectedEvent : Nat := 75104
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨10970⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10837⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10969⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨10837⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨10969⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 75102
def selectedLeftResultEvent : Nat := 75099
def selectedRightResultEvent : Nat := 75096
def selectedResultEvent : Nat := 75104
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 75100 .coefficient) (.predecessor 1 75101 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation4

namespace Operation5
def selectedEvent : Nat := 75127
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨10669⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9500⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨9500⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨10668⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 75125
def selectedLeftResultEvent : Nat := 75122
def selectedRightResultEvent : Nat := 75119
def selectedResultEvent : Nat := 75127
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 75123 .coefficient) (.predecessor 1 75124 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation5

namespace Operation6
def selectedEvent : Nat := 75150
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨10473⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9395⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10472⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨9395⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨10472⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 75148
def selectedLeftResultEvent : Nat := 75145
def selectedRightResultEvent : Nat := 75142
def selectedResultEvent : Nat := 75150
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 75146 .coefficient) (.predecessor 1 75147 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation6

namespace Operation7
def selectedEvent : Nat := 75166
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨15307⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨15262⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15306⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 75165
def selectedLeftResultEvent : Nat := 75162
def selectedRightResultEvent : Nat := 75139
def selectedResultEvent : Nat := 75166

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75163 .coefficient, .predecessor 1 75164 .coefficient])) := by
  rfl

end Operation7

namespace Operation8
def selectedEvent : Nat := 75170
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨15363⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨15307⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15362⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 2
def selectedSumRuleEvent : Nat := 75169
def selectedLeftResultEvent : Nat := 75166
def selectedRightResultEvent : Nat := 75116
def selectedResultEvent : Nat := 75170

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75167 .coefficient, .predecessor 1 75168 .coefficient])) := by
  rfl

end Operation8

namespace Operation9
def selectedEvent : Nat := 75174
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17319⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨15363⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17318⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 3
def selectedSumRuleEvent : Nat := 75173
def selectedLeftResultEvent : Nat := 75170
def selectedRightResultEvent : Nat := 75093
def selectedResultEvent : Nat := 75174

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75171 .coefficient, .predecessor 1 75172 .coefficient])) := by
  rfl

end Operation9

namespace Operation10
def selectedEvent : Nat := 75178
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17320⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17319⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15626⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 4
def selectedSumRuleEvent : Nat := 75177
def selectedLeftResultEvent : Nat := 75174
def selectedRightResultEvent : Nat := 75070
def selectedResultEvent : Nat := 75178

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75175 .coefficient, .predecessor 1 75176 .coefficient])) := by
  rfl

end Operation10

namespace Operation11
def selectedEvent : Nat := 75182
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17321⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17320⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15745⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 5
def selectedSumRuleEvent : Nat := 75181
def selectedLeftResultEvent : Nat := 75178
def selectedRightResultEvent : Nat := 75047
def selectedResultEvent : Nat := 75182

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75179 .coefficient, .predecessor 1 75180 .coefficient])) := by
  rfl

end Operation11

namespace Operation12
def selectedEvent : Nat := 75186
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17322⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17321⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15864⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 6
def selectedSumRuleEvent : Nat := 75185
def selectedLeftResultEvent : Nat := 75182
def selectedRightResultEvent : Nat := 75024
def selectedResultEvent : Nat := 75186

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75183 .coefficient, .predecessor 1 75184 .coefficient])) := by
  rfl

end Operation12

namespace Operation13
def selectedEvent : Nat := 75190
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17323⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17322⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15983⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 7
def selectedSumRuleEvent : Nat := 75189
def selectedLeftResultEvent : Nat := 75186
def selectedRightResultEvent : Nat := 75001
def selectedResultEvent : Nat := 75190

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75187 .coefficient, .predecessor 1 75188 .coefficient])) := by
  rfl

end Operation13

namespace Operation14
def selectedEvent : Nat := 75194
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17324⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17323⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16102⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 8
def selectedSumRuleEvent : Nat := 75193
def selectedLeftResultEvent : Nat := 75190
def selectedRightResultEvent : Nat := 74978
def selectedResultEvent : Nat := 75194

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75191 .coefficient, .predecessor 1 75192 .coefficient])) := by
  rfl

end Operation14

namespace Operation15
def selectedEvent : Nat := 75198
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18328⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17324⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18327⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 9
def selectedSumRuleEvent : Nat := 75197
def selectedLeftResultEvent : Nat := 75194
def selectedRightResultEvent : Nat := 74955
def selectedResultEvent : Nat := 75198

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75195 .coefficient, .predecessor 1 75196 .coefficient])) := by
  rfl

end Operation15

namespace Operation16
def selectedEvent : Nat := 75202
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18329⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18328⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16305⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 10
def selectedSumRuleEvent : Nat := 75201
def selectedLeftResultEvent : Nat := 75198
def selectedRightResultEvent : Nat := 74932
def selectedResultEvent : Nat := 75202

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75199 .coefficient, .predecessor 1 75200 .coefficient])) := by
  rfl

end Operation16

namespace Operation17
def selectedEvent : Nat := 75206
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18330⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18329⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17117⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 11
def selectedSumRuleEvent : Nat := 75205
def selectedLeftResultEvent : Nat := 75202
def selectedRightResultEvent : Nat := 74909
def selectedResultEvent : Nat := 75206

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75203 .coefficient, .predecessor 1 75204 .coefficient])) := by
  rfl

end Operation17

namespace Operation18
def selectedEvent : Nat := 75210
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18331⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18330⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17901⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 12
def selectedSumRuleEvent : Nat := 75209
def selectedLeftResultEvent : Nat := 75206
def selectedRightResultEvent : Nat := 74886
def selectedResultEvent : Nat := 75210

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75207 .coefficient, .predecessor 1 75208 .coefficient])) := by
  rfl

end Operation18

namespace Operation19
def selectedEvent : Nat := 75214
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18332⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18202⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18202⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18331⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18202⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 13
def selectedSumRuleEvent : Nat := 75213
def selectedLeftResultEvent : Nat := 75210
def selectedRightResultEvent : Nat := 74863
def selectedResultEvent : Nat := 75214

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75211 .coefficient, .predecessor 1 75212 .coefficient])) := by
  rfl

end Operation19

namespace Operation20
def selectedEvent : Nat := 75218
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18333⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18202⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16676⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16676⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18202⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18332⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16676⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 14
def selectedSumRuleEvent : Nat := 75217
def selectedLeftResultEvent : Nat := 75214
def selectedRightResultEvent : Nat := 74840
def selectedResultEvent : Nat := 75218

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75215 .coefficient, .predecessor 1 75216 .coefficient])) := by
  rfl

end Operation20

namespace Operation21
def selectedEvent : Nat := 75222
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18334⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16676⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18202⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16676⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18202⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18333⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16795⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 15
def selectedSumRuleEvent : Nat := 75221
def selectedLeftResultEvent : Nat := 75218
def selectedRightResultEvent : Nat := 74817
def selectedResultEvent : Nat := 75222

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75219 .coefficient, .predecessor 1 75220 .coefficient])) := by
  rfl

end Operation21

namespace Operation22
def selectedEvent : Nat := 75226
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18335⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16676⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18202⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17082⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16676⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17082⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18202⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18334⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17082⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 16
def selectedSumRuleEvent : Nat := 75225
def selectedLeftResultEvent : Nat := 75222
def selectedRightResultEvent : Nat := 74794
def selectedResultEvent : Nat := 75226

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75223 .coefficient, .predecessor 1 75224 .coefficient])) := by
  rfl

end Operation22

namespace Operation23
def selectedEvent : Nat := 75230
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18336⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16676⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17082⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18202⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18167⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16676⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17082⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18167⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18202⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18335⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18167⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 17
def selectedSumRuleEvent : Nat := 75229
def selectedLeftResultEvent : Nat := 75226
def selectedRightResultEvent : Nat := 74771
def selectedResultEvent : Nat := 75230

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 75227 .coefficient, .predecessor 1 75228 .coefficient])) := by
  rfl

end Operation23

def theoremCount : Nat := 144

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard293
