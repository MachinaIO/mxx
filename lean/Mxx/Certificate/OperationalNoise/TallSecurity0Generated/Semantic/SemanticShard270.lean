import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard270

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

def shardIndex : Nat := 270
def shardStartEvent : Nat := 69120
def shardEndEvent : Nat := 69376
def rawSemanticCount : Nat := 148
def rawBoundTransferCount : Nat := 75
def rawResultCount : Nat := 63
def rawRelationCount : Nat := 3
def rawSurvivorFoldCount : Nat := 3
def rawPreFoldCount : Nat := 2
def rawInvocationEndCount : Nat := 2
def canonicalWork : Nat := 21

namespace Operation0
def selectedEvent : Nat := 69122
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16307⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6729⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16306⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 69121
def selectedLeftResultEvent : Nat := 69118
def selectedRightResultEvent : Nat := 69115
def selectedResultEvent : Nat := 69122

theorem resultAgreement : CanonicalAgreement output (subtract left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left - evalPolynomial env right := by
  exact subCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 69119 .coefficient, .predecessor 1 69120 .coefficient])) := by
  rfl

end Operation0

namespace Operation1
def selectedEvent : Nat := 69126
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨28509⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24348⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24348⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨16307⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨28505⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 4
def selectedSumRuleEvent : Nat := 69125
def selectedLeftResultEvent : Nat := 69122
def selectedRightResultEvent : Nat := 69107
def selectedResultEvent : Nat := 69126

theorem resultAgreement : CanonicalAgreement output (subtract left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left - evalPolynomial env right := by
  exact subCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 69123 .coefficient, .predecessor 1 69124 .coefficient])) := by
  rfl

end Operation1

namespace Operation2
def selectedEvent : Nat := 69167
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨11634⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11633⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨11633⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6566⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 69165
def selectedLeftResultEvent : Nat := 3270
def selectedRightResultEvent : Nat := 65295
def selectedResultEvent : Nat := 69167
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.tensor (.predecessor 0 69163 .coefficient) (.predecessor 1 69164 .coefficient) true false)) := by
  rfl

end Operation2

namespace Operation3
def selectedEvent : Nat := 69172
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨7199⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨5533⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6781⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 69170
def selectedLeftResultEvent : Nat := 65165
def selectedRightResultEvent : Nat := 10480
def selectedResultEvent : Nat := 69172
def leftScalar : Bool := true
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 69168 .coefficient) (.predecessor 1 69169 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation3

namespace Operation4
def selectedEvent : Nat := 69176
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨11635⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨7199⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11634⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 69175
def selectedLeftResultEvent : Nat := 69172
def selectedRightResultEvent : Nat := 69167
def selectedResultEvent : Nat := 69176

theorem resultAgreement : CanonicalAgreement output (subtract left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left - evalPolynomial env right := by
  exact subCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 69173 .coefficient, .predecessor 1 69174 .coefficient])) := by
  rfl

end Operation4

namespace Operation5
def selectedEvent : Nat := 69195
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨14636⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨14632⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6566⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 69193
def selectedLeftResultEvent : Nat := 3273
def selectedRightResultEvent : Nat := 65295
def selectedResultEvent : Nat := 69195
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.tensor (.predecessor 0 69191 .coefficient) (.predecessor 1 69192 .coefficient) true false)) := by
  rfl

end Operation5

namespace Operation6
def selectedEvent : Nat := 69200
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨7180⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨5533⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6762⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 69198
def selectedLeftResultEvent : Nat := 65165
def selectedRightResultEvent : Nat := 10521
def selectedResultEvent : Nat := 69200
def leftScalar : Bool := true
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 69196 .coefficient) (.predecessor 1 69197 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation6

namespace Operation7
def selectedEvent : Nat := 69204
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨14637⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨7180⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨14636⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 69203
def selectedLeftResultEvent : Nat := 69200
def selectedRightResultEvent : Nat := 69195
def selectedResultEvent : Nat := 69204

theorem resultAgreement : CanonicalAgreement output (subtract left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left - evalPolynomial env right := by
  exact subCanonicalResultSound env left right output resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 69201 .coefficient, .predecessor 1 69202 .coefficient])) := by
  rfl

end Operation7

namespace Operation8
def selectedEvent : Nat := 69295
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨19669⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19668⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨19668⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 69293
def selectedLeftResultEvent : Nat := 69290
def selectedRightResultEvent : Nat := 69288
def selectedResultEvent : Nat := 69295
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 69291 .coefficient) (.predecessor 1 69292 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation8

namespace Operation9
def selectedEvent : Nat := 69329
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨14633⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11633⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨14632⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11633⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 69327
def selectedLeftResultEvent : Nat := 69324
def selectedRightResultEvent : Nat := 69321
def selectedResultEvent : Nat := 69329
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 69325 .coefficient) (.predecessor 1 69326 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation9

namespace Operation10
def selectedEvent : Nat := 69359
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨14746⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨14745⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 69357
def selectedLeftResultEvent : Nat := 69354
def selectedRightResultEvent : Nat := 69352
def selectedResultEvent : Nat := 69359
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 69355 .coefficient) (.predecessor 1 69356 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation10

namespace Relation0
def selectedEvent : Nat := 69130
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨21831⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24348⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24348⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24348⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24348⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationContext0 : MonomialContext Owner := relationContext sourceKey sourceKey.centralFactors 0 2

theorem relationShape : relationPoly accumulator sourceKey relationContext0 (1) relationRhs = relationExpected := by rfl

theorem relationAgreement : CanonicalAgreement relationOutput (relationPoly accumulator sourceKey relationContext0 (1) relationRhs) := by decide +kernel

theorem relationSound (env : Env Owner)
    (baseRelation : evalMonomial env lhsKey % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 =
      evalPolynomial env relationRhs % Int.ofNat 100418593683253592432016548326729029359133068138294319235841) :
    evalPolynomial env relationOutput % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 =
      evalPolynomial env accumulator % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 := by
  exact relationCanonicalResultSound 100418593683253592432016548326729029359133068138294319235841 env accumulator sourceKey lhsKey
    sourceKey.centralFactors 0 2 (1) relationRhs relationOutput
    (by decide +kernel) baseRelation relationAgreement

def relationRhsEvent : Nat := 69128
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨28509⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 69134
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩) (1) 0 2 (.universal 69129 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩) (none) 69128)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.invocationEndExact relationRhsOwner 69127 relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21831⟩⟩, .relation 69130 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16305⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation0

namespace Relation1
def selectedEvent : Nat := 69217
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨14639⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }]
def relationContext0 : MonomialContext Owner := relationContext sourceKey sourceKey.centralFactors 0 2

theorem relationShape : relationPoly accumulator sourceKey relationContext0 (-1) relationRhs = relationExpected := by rfl

theorem relationAgreement : CanonicalAgreement relationOutput (relationPoly accumulator sourceKey relationContext0 (-1) relationRhs) := by decide +kernel

theorem relationSound (env : Env Owner)
    (baseRelation : evalMonomial env lhsKey % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 =
      evalPolynomial env relationRhs % Int.ofNat 100418593683253592432016548326729029359133068138294319235841) :
    evalPolynomial env relationOutput % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 =
      evalPolynomial env accumulator % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 := by
  exact relationCanonicalResultSound 100418593683253592432016548326729029359133068138294319235841 env accumulator sourceKey lhsKey
    sourceKey.centralFactors 0 2 (-1) relationRhs relationOutput
    (by decide +kernel) baseRelation relationAgreement

def relationRhsEvent : Nat := 10480
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨6781⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 69218
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7858⟩⟩) ⟨6781⟩ 10480)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.resultExact relationRhsOwner relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14639⟩⟩, .relation 69217 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation1

namespace Relation2
def selectedEvent : Nat := 69233
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨26216⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23666⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23666⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23666⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23666⟩⟩] } }]
def relationContext0 : MonomialContext Owner := relationContext sourceKey sourceKey.centralFactors 0 2

theorem relationShape : relationPoly accumulator sourceKey relationContext0 (-1) relationRhs = relationExpected := by rfl

theorem relationAgreement : CanonicalAgreement relationOutput (relationPoly accumulator sourceKey relationContext0 (-1) relationRhs) := by decide +kernel

theorem relationSound (env : Env Owner)
    (baseRelation : evalMonomial env lhsKey % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 =
      evalPolynomial env relationRhs % Int.ofNat 100418593683253592432016548326729029359133068138294319235841) :
    evalPolynomial env relationOutput % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 =
      evalPolynomial env accumulator % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 := by
  exact relationCanonicalResultSound 100418593683253592432016548326729029359133068138294319235841 env accumulator sourceKey lhsKey
    sourceKey.centralFactors 0 2 (-1) relationRhs relationOutput
    (by decide +kernel) baseRelation relationAgreement

def relationRhsEvent : Nat := 69159
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨23666⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 69234
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26215⟩⟩) ⟨23666⟩ 69159)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.resultExact relationRhsOwner relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26216⟩⟩, .relation 69233 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨23666⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation2

namespace Bound0
def selectedEvent : Nat := 69128
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨28509⟩⟩
def rootResultEvent : Nat := 69126
def prefoldEvent : Nat := 69127
def endEvent : Nat := 69128
def survivorEvents : List Nat := []
def rootRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24348⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24348⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24348⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rootTerms : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24348⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldTerms : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24348⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endTerms : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24348⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rootSummary : Bound := .exactZero
def prefoldSummary : Bound := .exactZero
def endSummary : Bound := .exactZero
def rootBound : Nat := 0
def prefoldBound : Nat := 0
def survivorContributions : List Nat := []
def survivorBounds : List Nat := []
theorem survivorBoundsSound : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributions survivorBounds := by
  exact List.Forall₂.nil

theorem prefoldResult : prefoldTerms = rootTerms := by rfl

theorem prefoldBoundSound : rootBound ≤ prefoldBound := by decide +kernel


theorem prefoldSound :
  preFoldBound rootBound prefoldBound survivorContributions survivorBounds := by
  exact (preFoldSound rootTerms prefoldTerms prefoldResult prefoldBoundSound survivorBoundsSound).2

theorem endResult : endTerms = prefoldTerms := by rfl

theorem endSummaryResult : endSummary = prefoldSummary := by rfl

theorem endSound :
  endTerms = prefoldTerms ∧ endSummary = prefoldSummary := by
  exact ⟨endResult, endSummaryResult⟩

theorem invocationEndClaimSound (env : Env Owner) (actual : Int)
    (claim : ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 env actual (.exact rootTerms rootSummary)) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 env actual (.exact endTerms endSummary) := by
  exact invocationEndSound 100418593683253592432016548326729029359133068138294319235841 env actual rootTerms endTerms rootSummary endSummary
    claim endResult endSummaryResult

theorem selectedRootResultAt : (history.lookup rootResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner rootRaw rootSummary) := by
  rfl

theorem selectedPreFoldAt : (history.lookup prefoldEvent).map AnnotatedEvent.event = some (.preFoldPolynomial rootResultEvent prefoldRaw prefoldSummary (none)) := by
  rfl

theorem selectedInvocationEndAt : (history.lookup endEvent).map AnnotatedEvent.event = some (.invocationEndExact selectedOwner prefoldEvent endRaw endSummary) := by
  rfl

end Bound0

namespace Bound1
def selectedEvent : Nat := 69297
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨19669⟩⟩
def rootResultEvent : Nat := 69295
def prefoldEvent : Nat := 69296
def endEvent : Nat := 69297
def survivorEvents : List Nat := [69281]
def rootRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩] } }]
def prefoldRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩] } }]
def endRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩] } }]
def rootTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩] } }]
def prefoldTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩] } }]
def endTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩] } }]
def rootSummary : Bound := .exactZero
def prefoldSummary : Bound := .exactZero
def endSummary : Bound := .exactZero
def rootBound : Nat := 0
def prefoldBound : Nat := 0
def survivorContributionsChunk0 : List Nat := [1]
def survivorBoundsChunk0 : List Nat := [69280]
theorem survivorBoundsSoundChunk0 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk0 survivorBoundsChunk0 :=
by
  constructor
  · omega
  ·
    exact List.Forall₂.nil

def survivorContributions : List Nat := survivorContributionsChunk0
def survivorBounds : List Nat := survivorBoundsChunk0
theorem survivorBoundsSound : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributions survivorBounds := by
  exact survivorBoundsSoundChunk0

theorem prefoldResult : prefoldTerms = rootTerms := by rfl

theorem prefoldBoundSound : rootBound ≤ prefoldBound := by decide +kernel


theorem prefoldSound :
  preFoldBound rootBound prefoldBound survivorContributions survivorBounds := by
  exact (preFoldSound rootTerms prefoldTerms prefoldResult prefoldBoundSound survivorBoundsSound).2

theorem endResult : endTerms = prefoldTerms := by rfl

theorem endSummaryResult : endSummary = prefoldSummary := by rfl

theorem endSound :
  endTerms = prefoldTerms ∧ endSummary = prefoldSummary := by
  exact ⟨endResult, endSummaryResult⟩

theorem invocationEndClaimSound (env : Env Owner) (actual : Int)
    (claim : ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 env actual (.exact rootTerms rootSummary)) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 env actual (.exact endTerms endSummary) := by
  exact invocationEndSound 100418593683253592432016548326729029359133068138294319235841 env actual rootTerms endTerms rootSummary endSummary
    claim endResult endSummaryResult

theorem selectedRootResultAt : (history.lookup rootResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner rootRaw rootSummary) := by
  rfl

theorem selectedPreFoldAt : (history.lookup prefoldEvent).map AnnotatedEvent.event = some (.preFoldPolynomial rootResultEvent prefoldRaw prefoldSummary (none)) := by
  rfl

theorem selectedInvocationEndAt : (history.lookup endEvent).map AnnotatedEvent.event = some (.invocationEndExact selectedOwner prefoldEvent endRaw endSummary) := by
  rfl

end Bound1

def theoremCount : Nat := 107

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard270
