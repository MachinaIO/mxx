import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard032

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

def shardIndex : Nat := 32
def shardStartEvent : Nat := 8192
def shardEndEvent : Nat := 8448
def rawSemanticCount : Nat := 144
def rawBoundTransferCount : Nat := 69
def rawResultCount : Nat := 63
def rawRelationCount : Nat := 5
def rawSurvivorFoldCount : Nat := 1
def rawPreFoldCount : Nat := 3
def rawInvocationEndCount : Nat := 3
def canonicalWork : Nat := 34

namespace Operation0
def selectedEvent : Nat := 8203
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨7875⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7873⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6767⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨7874⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 8201
def selectedLeftResultEvent : Nat := 8198
def selectedRightResultEvent : Nat := 8195
def selectedResultEvent : Nat := 8203
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 8199 .coefficient) (.predecessor 1 8200 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation0

namespace Operation1
def selectedEvent : Nat := 8207
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨12877⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨7875⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨12876⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 8206
def selectedLeftResultEvent : Nat := 8203
def selectedRightResultEvent : Nat := 8180
def selectedResultEvent : Nat := 8207

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 8204 .coefficient, .predecessor 1 8205 .coefficient])) := by
  rfl

end Operation1

namespace Operation2
def selectedEvent : Nat := 8215
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨25550⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25547⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨12877⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨25547⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 2
def selectedSumRuleEvent : Nat := 8210
def selectedLeftResultEvent : Nat := 8207
def selectedRightResultEvent : Nat := 8164
def selectedResultEvent : Nat := 8215
open EventReplay
def leftScalar : Bool := false
def rightScalar : Bool := false
def expected0 : Polynomial Owner := productPoly left right leftScalar rightScalar
def sourceKey0 : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩]⟩
def lhsKey0 : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩]⟩
def relationRhs0Raw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }]
def relationRhs0 : Polynomial Owner := relationRhs0Raw.map Term.toExact
def relationContext0 : MonomialContext Owner := relationContext sourceKey0 sourceKey0.centralFactors 0 2
def expected1 : Polynomial Owner := relationPoly expected0 sourceKey0 relationContext0 (-1) relationRhs0

theorem productAgreement : CanonicalAgreement expected0 (productPoly left right leftScalar rightScalar) := by
  decide +kernel

theorem resultAgreement : CanonicalAgreement output expected1 := by
  decide +kernel

theorem resultSound (env : Env Owner)
    (baseRelation0 : evalMonomial env lhsKey0 % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 = evalPolynomial env relationRhs0 % Int.ofNat 100418593683253592432016548326729029359133068138294319235841)
    : evalPolynomial env output % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 =
      (evalPolynomial env left * evalPolynomial env right) % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 := by
  have productSound := productCanonicalResultSound env left right expected0 leftScalar rightScalar productAgreement
  have relationSound0 := relationCanonicalResultSound 100418593683253592432016548326729029359133068138294319235841 env expected0 sourceKey0 lhsKey0 sourceKey0.centralFactors 0 2 (-1) relationRhs0 expected1 (by decide +kernel) baseRelation0 (by decide +kernel)
  have outputSound := canonicalAgreement_eval env output expected1 resultAgreement
  calc
    evalPolynomial env output % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 = evalPolynomial env expected1 % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 := by rw [outputSound]
    _ = evalPolynomial env expected0 % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 := relationSound0
    _ = (evalPolynomial env left * evalPolynomial env right) % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 := by rw [productSound]

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 8208 .coefficient) (.predecessor 1 8209 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation2

namespace Operation3
def selectedEvent : Nat := 8223
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16651⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16649⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 8221
def selectedLeftResultEvent : Nat := 8175
def selectedRightResultEvent : Nat := 8218
def selectedResultEvent : Nat := 8223
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 8219 .coefficient) (.predecessor 1 8220 .coefficient) ⟨false, true, none, none, some 1⟩)) := by
  rfl

end Operation3

namespace Operation4
def selectedEvent : Nat := 8230
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16652⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6704⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16651⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 8229
def selectedLeftResultEvent : Nat := 8226
def selectedRightResultEvent : Nat := 8223
def selectedResultEvent : Nat := 8230

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 8227 .coefficient, .predecessor 1 8228 .coefficient])) := by
  rfl

end Operation4

namespace Operation5
def selectedEvent : Nat := 8234
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨25551⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨16652⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨25550⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 4
def selectedSumRuleEvent : Nat := 8233
def selectedLeftResultEvent : Nat := 8230
def selectedRightResultEvent : Nat := 8215
def selectedResultEvent : Nat := 8234

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 8231 .coefficient, .predecessor 1 8232 .coefficient])) := by
  rfl

end Operation5

namespace Operation6
def selectedEvent : Nat := 8325
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨22425⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22424⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22424⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨22424⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 8323
def selectedLeftResultEvent : Nat := 8320
def selectedRightResultEvent : Nat := 8318
def selectedResultEvent : Nat := 8325
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 8321 .coefficient) (.predecessor 1 8322 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation6

namespace Operation7
def selectedEvent : Nat := 8359
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨12795⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨10050⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨12794⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 8357
def selectedLeftResultEvent : Nat := 8354
def selectedRightResultEvent : Nat := 8351
def selectedResultEvent : Nat := 8359
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 8355 .coefficient) (.predecessor 1 8356 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation7

namespace Operation8
def selectedEvent : Nat := 8395
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16726⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16725⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 8393
def selectedLeftResultEvent : Nat := 8390
def selectedRightResultEvent : Nat := 8388
def selectedResultEvent : Nat := 8395
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 8391 .coefficient) (.predecessor 1 8392 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation8

namespace Operation9
def selectedEvent : Nat := 8402
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16727⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6704⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16726⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 8401
def selectedLeftResultEvent : Nat := 8398
def selectedRightResultEvent : Nat := 8395
def selectedResultEvent : Nat := 8402

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 8399 .coefficient, .predecessor 1 8400 .coefficient])) := by
  rfl

end Operation9

namespace Operation10
def selectedEvent : Nat := 8410
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨29438⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29437⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨16727⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨29437⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 2
def selectedSumRuleEvent : Nat := 8405
def selectedLeftResultEvent : Nat := 8402
def selectedRightResultEvent : Nat := 8379
def selectedResultEvent : Nat := 8410
open EventReplay
def leftScalar : Bool := false
def rightScalar : Bool := false
def expected0 : Polynomial Owner := productPoly left right leftScalar rightScalar
def sourceKey0 : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨16649⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩]⟩
def lhsKey0 : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩]⟩
def relationRhs0Raw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }]
def relationRhs0 : Polynomial Owner := relationRhs0Raw.map Term.toExact
def relationContext0 : MonomialContext Owner := relationContext sourceKey0 sourceKey0.centralFactors 0 2
def expected1 : Polynomial Owner := relationPoly expected0 sourceKey0 relationContext0 (-1) relationRhs0

theorem productAgreement : CanonicalAgreement expected0 (productPoly left right leftScalar rightScalar) := by
  decide +kernel

theorem resultAgreement : CanonicalAgreement output expected1 := by
  decide +kernel

theorem resultSound (env : Env Owner)
    (baseRelation0 : evalMonomial env lhsKey0 % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 = evalPolynomial env relationRhs0 % Int.ofNat 100418593683253592432016548326729029359133068138294319235841)
    : evalPolynomial env output % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 =
      (evalPolynomial env left * evalPolynomial env right) % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 := by
  have productSound := productCanonicalResultSound env left right expected0 leftScalar rightScalar productAgreement
  have relationSound0 := relationCanonicalResultSound 100418593683253592432016548326729029359133068138294319235841 env expected0 sourceKey0 lhsKey0 sourceKey0.centralFactors 0 2 (-1) relationRhs0 expected1 (by decide +kernel) baseRelation0 (by decide +kernel)
  have outputSound := canonicalAgreement_eval env output expected1 resultAgreement
  calc
    evalPolynomial env output % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 = evalPolynomial env expected1 % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 := by rw [outputSound]
    _ = evalPolynomial env expected0 % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 := relationSound0
    _ = (evalPolynomial env left * evalPolynomial env right) % Int.ofNat 100418593683253592432016548326729029359133068138294319235841 := by rw [productSound]

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 8403 .coefficient) (.predecessor 1 8404 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation10

namespace Operation11
def selectedEvent : Nat := 8418
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16692⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16691⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 8416
def selectedLeftResultEvent : Nat := 8390
def selectedRightResultEvent : Nat := 8413
def selectedResultEvent : Nat := 8418
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 8414 .coefficient) (.predecessor 1 8415 .coefficient) ⟨false, true, none, none, some 1⟩)) := by
  rfl

end Operation11

namespace Operation12
def selectedEvent : Nat := 8425
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16693⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6737⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16692⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 8424
def selectedLeftResultEvent : Nat := 8421
def selectedRightResultEvent : Nat := 8418
def selectedResultEvent : Nat := 8425

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 8422 .coefficient, .predecessor 1 8423 .coefficient])) := by
  rfl

end Operation12

namespace Operation13
def selectedEvent : Nat := 8429
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨29442⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨16693⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨29438⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 4
def selectedSumRuleEvent : Nat := 8428
def selectedLeftResultEvent : Nat := 8425
def selectedRightResultEvent : Nat := 8410
def selectedResultEvent : Nat := 8429

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 8426 .coefficient, .predecessor 1 8427 .coefficient])) := by
  rfl

end Operation13

namespace Relation0
def selectedEvent : Nat := 8212
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨25550⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }]
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

def relationRhsEvent : Nat := 8161
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨23298⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 8213
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25547⟩⟩) ⟨23298⟩ 8161)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.resultExact relationRhsOwner relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25550⟩⟩, .relation 8212 0, ⟨[⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], [⟨.program ⟨214⟩, ⟨23298⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation0

namespace Relation1
def selectedEvent : Nat := 8407
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨29438⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨16649⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }]
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

def relationRhsEvent : Nat := 8376
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨24615⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 8408
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨16649⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29437⟩⟩) ⟨24615⟩ 8376)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.resultExact relationRhsOwner relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29438⟩⟩, .relation 8407 0, ⟨[⟨.program ⟨214⟩, ⟨16649⟩⟩], [⟨.program ⟨214⟩, ⟨24615⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation1

namespace Relation2
def selectedEvent : Nat := 8238
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨20051⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20048⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20048⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20048⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20048⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20048⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20048⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
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

def relationRhsEvent : Nat := 8236
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨25551⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 8242
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20048⟩⟩]⟩) (1) 0 2 (.universal 8237 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20048⟩⟩]⟩) (none) 8236)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.invocationEndExact relationRhsOwner 8235 relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20051⟩⟩, .relation 8238 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩)) := by
  rfl

end Relation2

namespace Relation3
def selectedEvent : Nat := 8257
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨29439⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }]
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

def relationRhsEvent : Nat := 7951
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨24615⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 8258
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29437⟩⟩) ⟨24615⟩ 7951)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.resultExact relationRhsOwner relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29439⟩⟩, .relation 8257 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], [⟨.program ⟨214⟩, ⟨24615⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation3

namespace Relation4
def selectedEvent : Nat := 8433
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨22427⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22424⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22424⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22424⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22424⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22424⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22424⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
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

def relationRhsEvent : Nat := 8431
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨29442⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 8437
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22424⟩⟩]⟩) (1) 0 2 (.universal 8432 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22424⟩⟩]⟩) (none) 8431)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.invocationEndExact relationRhsOwner 8430 relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22427⟩⟩, .relation 8433 1, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩)) := by
  rfl

end Relation4

namespace Bound0
def selectedEvent : Nat := 8236
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨25551⟩⟩
def rootResultEvent : Nat := 8234
def prefoldEvent : Nat := 8235
def endEvent : Nat := 8236
def survivorEvents : List Nat := []
def rootRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rootTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25547⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10050⟩⟩, ⟨.program ⟨214⟩, ⟨12794⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23298⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
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
def selectedEvent : Nat := 8327
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨22425⟩⟩
def rootResultEvent : Nat := 8325
def prefoldEvent : Nat := 8326
def endEvent : Nat := 8327
def survivorEvents : List Nat := [8305]
def rootRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22424⟩⟩] } }]
def prefoldRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22424⟩⟩] } }]
def endRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22424⟩⟩] } }]
def rootTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22424⟩⟩] } }]
def prefoldTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22424⟩⟩] } }]
def endTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22424⟩⟩] } }]
def rootSummary : Bound := .exactZero
def prefoldSummary : Bound := .exactZero
def endSummary : Bound := .exactZero
def rootBound : Nat := 0
def prefoldBound : Nat := 0
def survivorContributionsChunk0 : List Nat := [1]
def survivorBoundsChunk0 : List Nat := [8304]
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

namespace Bound2
def selectedEvent : Nat := 8431
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨29442⟩⟩
def rootResultEvent : Nat := 8429
def prefoldEvent : Nat := 8430
def endEvent : Nat := 8431
def survivorEvents : List Nat := []
def rootRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rootTerms : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldTerms : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endTerms : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16691⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
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

end Bound2

def theoremCount : Nat := 150

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard032
