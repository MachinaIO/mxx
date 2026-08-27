import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard214

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

def shardIndex : Nat := 214
def shardStartEvent : Nat := 54784
def shardEndEvent : Nat := 55040
def rawSemanticCount : Nat := 147
def rawBoundTransferCount : Nat := 70
def rawResultCount : Nat := 65
def rawRelationCount : Nat := 4
def rawSurvivorFoldCount : Nat := 2
def rawPreFoldCount : Nat := 3
def rawInvocationEndCount : Nat := 3
def canonicalWork : Nat := 33

namespace Operation0
def selectedEvent : Nat := 54784
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16185⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6699⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16184⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 54783
def selectedLeftResultEvent : Nat := 54780
def selectedRightResultEvent : Nat := 54777
def selectedResultEvent : Nat := 54784

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 54781 .coefficient, .predecessor 1 54782 .coefficient])) := by
  rfl

end Operation0

namespace Operation1
def selectedEvent : Nat := 54788
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨26229⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨16185⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨26228⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 4
def selectedSumRuleEvent : Nat := 54787
def selectedLeftResultEvent : Nat := 54784
def selectedRightResultEvent : Nat := 54769
def selectedResultEvent : Nat := 54788

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 54785 .coefficient, .predecessor 1 54786 .coefficient])) := by
  rfl

end Operation1

namespace Operation2
def selectedEvent : Nat := 54879
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨21693⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21692⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨21692⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 54877
def selectedLeftResultEvent : Nat := 54874
def selectedRightResultEvent : Nat := 54872
def selectedResultEvent : Nat := 54879
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 54875 .coefficient) (.predecessor 1 54876 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation2

namespace Operation3
def selectedEvent : Nat := 54913
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨14651⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨14650⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11641⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 54911
def selectedLeftResultEvent : Nat := 54908
def selectedRightResultEvent : Nat := 54905
def selectedResultEvent : Nat := 54913
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 54909 .coefficient) (.predecessor 1 54910 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation3

namespace Operation4
def selectedEvent : Nat := 54949
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16224⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16223⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 54947
def selectedLeftResultEvent : Nat := 54944
def selectedRightResultEvent : Nat := 54942
def selectedResultEvent : Nat := 54949
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 54945 .coefficient) (.predecessor 1 54946 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation4

namespace Operation5
def selectedEvent : Nat := 54956
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16225⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6699⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16224⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 54955
def selectedLeftResultEvent : Nat := 54952
def selectedRightResultEvent : Nat := 54949
def selectedResultEvent : Nat := 54956

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 54953 .coefficient, .predecessor 1 54954 .coefficient])) := by
  rfl

end Operation5

namespace Operation6
def selectedEvent : Nat := 54964
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨28314⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28313⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨16225⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨28313⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 2
def selectedSumRuleEvent : Nat := 54959
def selectedLeftResultEvent : Nat := 54956
def selectedRightResultEvent : Nat := 54933
def selectedResultEvent : Nat := 54964
open EventReplay
def leftScalar : Bool := false
def rightScalar : Bool := false
def expected0 : Polynomial Owner := productPoly left right leftScalar rightScalar
def sourceKey0 : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩
def lhsKey0 : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩
def relationRhs0Raw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }]
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 54957 .coefficient) (.predecessor 1 54958 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation6

namespace Operation7
def selectedEvent : Nat := 54972
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18364⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18353⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 54970
def selectedLeftResultEvent : Nat := 54944
def selectedRightResultEvent : Nat := 54967
def selectedResultEvent : Nat := 54972
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 54968 .coefficient) (.predecessor 1 54969 .coefficient) ⟨false, true, none, none, some 1⟩)) := by
  rfl

end Operation7

namespace Operation8
def selectedEvent : Nat := 54979
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18365⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6727⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18364⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 54978
def selectedLeftResultEvent : Nat := 54975
def selectedRightResultEvent : Nat := 54972
def selectedResultEvent : Nat := 54979

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 54976 .coefficient, .predecessor 1 54977 .coefficient])) := by
  rfl

end Operation8

namespace Operation9
def selectedEvent : Nat := 54983
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨28318⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18365⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨28314⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 4
def selectedSumRuleEvent : Nat := 54982
def selectedLeftResultEvent : Nat := 54979
def selectedRightResultEvent : Nat := 54964
def selectedResultEvent : Nat := 54983

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 54980 .coefficient, .predecessor 1 54981 .coefficient])) := by
  rfl

end Operation9

namespace Operation10
def selectedEvent : Nat := 55024
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨11558⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11557⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨11557⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6568⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 55022
def selectedLeftResultEvent : Nat := 2545
def selectedRightResultEvent : Nat := 50670
def selectedResultEvent : Nat := 55024
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.tensor (.predecessor 0 55020 .coefficient) (.predecessor 1 55021 .coefficient) true false)) := by
  rfl

end Operation10

namespace Operation11
def selectedEvent : Nat := 55029
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨7274⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨5545⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6780⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 55027
def selectedLeftResultEvent : Nat := 50540
def selectedRightResultEvent : Nat := 10981
def selectedResultEvent : Nat := 55029
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 55025 .coefficient) (.predecessor 1 55026 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation11

namespace Operation12
def selectedEvent : Nat := 55033
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨11559⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨7274⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11558⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 55032
def selectedLeftResultEvent : Nat := 55029
def selectedRightResultEvent : Nat := 55024
def selectedResultEvent : Nat := 55033

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 55030 .coefficient, .predecessor 1 55031 .coefficient])) := by
  rfl

end Operation12

namespace Relation0
def selectedEvent : Nat := 54962
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨28314⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }]
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

def relationRhsEvent : Nat := 54930
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨24291⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 54963
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28313⟩⟩) ⟨24291⟩ 54930)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.resultExact relationRhsOwner relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28314⟩⟩, .relation 54962 0, ⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24291⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation0

namespace Relation1
def selectedEvent : Nat := 54792
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨19679⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19676⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19676⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19676⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19676⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19676⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19676⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
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

def relationRhsEvent : Nat := 54790
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨26229⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 54796
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19676⟩⟩]⟩) (1) 0 2 (.universal 54791 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19676⟩⟩]⟩) (none) 54790)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.invocationEndExact relationRhsOwner 54789 relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19679⟩⟩, .relation 54792 3, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation1

namespace Relation2
def selectedEvent : Nat := 54812
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨28315⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }]
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

def relationRhsEvent : Nat := 54524
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨24291⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 54813
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28313⟩⟩) ⟨24291⟩ 54524)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.resultExact relationRhsOwner relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28315⟩⟩, .relation 54812 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24291⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation2

namespace Relation3
def selectedEvent : Nat := 54987
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨21695⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
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

def relationRhsEvent : Nat := 54985
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨28318⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 54991
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩) (1) 0 2 (.universal 54986 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩) (none) 54985)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.invocationEndExact relationRhsOwner 54984 relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21695⟩⟩, .relation 54987 3, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation3

namespace Bound0
def selectedEvent : Nat := 54790
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨26229⟩⟩
def rootResultEvent : Nat := 54788
def prefoldEvent : Nat := 54789
def endEvent : Nat := 54790
def survivorEvents : List Nat := []
def rootRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rootTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
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
def selectedEvent : Nat := 54881
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨21693⟩⟩
def rootResultEvent : Nat := 54879
def prefoldEvent : Nat := 54880
def endEvent : Nat := 54881
def survivorEvents : List Nat := [54859]
def rootRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩] } }]
def prefoldRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩] } }]
def endRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩] } }]
def rootTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩] } }]
def prefoldTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩] } }]
def endTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩] } }]
def rootSummary : Bound := .exactZero
def prefoldSummary : Bound := .exactZero
def endSummary : Bound := .exactZero
def rootBound : Nat := 0
def prefoldBound : Nat := 0
def survivorContributionsChunk0 : List Nat := [1]
def survivorBoundsChunk0 : List Nat := [54858]
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
def selectedEvent : Nat := 54985
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨28318⟩⟩
def rootResultEvent : Nat := 54983
def prefoldEvent : Nat := 54984
def endEvent : Nat := 54985
def survivorEvents : List Nat := []
def rootRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rootTerms : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldTerms : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endTerms : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
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

def theoremCount : Nat := 137

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard214
