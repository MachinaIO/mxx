import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard408

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

def shardIndex : Nat := 408
def shardStartEvent : Nat := 104448
def shardEndEvent : Nat := 104704
def rawSemanticCount : Nat := 144
def rawBoundTransferCount : Nat := 68
def rawResultCount : Nat := 63
def rawRelationCount : Nat := 6
def rawSurvivorFoldCount : Nat := 1
def rawPreFoldCount : Nat := 3
def rawInvocationEndCount : Nat := 3
def canonicalWork : Nat := 39

namespace Operation0
def selectedEvent : Nat := 104471
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16702⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16701⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 104469
def selectedLeftResultEvent : Nat := 104466
def selectedRightResultEvent : Nat := 104464
def selectedResultEvent : Nat := 104471
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 104467 .coefficient) (.predecessor 1 104468 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation0

namespace Operation1
def selectedEvent : Nat := 104478
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16703⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6704⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16702⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 104477
def selectedLeftResultEvent : Nat := 104474
def selectedRightResultEvent : Nat := 104471
def selectedResultEvent : Nat := 104478

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 104475 .coefficient, .predecessor 1 104476 .coefficient])) := by
  rfl

end Operation1

namespace Operation2
def selectedEvent : Nat := 104486
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨29344⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29343⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨16703⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨29343⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 2
def selectedSumRuleEvent : Nat := 104481
def selectedLeftResultEvent : Nat := 104478
def selectedRightResultEvent : Nat := 104455
def selectedResultEvent : Nat := 104486
open EventReplay
def leftScalar : Bool := false
def rightScalar : Bool := false
def expected0 : Polynomial Owner := productPoly left right leftScalar rightScalar
def sourceKey0 : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨16623⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩]⟩
def lhsKey0 : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩]⟩
def relationRhs0Raw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }]
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 104479 .coefficient) (.predecessor 1 104480 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation2

namespace Operation3
def selectedEvent : Nat := 104494
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17710⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17708⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 104492
def selectedLeftResultEvent : Nat := 104466
def selectedRightResultEvent : Nat := 104489
def selectedResultEvent : Nat := 104494
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 104490 .coefficient) (.predecessor 1 104491 .coefficient) ⟨false, true, none, none, some 1⟩)) := by
  rfl

end Operation3

namespace Operation4
def selectedEvent : Nat := 104501
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17711⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6736⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17710⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 104500
def selectedLeftResultEvent : Nat := 104497
def selectedRightResultEvent : Nat := 104494
def selectedResultEvent : Nat := 104501

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 104498 .coefficient, .predecessor 1 104499 .coefficient])) := by
  rfl

end Operation4

namespace Operation5
def selectedEvent : Nat := 104505
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨29349⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17711⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨29344⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 4
def selectedSumRuleEvent : Nat := 104504
def selectedLeftResultEvent : Nat := 104501
def selectedRightResultEvent : Nat := 104486
def selectedResultEvent : Nat := 104505

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 104502 .coefficient, .predecessor 1 104503 .coefficient])) := by
  rfl

end Operation5

namespace Operation6
def selectedEvent : Nat := 104601
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨22182⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22181⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22181⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨22181⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 104599
def selectedLeftResultEvent : Nat := 104596
def selectedRightResultEvent : Nat := 104594
def selectedResultEvent : Nat := 104601
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 104597 .coefficient) (.predecessor 1 104598 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation6

namespace Operation7
def selectedEvent : Nat := 104623
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨12543⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9910⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12542⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨9910⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨12542⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 104621
def selectedLeftResultEvent : Nat := 104618
def selectedRightResultEvent : Nat := 104615
def selectedResultEvent : Nat := 104623
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 104619 .coefficient) (.predecessor 1 104620 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation7

namespace Operation8
def selectedEvent : Nat := 104659
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16583⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16582⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 104657
def selectedLeftResultEvent : Nat := 104654
def selectedRightResultEvent : Nat := 104652
def selectedResultEvent : Nat := 104659
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 104655 .coefficient) (.predecessor 1 104656 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation8

namespace Operation9
def selectedEvent : Nat := 104666
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16584⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6703⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16583⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 104665
def selectedLeftResultEvent : Nat := 104662
def selectedRightResultEvent : Nat := 104659
def selectedResultEvent : Nat := 104666

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 104663 .coefficient, .predecessor 1 104664 .coefficient])) := by
  rfl

end Operation9

namespace Operation10
def selectedEvent : Nat := 104674
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨29127⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29126⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨16584⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨29126⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 2
def selectedSumRuleEvent : Nat := 104669
def selectedLeftResultEvent : Nat := 104666
def selectedRightResultEvent : Nat := 104643
def selectedResultEvent : Nat := 104674
open EventReplay
def leftScalar : Bool := false
def rightScalar : Bool := false
def expected0 : Polynomial Owner := productPoly left right leftScalar rightScalar
def sourceKey0 : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩]⟩
def lhsKey0 : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩]⟩
def relationRhs0Raw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }]
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 104667 .coefficient) (.predecessor 1 104668 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation10

namespace Operation11
def selectedEvent : Nat := 104682
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17941⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17939⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 104680
def selectedLeftResultEvent : Nat := 104654
def selectedRightResultEvent : Nat := 104677
def selectedResultEvent : Nat := 104682
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 104678 .coefficient) (.predecessor 1 104679 .coefficient) ⟨false, true, none, none, some 1⟩)) := by
  rfl

end Operation11

namespace Operation12
def selectedEvent : Nat := 104689
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17942⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6734⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17941⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 104688
def selectedLeftResultEvent : Nat := 104685
def selectedRightResultEvent : Nat := 104682
def selectedResultEvent : Nat := 104689

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 104686 .coefficient, .predecessor 1 104687 .coefficient])) := by
  rfl

end Operation12

namespace Operation13
def selectedEvent : Nat := 104693
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨29132⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17942⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨29127⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 4
def selectedSumRuleEvent : Nat := 104692
def selectedLeftResultEvent : Nat := 104689
def selectedRightResultEvent : Nat := 104674
def selectedResultEvent : Nat := 104693

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 104690 .coefficient, .predecessor 1 104691 .coefficient])) := by
  rfl

end Operation13

namespace Relation0
def selectedEvent : Nat := 104484
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨29344⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨16623⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }]
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

def relationRhsEvent : Nat := 104452
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨24593⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 104485
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨16623⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29343⟩⟩) ⟨24593⟩ 104452)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.resultExact relationRhsOwner relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29344⟩⟩, .relation 104484 0, ⟨[⟨.program ⟨214⟩, ⟨16623⟩⟩], [⟨.program ⟨214⟩, ⟨24593⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation0

namespace Relation1
def selectedEvent : Nat := 104672
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨29127⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }]
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

def relationRhsEvent : Nat := 104640
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨24530⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 104673
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29126⟩⟩) ⟨24530⟩ 104640)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.resultExact relationRhsOwner relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29127⟩⟩, .relation 104672 0, ⟨[⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨24530⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation1

namespace Relation2
def selectedEvent : Nat := 104509
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨22328⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22325⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22325⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22325⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22325⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22325⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22325⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
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

def relationRhsEvent : Nat := 104507
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨29349⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 104513
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22325⟩⟩]⟩) (1) 0 2 (.universal 104508 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22325⟩⟩]⟩) (none) 104507)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.invocationEndExact relationRhsOwner 104506 relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22328⟩⟩, .relation 104509 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation2

namespace Relation3
def selectedEvent : Nat := 104529
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨29347⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
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

def relationRhsEvent : Nat := 5572
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨6604⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 104530
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6665⟩⟩) ⟨6604⟩ 5572)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.resultExact relationRhsOwner relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29347⟩⟩, .relation 104529 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17708⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation3

namespace Relation4
def selectedEvent : Nat := 104546
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨29128⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }]
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

def relationRhsEvent : Nat := 104535
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨24530⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 104547
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29126⟩⟩) ⟨24530⟩ 104535)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.resultExact relationRhsOwner relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29128⟩⟩, .relation 104546 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], [⟨.program ⟨214⟩, ⟨24530⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation4

namespace Relation5
def selectedEvent : Nat := 104697
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨22184⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22181⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22181⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22181⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22181⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22181⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22181⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
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

def relationRhsEvent : Nat := 104695
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨29132⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 104701
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22181⟩⟩]⟩) (1) 0 2 (.universal 104696 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22181⟩⟩]⟩) (none) 104695)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.invocationEndExact relationRhsOwner 104694 relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22184⟩⟩, .relation 104697 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17939⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation5

namespace Bound0
def selectedEvent : Nat := 104507
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨29349⟩⟩
def rootResultEvent : Nat := 104505
def prefoldEvent : Nat := 104506
def endEvent : Nat := 104507
def survivorEvents : List Nat := []
def rootRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rootTerms : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldTerms : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endTerms : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29343⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24593⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
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
def selectedEvent : Nat := 104603
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨22182⟩⟩
def rootResultEvent : Nat := 104601
def prefoldEvent : Nat := 104602
def endEvent : Nat := 104603
def survivorEvents : List Nat := [104581]
def rootRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22181⟩⟩] } }]
def prefoldRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22181⟩⟩] } }]
def endRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22181⟩⟩] } }]
def rootTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22181⟩⟩] } }]
def prefoldTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22181⟩⟩] } }]
def endTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22181⟩⟩] } }]
def rootSummary : Bound := .exactZero
def prefoldSummary : Bound := .exactZero
def endSummary : Bound := .exactZero
def rootBound : Nat := 0
def prefoldBound : Nat := 0
def survivorContributionsChunk0 : List Nat := [1]
def survivorBoundsChunk0 : List Nat := [104580]
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
def selectedEvent : Nat := 104695
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨29132⟩⟩
def rootResultEvent : Nat := 104693
def prefoldEvent : Nat := 104694
def endEvent : Nat := 104695
def survivorEvents : List Nat := []
def rootRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rootTerms : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldTerms : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endTerms : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29126⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16539⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24530⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
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

def theoremCount : Nat := 156

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard408
