import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard257

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

def shardIndex : Nat := 257
def shardStartEvent : Nat := 65792
def shardEndEvent : Nat := 66048
def rawSemanticCount : Nat := 146
def rawBoundTransferCount : Nat := 73
def rawResultCount : Nat := 62
def rawRelationCount : Nat := 4
def rawSurvivorFoldCount : Nat := 3
def rawPreFoldCount : Nat := 2
def rawInvocationEndCount : Nat := 2
def canonicalWork : Nat := 28

namespace Operation0
def selectedEvent : Nat := 65793
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨13149⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨13146⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6566⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 65791
def selectedLeftResultEvent : Nat := 3109
def selectedRightResultEvent : Nat := 65295
def selectedResultEvent : Nat := 65793
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.tensor (.predecessor 0 65789 .coefficient) (.predecessor 1 65790 .coefficient) true false)) := by
  rfl

end Operation0

namespace Operation1
def selectedEvent : Nat := 65798
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨7207⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨5533⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6789⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 65796
def selectedLeftResultEvent : Nat := 65165
def selectedRightResultEvent : Nat := 6973
def selectedResultEvent : Nat := 65798
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 65794 .coefficient) (.predecessor 1 65795 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation1

namespace Operation2
def selectedEvent : Nat := 65802
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨13150⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨7207⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨13149⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 65801
def selectedLeftResultEvent : Nat := 65798
def selectedRightResultEvent : Nat := 65793
def selectedResultEvent : Nat := 65802

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 65799 .coefficient, .predecessor 1 65800 .coefficient])) := by
  rfl

end Operation2

namespace Operation3
def selectedEvent : Nat := 65821
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨10236⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨10235⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6566⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 65819
def selectedLeftResultEvent : Nat := 3112
def selectedRightResultEvent : Nat := 65295
def selectedResultEvent : Nat := 65821
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.tensor (.predecessor 0 65817 .coefficient) (.predecessor 1 65818 .coefficient) true false)) := by
  rfl

end Operation3

namespace Operation4
def selectedEvent : Nat := 65826
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨7187⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨5533⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨6769⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 65824
def selectedLeftResultEvent : Nat := 65165
def selectedRightResultEvent : Nat := 7014
def selectedResultEvent : Nat := 65826
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 65822 .coefficient) (.predecessor 1 65823 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation4

namespace Operation5
def selectedEvent : Nat := 65830
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨10237⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨7187⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨10236⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 65829
def selectedLeftResultEvent : Nat := 65826
def selectedRightResultEvent : Nat := 65821
def selectedResultEvent : Nat := 65830

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 65827 .coefficient, .predecessor 1 65828 .coefficient])) := by
  rfl

end Operation5

namespace Operation6
def selectedEvent : Nat := 65921
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨20173⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20172⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨20172⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 65919
def selectedLeftResultEvent : Nat := 65916
def selectedRightResultEvent : Nat := 65914
def selectedResultEvent : Nat := 65921
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 65917 .coefficient) (.predecessor 1 65918 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation6

namespace Operation7
def selectedEvent : Nat := 65955
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨13147⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨10235⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨13146⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 65953
def selectedLeftResultEvent : Nat := 65950
def selectedRightResultEvent : Nat := 65947
def selectedResultEvent : Nat := 65955
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 65951 .coefficient) (.predecessor 1 65952 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation7

namespace Operation8
def selectedEvent : Nat := 65985
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨13248⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨13247⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 65983
def selectedLeftResultEvent : Nat := 65980
def selectedRightResultEvent : Nat := 65978
def selectedResultEvent : Nat := 65985
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 65981 .coefficient) (.predecessor 1 65982 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation8

namespace Operation9
def selectedEvent : Nat := 66008
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨7881⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7879⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6769⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨7880⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 66006
def selectedLeftResultEvent : Nat := 66003
def selectedRightResultEvent : Nat := 66000
def selectedResultEvent : Nat := 66008
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 66004 .coefficient) (.predecessor 1 66005 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation9

namespace Operation10
def selectedEvent : Nat := 66012
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨13249⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨7881⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨13248⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 66011
def selectedLeftResultEvent : Nat := 66008
def selectedRightResultEvent : Nat := 65985
def selectedResultEvent : Nat := 66012

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 66009 .coefficient, .predecessor 1 66010 .coefficient])) := by
  rfl

end Operation10

namespace Operation11
def selectedEvent : Nat := 66020
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨25679⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25676⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨13249⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨25676⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 2
def selectedSumRuleEvent : Nat := 66015
def selectedLeftResultEvent : Nat := 66012
def selectedRightResultEvent : Nat := 65969
def selectedResultEvent : Nat := 66020
open EventReplay
def leftScalar : Bool := false
def rightScalar : Bool := false
def expected0 : Polynomial Owner := productPoly left right leftScalar rightScalar
def sourceKey0 : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩
def lhsKey0 : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩
def relationRhs0Raw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }]
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 66013 .coefficient) (.predecessor 1 66014 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation11

namespace Operation12
def selectedEvent : Nat := 66028
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16869⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16867⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 66026
def selectedLeftResultEvent : Nat := 65980
def selectedRightResultEvent : Nat := 66023
def selectedResultEvent : Nat := 66028
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 66024 .coefficient) (.predecessor 1 66025 .coefficient) ⟨false, true, none, none, some 1⟩)) := by
  rfl

end Operation12

namespace Operation13
def selectedEvent : Nat := 66035
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨16870⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6706⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16869⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 66034
def selectedLeftResultEvent : Nat := 66031
def selectedRightResultEvent : Nat := 66028
def selectedResultEvent : Nat := 66035

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 66032 .coefficient, .predecessor 1 66033 .coefficient])) := by
  rfl

end Operation13

namespace Operation14
def selectedEvent : Nat := 66039
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨25680⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨16870⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨25679⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 4
def selectedSumRuleEvent : Nat := 66038
def selectedLeftResultEvent : Nat := 66035
def selectedRightResultEvent : Nat := 66020
def selectedResultEvent : Nat := 66039

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 66036 .coefficient, .predecessor 1 66037 .coefficient])) := by
  rfl

end Operation14

namespace Relation0
def selectedEvent : Nat := 66018
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨25679⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }]
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

def relationRhsEvent : Nat := 65966
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨23372⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 66019
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25676⟩⟩) ⟨23372⟩ 65966)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.resultExact relationRhsOwner relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25679⟩⟩, .relation 66018 0, ⟨[⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], [⟨.program ⟨214⟩, ⟨23372⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation0

namespace Relation1
def selectedEvent : Nat := 65843
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨10239⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }]
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

def relationRhsEvent : Nat := 6973
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨6789⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 65844
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7879⟩⟩) ⟨6789⟩ 6973)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.resultExact relationRhsOwner relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10239⟩⟩, .relation 65843 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation1

namespace Relation2
def selectedEvent : Nat := 65859
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨25677⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }]
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

def relationRhsEvent : Nat := 65785
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨23372⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 65860
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25676⟩⟩) ⟨23372⟩ 65785)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.resultExact relationRhsOwner relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25677⟩⟩, .relation 65859 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], [⟨.program ⟨214⟩, ⟨23372⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation2

namespace Relation3
def selectedEvent : Nat := 66043
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨20175⟩⟩
open EventReplay
def accumulatorRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩] } }]
def relationRhsRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationOutputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩] } }]
def sourceKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩]⟩
def lhsKey : MonomialKey Owner := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩]⟩
def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact
def relationOutput : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationExpected : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
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

def relationRhsEvent : Nat := 66041
def relationRhsOwner : Owner := ⟨.program ⟨214⟩, ⟨25680⟩⟩
def relationRhsSummary : Bound := .exactZero
def relationOutputEvent : Nat := 66047
theorem selectedRelationAt : (history.lookup selectedEvent).map AnnotatedEvent.event = some (.appliedRelation selectedOwner (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩]⟩) (1) 0 2 (.universal 66042 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩]⟩) (none) 66041)) := by
  rfl

theorem selectedRhsResultAt : (history.lookup relationRhsEvent).map AnnotatedEvent.event = some (.invocationEndExact relationRhsOwner 66040 relationRhsRaw relationRhsSummary) := by
  rfl

theorem selectedRelationOutputAt : (history.lookup relationOutputEvent).map AnnotatedEvent.event = some (.coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20175⟩⟩, .relation 66043 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)) := by
  rfl

end Relation3

namespace Bound0
def selectedEvent : Nat := 65923
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨20173⟩⟩
def rootResultEvent : Nat := 65921
def prefoldEvent : Nat := 65922
def endEvent : Nat := 65923
def survivorEvents : List Nat := [65907]
def rootRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩] } }]
def prefoldRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩] } }]
def endRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩] } }]
def rootTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩] } }]
def prefoldTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩] } }]
def endTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20172⟩⟩] } }]
def rootSummary : Bound := .exactZero
def prefoldSummary : Bound := .exactZero
def endSummary : Bound := .exactZero
def rootBound : Nat := 0
def prefoldBound : Nat := 0
def survivorContributionsChunk0 : List Nat := [1]
def survivorBoundsChunk0 : List Nat := [65906]
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

end Bound0

namespace Bound1
def selectedEvent : Nat := 66041
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨25680⟩⟩
def rootResultEvent : Nat := 66039
def prefoldEvent : Nat := 66040
def endEvent : Nat := 66041
def survivorEvents : List Nat := []
def rootRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rootTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def prefoldTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def endTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25676⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10235⟩⟩, ⟨.program ⟨214⟩, ⟨13146⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23372⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
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

end Bound1

def theoremCount : Nat := 138

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard257
