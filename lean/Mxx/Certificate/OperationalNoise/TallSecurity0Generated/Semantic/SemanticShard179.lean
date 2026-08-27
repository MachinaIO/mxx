import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard179

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

def shardIndex : Nat := 179
def shardStartEvent : Nat := 45824
def shardEndEvent : Nat := 46080
def rawSemanticCount : Nat := 141
def rawBoundTransferCount : Nat := 70
def rawResultCount : Nat := 71
def rawRelationCount : Nat := 0
def rawSurvivorFoldCount : Nat := 0
def rawPreFoldCount : Nat := 0
def rawInvocationEndCount : Nat := 0
def canonicalWork : Nat := 175

namespace Operation0
def selectedEvent : Nat := 45831
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨12182⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11141⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨12181⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11141⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 45829
def selectedLeftResultEvent : Nat := 45826
def selectedRightResultEvent : Nat := 45823
def selectedResultEvent : Nat := 45831
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 45827 .coefficient) (.predecessor 1 45828 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation0

namespace Operation1
def selectedEvent : Nat := 45854
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨10994⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10852⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10993⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨10852⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨10993⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 45852
def selectedLeftResultEvent : Nat := 45849
def selectedRightResultEvent : Nat := 45846
def selectedResultEvent : Nat := 45854
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 45850 .coefficient) (.predecessor 1 45851 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation1

namespace Operation2
def selectedEvent : Nat := 45877
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨10693⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9515⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10692⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨9515⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨10692⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 45875
def selectedLeftResultEvent : Nat := 45872
def selectedRightResultEvent : Nat := 45869
def selectedResultEvent : Nat := 45877
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 45873 .coefficient) (.predecessor 1 45874 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation2

namespace Operation3
def selectedEvent : Nat := 45900
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨10497⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9410⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨9410⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨10496⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 45898
def selectedLeftResultEvent : Nat := 45895
def selectedRightResultEvent : Nat := 45892
def selectedResultEvent : Nat := 45900
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 45896 .coefficient) (.predecessor 1 45897 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation3

namespace Operation4
def selectedEvent : Nat := 45916
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨15319⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨15271⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15318⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 45915
def selectedLeftResultEvent : Nat := 45912
def selectedRightResultEvent : Nat := 45889
def selectedResultEvent : Nat := 45916

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45913 .coefficient, .predecessor 1 45914 .coefficient])) := by
  rfl

end Operation4

namespace Operation5
def selectedEvent : Nat := 45920
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨15375⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨15319⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15374⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 2
def selectedSumRuleEvent : Nat := 45919
def selectedLeftResultEvent : Nat := 45916
def selectedRightResultEvent : Nat := 45866
def selectedResultEvent : Nat := 45920

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45917 .coefficient, .predecessor 1 45918 .coefficient])) := by
  rfl

end Operation5

namespace Operation6
def selectedEvent : Nat := 45924
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17346⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨15375⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17345⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 3
def selectedSumRuleEvent : Nat := 45923
def selectedLeftResultEvent : Nat := 45920
def selectedRightResultEvent : Nat := 45843
def selectedResultEvent : Nat := 45924

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45921 .coefficient, .predecessor 1 45922 .coefficient])) := by
  rfl

end Operation6

namespace Operation7
def selectedEvent : Nat := 45928
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17347⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17346⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15635⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 4
def selectedSumRuleEvent : Nat := 45927
def selectedLeftResultEvent : Nat := 45924
def selectedRightResultEvent : Nat := 45820
def selectedResultEvent : Nat := 45928

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45925 .coefficient, .predecessor 1 45926 .coefficient])) := by
  rfl

end Operation7

namespace Operation8
def selectedEvent : Nat := 45932
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17348⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17347⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15754⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 5
def selectedSumRuleEvent : Nat := 45931
def selectedLeftResultEvent : Nat := 45928
def selectedRightResultEvent : Nat := 45797
def selectedResultEvent : Nat := 45932

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45929 .coefficient, .predecessor 1 45930 .coefficient])) := by
  rfl

end Operation8

namespace Operation9
def selectedEvent : Nat := 45936
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17349⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17348⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15873⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 6
def selectedSumRuleEvent : Nat := 45935
def selectedLeftResultEvent : Nat := 45932
def selectedRightResultEvent : Nat := 45774
def selectedResultEvent : Nat := 45936

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45933 .coefficient, .predecessor 1 45934 .coefficient])) := by
  rfl

end Operation9

namespace Operation10
def selectedEvent : Nat := 45940
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17350⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17349⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15992⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 7
def selectedSumRuleEvent : Nat := 45939
def selectedLeftResultEvent : Nat := 45936
def selectedRightResultEvent : Nat := 45751
def selectedResultEvent : Nat := 45940

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45937 .coefficient, .predecessor 1 45938 .coefficient])) := by
  rfl

end Operation10

namespace Operation11
def selectedEvent : Nat := 45944
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17351⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17350⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16111⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 8
def selectedSumRuleEvent : Nat := 45943
def selectedLeftResultEvent : Nat := 45940
def selectedRightResultEvent : Nat := 45728
def selectedResultEvent : Nat := 45944

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45941 .coefficient, .predecessor 1 45942 .coefficient])) := by
  rfl

end Operation11

namespace Operation12
def selectedEvent : Nat := 45948
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18367⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17351⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18366⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 9
def selectedSumRuleEvent : Nat := 45947
def selectedLeftResultEvent : Nat := 45944
def selectedRightResultEvent : Nat := 45705
def selectedResultEvent : Nat := 45948

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45945 .coefficient, .predecessor 1 45946 .coefficient])) := by
  rfl

end Operation12

namespace Operation13
def selectedEvent : Nat := 45952
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18368⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18367⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16314⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 10
def selectedSumRuleEvent : Nat := 45951
def selectedLeftResultEvent : Nat := 45948
def selectedRightResultEvent : Nat := 45682
def selectedResultEvent : Nat := 45952

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45949 .coefficient, .predecessor 1 45950 .coefficient])) := by
  rfl

end Operation13

namespace Operation14
def selectedEvent : Nat := 45956
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18369⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18368⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17126⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 11
def selectedSumRuleEvent : Nat := 45955
def selectedLeftResultEvent : Nat := 45952
def selectedRightResultEvent : Nat := 45659
def selectedResultEvent : Nat := 45956

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45953 .coefficient, .predecessor 1 45954 .coefficient])) := by
  rfl

end Operation14

namespace Operation15
def selectedEvent : Nat := 45960
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18370⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18369⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17910⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 12
def selectedSumRuleEvent : Nat := 45959
def selectedLeftResultEvent : Nat := 45956
def selectedRightResultEvent : Nat := 45636
def selectedResultEvent : Nat := 45960

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45957 .coefficient, .predecessor 1 45958 .coefficient])) := by
  rfl

end Operation15

namespace Operation16
def selectedEvent : Nat := 45964
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18371⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18370⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18211⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 13
def selectedSumRuleEvent : Nat := 45963
def selectedLeftResultEvent : Nat := 45960
def selectedRightResultEvent : Nat := 45613
def selectedResultEvent : Nat := 45964

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45961 .coefficient, .predecessor 1 45962 .coefficient])) := by
  rfl

end Operation16

namespace Operation17
def selectedEvent : Nat := 45968
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18372⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16685⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16685⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18371⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16685⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 14
def selectedSumRuleEvent : Nat := 45967
def selectedLeftResultEvent : Nat := 45964
def selectedRightResultEvent : Nat := 45590
def selectedResultEvent : Nat := 45968

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45965 .coefficient, .predecessor 1 45966 .coefficient])) := by
  rfl

end Operation17

namespace Operation18
def selectedEvent : Nat := 45972
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18373⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16685⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16804⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16685⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16804⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18372⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨16804⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 15
def selectedSumRuleEvent : Nat := 45971
def selectedLeftResultEvent : Nat := 45968
def selectedRightResultEvent : Nat := 45567
def selectedResultEvent : Nat := 45972

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45969 .coefficient, .predecessor 1 45970 .coefficient])) := by
  rfl

end Operation18

namespace Operation19
def selectedEvent : Nat := 45976
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18374⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16685⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16804⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17091⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16685⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16804⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17091⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18373⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17091⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 16
def selectedSumRuleEvent : Nat := 45975
def selectedLeftResultEvent : Nat := 45972
def selectedRightResultEvent : Nat := 45544
def selectedResultEvent : Nat := 45976

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45973 .coefficient, .predecessor 1 45974 .coefficient])) := by
  rfl

end Operation19

namespace Operation20
def selectedEvent : Nat := 45980
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18375⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16685⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16804⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17091⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18176⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16685⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16804⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17091⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18176⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨18374⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18176⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 17
def selectedSumRuleEvent : Nat := 45979
def selectedLeftResultEvent : Nat := 45976
def selectedRightResultEvent : Nat := 45521
def selectedResultEvent : Nat := 45980

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 45977 .coefficient, .predecessor 1 45978 .coefficient])) := by
  rfl

end Operation20

namespace Operation21
def selectedEvent : Nat := 46027
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18657⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16685⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16804⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17091⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18176⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15318⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16685⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16804⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17091⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18176⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18656⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 18
def selectedSumRuleEvent : Nat := 46008
def selectedLeftResultEvent : Nat := 46005
def selectedRightResultEvent : Nat := 46003
def selectedResultEvent : Nat := 46027
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 46006 .coefficient) (.predecessor 1 46007 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation21

def theoremCount : Nat := 132

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard179
