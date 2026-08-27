import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard004

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

def shardIndex : Nat := 4
def shardStartEvent : Nat := 1024
def shardEndEvent : Nat := 1280
def rawSemanticCount : Nat := 152
def rawBoundTransferCount : Nat := 76
def rawResultCount : Nat := 76
def rawRelationCount : Nat := 0
def rawSurvivorFoldCount : Nat := 0
def rawPreFoldCount : Nat := 0
def rawInvocationEndCount : Nat := 0
def canonicalWork : Nat := 31

namespace Operation0
def selectedEvent : Nat := 1034
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨14669⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14668⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11649⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11649⟩⟩, ⟨.program ⟨214⟩, ⟨14668⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨14668⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11649⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 1032
def selectedLeftResultEvent : Nat := 1029
def selectedRightResultEvent : Nat := 1026
def selectedResultEvent : Nat := 1034
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 1030 .coefficient) (.predecessor 1 1031 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation0

namespace Operation1
def selectedEvent : Nat := 1057
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨14452⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11565⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨14451⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11565⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 1055
def selectedLeftResultEvent : Nat := 1052
def selectedRightResultEvent : Nat := 1049
def selectedResultEvent : Nat := 1057
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 1053 .coefficient) (.predecessor 1 1054 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation1

namespace Operation2
def selectedEvent : Nat := 1080
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨14235⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11481⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨14234⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11481⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 1078
def selectedLeftResultEvent : Nat := 1075
def selectedRightResultEvent : Nat := 1072
def selectedResultEvent : Nat := 1080
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 1076 .coefficient) (.predecessor 1 1077 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation2

namespace Operation3
def selectedEvent : Nat := 1103
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨14018⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14017⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11397⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨14017⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11397⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 1101
def selectedLeftResultEvent : Nat := 1098
def selectedRightResultEvent : Nat := 1095
def selectedResultEvent : Nat := 1103
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 1099 .coefficient) (.predecessor 1 1100 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation3

namespace Operation4
def selectedEvent : Nat := 1126
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨13801⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13800⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11313⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨13800⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11313⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 1124
def selectedLeftResultEvent : Nat := 1121
def selectedRightResultEvent : Nat := 1118
def selectedResultEvent : Nat := 1126
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 1122 .coefficient) (.predecessor 1 1123 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation4

namespace Operation5
def selectedEvent : Nat := 1149
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨13584⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11229⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨13583⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11229⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 1147
def selectedLeftResultEvent : Nat := 1144
def selectedRightResultEvent : Nat := 1141
def selectedResultEvent : Nat := 1149
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 1145 .coefficient) (.predecessor 1 1146 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation5

namespace Operation6
def selectedEvent : Nat := 1172
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨12191⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12190⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11145⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨12190⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11145⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 1170
def selectedLeftResultEvent : Nat := 1167
def selectedRightResultEvent : Nat := 1164
def selectedResultEvent : Nat := 1172
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 1168 .coefficient) (.predecessor 1 1169 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation6

namespace Operation7
def selectedEvent : Nat := 1195
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨11002⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10857⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11001⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10857⟩⟩, ⟨.program ⟨214⟩, ⟨11001⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨10857⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨11001⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 1193
def selectedLeftResultEvent : Nat := 1190
def selectedRightResultEvent : Nat := 1187
def selectedResultEvent : Nat := 1195
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 1191 .coefficient) (.predecessor 1 1192 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation7

namespace Operation8
def selectedEvent : Nat := 1218
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨10701⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9520⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10700⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨9520⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨10700⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 1216
def selectedLeftResultEvent : Nat := 1213
def selectedRightResultEvent : Nat := 1210
def selectedResultEvent : Nat := 1218
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 1214 .coefficient) (.predecessor 1 1215 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation8

namespace Operation9
def selectedEvent : Nat := 1241
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨10505⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9415⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10504⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨9415⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨10504⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 1239
def selectedLeftResultEvent : Nat := 1236
def selectedRightResultEvent : Nat := 1233
def selectedResultEvent : Nat := 1241
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 1237 .coefficient) (.predecessor 1 1238 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation9

namespace Operation10
def selectedEvent : Nat := 1257
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨15323⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨15274⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15322⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 1256
def selectedLeftResultEvent : Nat := 1253
def selectedRightResultEvent : Nat := 1230
def selectedResultEvent : Nat := 1257

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 1254 .coefficient, .predecessor 1 1255 .coefficient])) := by
  rfl

end Operation10

namespace Operation11
def selectedEvent : Nat := 1261
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨15379⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15378⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15378⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨15323⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15378⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 2
def selectedSumRuleEvent : Nat := 1260
def selectedLeftResultEvent : Nat := 1257
def selectedRightResultEvent : Nat := 1207
def selectedResultEvent : Nat := 1261

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 1258 .coefficient, .predecessor 1 1259 .coefficient])) := by
  rfl

end Operation11

namespace Operation12
def selectedEvent : Nat := 1265
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17355⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15378⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17354⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15378⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17354⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨15379⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨17354⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 3
def selectedSumRuleEvent : Nat := 1264
def selectedLeftResultEvent : Nat := 1261
def selectedRightResultEvent : Nat := 1184
def selectedResultEvent : Nat := 1265

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 1262 .coefficient, .predecessor 1 1263 .coefficient])) := by
  rfl

end Operation12

namespace Operation13
def selectedEvent : Nat := 1269
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17356⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15378⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17354⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15638⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15378⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15638⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17354⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17355⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15638⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 4
def selectedSumRuleEvent : Nat := 1268
def selectedLeftResultEvent : Nat := 1265
def selectedRightResultEvent : Nat := 1161
def selectedResultEvent : Nat := 1269

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 1266 .coefficient, .predecessor 1 1267 .coefficient])) := by
  rfl

end Operation13

namespace Operation14
def selectedEvent : Nat := 1273
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17357⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15378⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15638⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17354⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15757⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15378⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15638⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15757⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17354⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17356⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15757⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 5
def selectedSumRuleEvent : Nat := 1272
def selectedLeftResultEvent : Nat := 1269
def selectedRightResultEvent : Nat := 1138
def selectedResultEvent : Nat := 1273

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 1270 .coefficient, .predecessor 1 1271 .coefficient])) := by
  rfl

end Operation14

namespace Operation15
def selectedEvent : Nat := 1277
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨17358⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15378⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15638⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15757⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17354⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15876⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15378⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15638⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15757⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15876⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17354⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨17357⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨15876⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 6
def selectedSumRuleEvent : Nat := 1276
def selectedLeftResultEvent : Nat := 1273
def selectedRightResultEvent : Nat := 1115
def selectedResultEvent : Nat := 1277

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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.sum [.predecessor 0 1274 .coefficient, .predecessor 1 1275 .coefficient])) := by
  rfl

end Operation15

def theoremCount : Nat := 96

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard004
