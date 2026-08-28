import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard363
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard052
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard337
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard341
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard342
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard344
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard345
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard347
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard348
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard349
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard351
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard352
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard362

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult50225
def owner : Owner := ⟨.program ⟨214⟩, ⟨27890⟩⟩
def rawTerms : List Term := Proof.Events196.exact50225RawTerms
def summary : Bound := (.finite 37932293507469318446662025268)
def resultEvent : Nat := 50225
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50225.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50220.owner)
    (rightOwner := SemanticResult48674.owner)
    (leftResult := 50220) (rightResult := 48674)
    (leftActual := SemanticResult50220.actual selector witness)
    (rightActual := SemanticResult48674.actual selector witness)
    (leftRaw := SemanticResult50220.rawTerms)
    (rightRaw := SemanticResult48674.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 33190381535015453579890655284)
    (rightMaximum := 4741911972453864866771369984) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50221) (rightBinding := 50222)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27673⟩) (rightExpression := ⟨27889⟩)
    (transferEvent := 50223) (summaryTransferEvent := 50224)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50220.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult48674.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50225

namespace SemanticResult50230
def owner : Owner := ⟨.program ⟨214⟩, ⟨28107⟩⟩
def rawTerms : List Term := Proof.Events196.exact50230RawTerms
def summary : Bound := (.finite 42674369987986832655214706740)
def resultEvent : Nat := 50230
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50230.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50225.owner)
    (rightOwner := SemanticResult48462.owner)
    (leftResult := 50225) (rightResult := 48462)
    (leftActual := SemanticResult50225.actual selector witness)
    (rightActual := SemanticResult48462.actual selector witness)
    (leftRaw := SemanticResult50225.rawTerms)
    (rightRaw := SemanticResult48462.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 37932293507469318446662025268)
    (rightMaximum := 4742076480517514208552681472) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50226) (rightBinding := 50227)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27890⟩) (rightExpression := ⟨28106⟩)
    (transferEvent := 50228) (summaryTransferEvent := 50229)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50225.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult48462.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50230

namespace SemanticResult50235
def owner : Owner := ⟨.program ⟨214⟩, ⟨28324⟩⟩
def rawTerms : List Term := Proof.Events196.exact50235RawTerms
def summary : Bound := (.finite 47416693230599820876439355444)
def resultEvent : Nat := 50235
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50235.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50230.owner)
    (rightOwner := SemanticResult48250.owner)
    (leftResult := 50230) (rightResult := 48250)
    (leftActual := SemanticResult50230.actual selector witness)
    (rightActual := SemanticResult48250.actual selector witness)
    (leftRaw := SemanticResult50230.rawTerms)
    (rightRaw := SemanticResult48250.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 42674369987986832655214706740)
    (rightMaximum := 4742323242612988221224648704) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50231) (rightBinding := 50232)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28107⟩) (rightExpression := ⟨28323⟩)
    (transferEvent := 50233) (summaryTransferEvent := 50234)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50230.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult48250.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50235

namespace SemanticResult50240
def owner : Owner := ⟨.program ⟨214⟩, ⟨28541⟩⟩
def rawTerms : List Term := Proof.Events196.exact50240RawTerms
def summary : Bound := (.finite 52159098727244633768554659892)
def resultEvent : Nat := 50240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50240.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50235.owner)
    (rightOwner := SemanticResult48038.owner)
    (leftResult := 50235) (rightResult := 48038)
    (leftActual := SemanticResult50235.actual selector witness)
    (rightActual := SemanticResult48038.actual selector witness)
    (leftRaw := SemanticResult50235.rawTerms)
    (rightRaw := SemanticResult48038.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 47416693230599820876439355444)
    (rightMaximum := 4742405496644812892115304448) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50236) (rightBinding := 50237)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28324⟩) (rightExpression := ⟨28540⟩)
    (transferEvent := 50238) (summaryTransferEvent := 50239)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50235.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult48038.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50240

namespace SemanticResult50245
def owner : Owner := ⟨.program ⟨214⟩, ⟨28758⟩⟩
def rawTerms : List Term := Proof.Events196.exact50245RawTerms
def summary : Bound := (.finite 56901750985984920673341931572)
def resultEvent : Nat := 50245
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50245.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50240.owner)
    (rightOwner := SemanticResult47826.owner)
    (leftResult := 50240) (rightResult := 47826)
    (leftActual := SemanticResult50240.actual selector witness)
    (rightActual := SemanticResult47826.actual selector witness)
    (leftRaw := SemanticResult50240.rawTerms)
    (rightRaw := SemanticResult47826.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52159098727244633768554659892)
    (rightMaximum := 4742652258740286904787271680) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50241) (rightBinding := 50242)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28541⟩) (rightExpression := ⟨28757⟩)
    (transferEvent := 50243) (summaryTransferEvent := 50244)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50240.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult47826.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50245

namespace SemanticResult50250
def owner : Owner := ⟨.program ⟨214⟩, ⟨28975⟩⟩
def rawTerms : List Term := Proof.Events196.exact50250RawTerms
def summary : Bound := (.finite 61644567752788856919910514740)
def resultEvent : Nat := 50250
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50250.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50245.owner)
    (rightOwner := SemanticResult47614.owner)
    (leftResult := 50245) (rightResult := 47614)
    (leftActual := SemanticResult50245.actual selector witness)
    (rightActual := SemanticResult47614.actual selector witness)
    (leftRaw := SemanticResult50245.rawTerms)
    (rightRaw := SemanticResult47614.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 56901750985984920673341931572)
    (rightMaximum := 4742816766803936246568583168) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50246) (rightBinding := 50247)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28758⟩) (rightExpression := ⟨28974⟩)
    (transferEvent := 50248) (summaryTransferEvent := 50249)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50245.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult47614.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50250

namespace SemanticResult50255
def owner : Owner := ⟨.program ⟨214⟩, ⟨29192⟩⟩
def rawTerms : List Term := Proof.Events196.exact50255RawTerms
def summary : Bound := (.finite 66387466773624617837369753652)
def resultEvent : Nat := 50255
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50255.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50250.owner)
    (rightOwner := SemanticResult47402.owner)
    (leftResult := 50250) (rightResult := 47402)
    (leftActual := SemanticResult50250.actual selector witness)
    (rightActual := SemanticResult47402.actual selector witness)
    (leftRaw := SemanticResult50250.rawTerms)
    (rightRaw := SemanticResult47402.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 61644567752788856919910514740)
    (rightMaximum := 4742899020835760917459238912) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50251) (rightBinding := 50252)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28975⟩) (rightExpression := ⟨29191⟩)
    (transferEvent := 50253) (summaryTransferEvent := 50254)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50250.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult47402.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50255

namespace SemanticResult50260
def owner : Owner := ⟨.program ⟨214⟩, ⟨29409⟩⟩
def rawTerms : List Term := Proof.Events196.exact50260RawTerms
def summary : Bound := (.finite 71130530302524028096610304052)
def resultEvent : Nat := 50260
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50260.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50255.owner)
    (rightOwner := SemanticResult47190.owner)
    (leftResult := 50255) (rightResult := 47190)
    (leftActual := SemanticResult50255.actual selector witness)
    (rightActual := SemanticResult47190.actual selector witness)
    (leftRaw := SemanticResult50255.rawTerms)
    (rightRaw := SemanticResult47190.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 66387466773624617837369753652)
    (rightMaximum := 4743063528899410259240550400) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50256) (rightBinding := 50257)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29192⟩) (rightExpression := ⟨29408⟩)
    (transferEvent := 50258) (summaryTransferEvent := 50259)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50255.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult47190.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50260

namespace SemanticResult50265
def owner : Owner := ⟨.program ⟨214⟩, ⟨29626⟩⟩
def rawTerms : List Term := Proof.Events196.exact50265RawTerms
def summary : Bound := (.finite 75873840593518912368522821684)
def resultEvent : Nat := 50265
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50265.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50260.owner)
    (rightOwner := SemanticResult46978.owner)
    (leftResult := 50260) (rightResult := 46978)
    (leftActual := SemanticResult50260.actual selector witness)
    (rightActual := SemanticResult46978.actual selector witness)
    (leftRaw := SemanticResult50260.rawTerms)
    (rightRaw := SemanticResult46978.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 71130530302524028096610304052)
    (rightMaximum := 4743310290994884271912517632) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50261) (rightBinding := 50262)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29409⟩) (rightExpression := ⟨29625⟩)
    (transferEvent := 50263) (summaryTransferEvent := 50264)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50260.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult46978.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50265

namespace SemanticResult50270
def owner : Owner := ⟨.program ⟨214⟩, ⟨29843⟩⟩
def rawTerms : List Term := Proof.Events196.exact50270RawTerms
def summary : Bound := (.finite 80617397646609270653107306548)
def resultEvent : Nat := 50270
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50270.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50265.owner)
    (rightOwner := SemanticResult46766.owner)
    (leftResult := 50265) (rightResult := 46766)
    (leftActual := SemanticResult50265.actual selector witness)
    (rightActual := SemanticResult46766.actual selector witness)
    (leftRaw := SemanticResult50265.rawTerms)
    (rightRaw := SemanticResult46766.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 75873840593518912368522821684)
    (rightMaximum := 4743557053090358284584484864) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50266) (rightBinding := 50267)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29626⟩) (rightExpression := ⟨29842⟩)
    (transferEvent := 50268) (summaryTransferEvent := 50269)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50265.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult46766.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50270

namespace SemanticResult50275
def owner : Owner := ⟨.program ⟨214⟩, ⟨30159⟩⟩
def rawTerms : List Term := Proof.Events196.exact50275RawTerms
def summary : Bound := (.finite 85361036953731453608582447156)
def resultEvent : Nat := 50275
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50275.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50270.owner)
    (rightOwner := SemanticResult46554.owner)
    (leftResult := 50270) (rightResult := 46554)
    (leftActual := SemanticResult50270.actual selector witness)
    (rightActual := SemanticResult46554.actual selector witness)
    (leftRaw := SemanticResult50270.rawTerms)
    (rightRaw := SemanticResult46554.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 80617397646609270653107306548)
    (rightMaximum := 4743639307122182955475140608) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50271) (rightBinding := 50272)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29843⟩) (rightExpression := ⟨30158⟩)
    (transferEvent := 50273) (summaryTransferEvent := 50274)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50270.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult46554.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50275

namespace SemanticResult50280
def owner : Owner := ⟨.program ⟨214⟩, ⟨30170⟩⟩
def rawTerms : List Term := Proof.Events196.exact50280RawTerms
def summary : Bound := (.finite 313276456757822654825721789388161076)
def resultEvent : Nat := 50280
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50280.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50275.owner)
    (rightOwner := SemanticResult46342.owner)
    (leftResult := 50275) (rightResult := 46342)
    (leftActual := SemanticResult50275.actual selector witness)
    (rightActual := SemanticResult46342.actual selector witness)
    (leftRaw := SemanticResult50275.rawTerms)
    (rightRaw := SemanticResult46342.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 85361036953731453608582447156)
    (rightMaximum := 313276371396785701094268180805713920) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50276) (rightBinding := 50277)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨30159⟩) (rightExpression := ⟨30168⟩)
    (transferEvent := 50278) (summaryTransferEvent := 50279)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50275.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult46342.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50280

namespace SemanticResult50282
def owner : Owner := ⟨.program ⟨214⟩, ⟨70⟩⟩
def rawTerms : List Term := Proof.Events196.exact50282RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50282
def producerEvent : Nat := 50281
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50282.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .finite 26, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult50282

namespace SemanticResult50287
def owner : Owner := ⟨.program ⟨214⟩, ⟨7091⟩⟩
def rawTerms : List Term := Proof.Events196.exact50287RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50287
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50287.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50286.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge50286.frameStart)
    (transferEvent := 50285) (owner := owner)
    (leftResult := 27) (rightResult := 6044)
    (working := LeftOperatorMerge50286.working)
    (reconstruction := LeftOperatorMerge50286.reconstruction)
    (leftReference := .predecessor 0 50283 .coefficient) (rightReference := .predecessor 1 50284 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6044.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge50286.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult50287

namespace SemanticResult50291
def owner : Owner := ⟨.program ⟨214⟩, ⟨7723⟩⟩
def rawTerms : List Term := Proof.Events196.exact50291RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50291
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50291.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 50288) (rightBinding := 50289)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7091⟩) (rightExpression := ⟨6569⟩)
    (transferEvent := 50290)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50287.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50291

namespace SemanticResult50297
def owner : Owner := ⟨.program ⟨214⟩, ⟨7724⟩⟩
def rawTerms : List Term := Proof.Events196.exact50297RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 50297
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50297.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 50294) (survivorTransfer := 50295)
    (survivorEvent := 50296) (resultEvent := resultEvent)
    (rightCoefficientProducer := 50281)
    (owner := owner) (leftOwner := SemanticResult50291.owner)
    (rightOwner := SemanticResult50282.owner)
    (leftResult := 50291) (rightResult := 50282)
    (leftBinding := 50292) (rightBinding := 50293)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7723⟩) (rightExpression := ⟨70⟩)
    (leftActual := SemanticResult50291.actual selector witness)
    (rightActual := SemanticResult50282.actual selector witness)
    (leftRaw := SemanticResult50291.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨70⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftAuthority50281.actual selector witness)
    (survivorMagnitude := LeftBound50295.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50291.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50282.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50281.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50281.derived selector witness)
  · exact LeftBound50295.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult50297

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
