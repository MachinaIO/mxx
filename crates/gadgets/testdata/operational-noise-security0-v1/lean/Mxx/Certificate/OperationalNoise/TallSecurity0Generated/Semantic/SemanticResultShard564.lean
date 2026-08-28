import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard564
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard053
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard539
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard544
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard545
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard546
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard548
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard549
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard550
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard552
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard553
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard555
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard556
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard563

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult79465
def owner : Owner := ⟨.program ⟨214⟩, ⟨27417⟩⟩
def rawTerms : List Term := Proof.Events310.exact79465RawTerms
def summary : Bound := (.finite 28448551816593413384009941044)
def resultEvent : Nat := 79465
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79465.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79460.owner)
    (rightOwner := SemanticResult78348.owner)
    (leftResult := 79460) (rightResult := 78348)
    (leftActual := SemanticResult79460.actual selector witness)
    (rightActual := SemanticResult78348.actual selector witness)
    (leftRaw := SemanticResult79460.rawTerms)
    (rightRaw := SemanticResult78348.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 23706886606235022529910538292)
    (rightMaximum := 4741665210358390854099402752) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79461) (rightBinding := 79462)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27200⟩) (rightExpression := ⟨27416⟩)
    (transferEvent := 79463) (summaryTransferEvent := 79464)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79460.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult78348.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79465

namespace SemanticResult79470
def owner : Owner := ⟨.program ⟨214⟩, ⟨27634⟩⟩
def rawTerms : List Term := Proof.Events310.exact79470RawTerms
def summary : Bound := (.finite 33190381535015453579890655284)
def resultEvent : Nat := 79470
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79470.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79465.owner)
    (rightOwner := SemanticResult78136.owner)
    (leftResult := 79465) (rightResult := 78136)
    (leftActual := SemanticResult79465.actual selector witness)
    (rightActual := SemanticResult78136.actual selector witness)
    (leftRaw := SemanticResult79465.rawTerms)
    (rightRaw := SemanticResult78136.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 28448551816593413384009941044)
    (rightMaximum := 4741829718422040195880714240) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79466) (rightBinding := 79467)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27417⟩) (rightExpression := ⟨27633⟩)
    (transferEvent := 79468) (summaryTransferEvent := 79469)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79465.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult78136.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79470

namespace SemanticResult79475
def owner : Owner := ⟨.program ⟨214⟩, ⟨27851⟩⟩
def rawTerms : List Term := Proof.Events310.exact79475RawTerms
def summary : Bound := (.finite 37932293507469318446662025268)
def resultEvent : Nat := 79475
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79475.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79470.owner)
    (rightOwner := SemanticResult77924.owner)
    (leftResult := 79470) (rightResult := 77924)
    (leftActual := SemanticResult79470.actual selector witness)
    (rightActual := SemanticResult77924.actual selector witness)
    (leftRaw := SemanticResult79470.rawTerms)
    (rightRaw := SemanticResult77924.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 33190381535015453579890655284)
    (rightMaximum := 4741911972453864866771369984) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79471) (rightBinding := 79472)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27634⟩) (rightExpression := ⟨27850⟩)
    (transferEvent := 79473) (summaryTransferEvent := 79474)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79470.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult77924.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79475

namespace SemanticResult79480
def owner : Owner := ⟨.program ⟨214⟩, ⟨28068⟩⟩
def rawTerms : List Term := Proof.Events310.exact79480RawTerms
def summary : Bound := (.finite 42674369987986832655214706740)
def resultEvent : Nat := 79480
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79480.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79475.owner)
    (rightOwner := SemanticResult77712.owner)
    (leftResult := 79475) (rightResult := 77712)
    (leftActual := SemanticResult79475.actual selector witness)
    (rightActual := SemanticResult77712.actual selector witness)
    (leftRaw := SemanticResult79475.rawTerms)
    (rightRaw := SemanticResult77712.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 37932293507469318446662025268)
    (rightMaximum := 4742076480517514208552681472) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79476) (rightBinding := 79477)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27851⟩) (rightExpression := ⟨28067⟩)
    (transferEvent := 79478) (summaryTransferEvent := 79479)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79475.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult77712.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79480

namespace SemanticResult79485
def owner : Owner := ⟨.program ⟨214⟩, ⟨28285⟩⟩
def rawTerms : List Term := Proof.Events310.exact79485RawTerms
def summary : Bound := (.finite 47416693230599820876439355444)
def resultEvent : Nat := 79485
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79485.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79480.owner)
    (rightOwner := SemanticResult77500.owner)
    (leftResult := 79480) (rightResult := 77500)
    (leftActual := SemanticResult79480.actual selector witness)
    (rightActual := SemanticResult77500.actual selector witness)
    (leftRaw := SemanticResult79480.rawTerms)
    (rightRaw := SemanticResult77500.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 42674369987986832655214706740)
    (rightMaximum := 4742323242612988221224648704) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79481) (rightBinding := 79482)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28068⟩) (rightExpression := ⟨28284⟩)
    (transferEvent := 79483) (summaryTransferEvent := 79484)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79480.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult77500.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79485

namespace SemanticResult79490
def owner : Owner := ⟨.program ⟨214⟩, ⟨28502⟩⟩
def rawTerms : List Term := Proof.Events310.exact79490RawTerms
def summary : Bound := (.finite 52159098727244633768554659892)
def resultEvent : Nat := 79490
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79490.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79485.owner)
    (rightOwner := SemanticResult77288.owner)
    (leftResult := 79485) (rightResult := 77288)
    (leftActual := SemanticResult79485.actual selector witness)
    (rightActual := SemanticResult77288.actual selector witness)
    (leftRaw := SemanticResult79485.rawTerms)
    (rightRaw := SemanticResult77288.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 47416693230599820876439355444)
    (rightMaximum := 4742405496644812892115304448) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79486) (rightBinding := 79487)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28285⟩) (rightExpression := ⟨28501⟩)
    (transferEvent := 79488) (summaryTransferEvent := 79489)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79485.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult77288.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79490

namespace SemanticResult79495
def owner : Owner := ⟨.program ⟨214⟩, ⟨28719⟩⟩
def rawTerms : List Term := Proof.Events310.exact79495RawTerms
def summary : Bound := (.finite 56901750985984920673341931572)
def resultEvent : Nat := 79495
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79495.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79490.owner)
    (rightOwner := SemanticResult77076.owner)
    (leftResult := 79490) (rightResult := 77076)
    (leftActual := SemanticResult79490.actual selector witness)
    (rightActual := SemanticResult77076.actual selector witness)
    (leftRaw := SemanticResult79490.rawTerms)
    (rightRaw := SemanticResult77076.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52159098727244633768554659892)
    (rightMaximum := 4742652258740286904787271680) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79491) (rightBinding := 79492)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28502⟩) (rightExpression := ⟨28718⟩)
    (transferEvent := 79493) (summaryTransferEvent := 79494)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79490.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult77076.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79495

namespace SemanticResult79500
def owner : Owner := ⟨.program ⟨214⟩, ⟨28936⟩⟩
def rawTerms : List Term := Proof.Events310.exact79500RawTerms
def summary : Bound := (.finite 61644567752788856919910514740)
def resultEvent : Nat := 79500
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79500.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79495.owner)
    (rightOwner := SemanticResult76864.owner)
    (leftResult := 79495) (rightResult := 76864)
    (leftActual := SemanticResult79495.actual selector witness)
    (rightActual := SemanticResult76864.actual selector witness)
    (leftRaw := SemanticResult79495.rawTerms)
    (rightRaw := SemanticResult76864.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 56901750985984920673341931572)
    (rightMaximum := 4742816766803936246568583168) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79496) (rightBinding := 79497)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28719⟩) (rightExpression := ⟨28935⟩)
    (transferEvent := 79498) (summaryTransferEvent := 79499)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79495.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult76864.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79500

namespace SemanticResult79505
def owner : Owner := ⟨.program ⟨214⟩, ⟨29153⟩⟩
def rawTerms : List Term := Proof.Events310.exact79505RawTerms
def summary : Bound := (.finite 66387466773624617837369753652)
def resultEvent : Nat := 79505
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79505.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79500.owner)
    (rightOwner := SemanticResult76652.owner)
    (leftResult := 79500) (rightResult := 76652)
    (leftActual := SemanticResult79500.actual selector witness)
    (rightActual := SemanticResult76652.actual selector witness)
    (leftRaw := SemanticResult79500.rawTerms)
    (rightRaw := SemanticResult76652.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 61644567752788856919910514740)
    (rightMaximum := 4742899020835760917459238912) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79501) (rightBinding := 79502)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28936⟩) (rightExpression := ⟨29152⟩)
    (transferEvent := 79503) (summaryTransferEvent := 79504)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79500.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult76652.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79505

namespace SemanticResult79510
def owner : Owner := ⟨.program ⟨214⟩, ⟨29370⟩⟩
def rawTerms : List Term := Proof.Events310.exact79510RawTerms
def summary : Bound := (.finite 71130530302524028096610304052)
def resultEvent : Nat := 79510
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79510.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79505.owner)
    (rightOwner := SemanticResult76440.owner)
    (leftResult := 79505) (rightResult := 76440)
    (leftActual := SemanticResult79505.actual selector witness)
    (rightActual := SemanticResult76440.actual selector witness)
    (leftRaw := SemanticResult79505.rawTerms)
    (rightRaw := SemanticResult76440.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 66387466773624617837369753652)
    (rightMaximum := 4743063528899410259240550400) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79506) (rightBinding := 79507)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29153⟩) (rightExpression := ⟨29369⟩)
    (transferEvent := 79508) (summaryTransferEvent := 79509)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79505.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult76440.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79510

namespace SemanticResult79515
def owner : Owner := ⟨.program ⟨214⟩, ⟨29587⟩⟩
def rawTerms : List Term := Proof.Events310.exact79515RawTerms
def summary : Bound := (.finite 75873840593518912368522821684)
def resultEvent : Nat := 79515
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79515.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79510.owner)
    (rightOwner := SemanticResult76228.owner)
    (leftResult := 79510) (rightResult := 76228)
    (leftActual := SemanticResult79510.actual selector witness)
    (rightActual := SemanticResult76228.actual selector witness)
    (leftRaw := SemanticResult79510.rawTerms)
    (rightRaw := SemanticResult76228.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 71130530302524028096610304052)
    (rightMaximum := 4743310290994884271912517632) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79511) (rightBinding := 79512)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29370⟩) (rightExpression := ⟨29586⟩)
    (transferEvent := 79513) (summaryTransferEvent := 79514)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79510.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult76228.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79515

namespace SemanticResult79520
def owner : Owner := ⟨.program ⟨214⟩, ⟨29804⟩⟩
def rawTerms : List Term := Proof.Events310.exact79520RawTerms
def summary : Bound := (.finite 80617397646609270653107306548)
def resultEvent : Nat := 79520
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79520.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79515.owner)
    (rightOwner := SemanticResult76016.owner)
    (leftResult := 79515) (rightResult := 76016)
    (leftActual := SemanticResult79515.actual selector witness)
    (rightActual := SemanticResult76016.actual selector witness)
    (leftRaw := SemanticResult79515.rawTerms)
    (rightRaw := SemanticResult76016.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 75873840593518912368522821684)
    (rightMaximum := 4743557053090358284584484864) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79516) (rightBinding := 79517)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29587⟩) (rightExpression := ⟨29803⟩)
    (transferEvent := 79518) (summaryTransferEvent := 79519)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79515.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult76016.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79520

namespace SemanticResult79525
def owner : Owner := ⟨.program ⟨214⟩, ⟨30093⟩⟩
def rawTerms : List Term := Proof.Events310.exact79525RawTerms
def summary : Bound := (.finite 85361036953731453608582447156)
def resultEvent : Nat := 79525
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79525.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79520.owner)
    (rightOwner := SemanticResult75804.owner)
    (leftResult := 79520) (rightResult := 75804)
    (leftActual := SemanticResult79520.actual selector witness)
    (rightActual := SemanticResult75804.actual selector witness)
    (leftRaw := SemanticResult79520.rawTerms)
    (rightRaw := SemanticResult75804.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 80617397646609270653107306548)
    (rightMaximum := 4743639307122182955475140608) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79521) (rightBinding := 79522)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29804⟩) (rightExpression := ⟨30092⟩)
    (transferEvent := 79523) (summaryTransferEvent := 79524)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79520.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75804.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79525

namespace SemanticResult79530
def owner : Owner := ⟨.program ⟨214⟩, ⟨30104⟩⟩
def rawTerms : List Term := Proof.Events310.exact79530RawTerms
def summary : Bound := (.finite 313276456757822654825721789388161076)
def resultEvent : Nat := 79530
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79530.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79525.owner)
    (rightOwner := SemanticResult75592.owner)
    (leftResult := 79525) (rightResult := 75592)
    (leftActual := SemanticResult79525.actual selector witness)
    (rightActual := SemanticResult75592.actual selector witness)
    (leftRaw := SemanticResult79525.rawTerms)
    (rightRaw := SemanticResult75592.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 85361036953731453608582447156)
    (rightMaximum := 313276371396785701094268180805713920) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79526) (rightBinding := 79527)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨30093⟩) (rightExpression := ⟨30102⟩)
    (transferEvent := 79528) (summaryTransferEvent := 79529)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79525.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75592.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79530

namespace SemanticResult79532
def owner : Owner := ⟨.program ⟨214⟩, ⟨55⟩⟩
def rawTerms : List Term := Proof.Events310.exact79532RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 79532
def producerEvent : Nat := 79531
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79532.actual selector witness
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
end SemanticResult79532

namespace SemanticResult79537
def owner : Owner := ⟨.program ⟨214⟩, ⟨7093⟩⟩
def rawTerms : List Term := Proof.Events310.exact79537RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 79537
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79537.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge79536.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge79536.frameStart)
    (transferEvent := 79535) (owner := owner)
    (leftResult := 27) (rightResult := 6124)
    (working := LeftOperatorMerge79536.working)
    (reconstruction := LeftOperatorMerge79536.reconstruction)
    (leftReference := .predecessor 0 79533 .coefficient) (rightReference := .predecessor 1 79534 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6124.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge79536.operationAgreement
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
end SemanticResult79537

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
