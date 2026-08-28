import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard764
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard050
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard161
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard748
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard750
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard751
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard752
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard754
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard755
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard757
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard758
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard759
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard761
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard762
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard763

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult107173
def owner : Owner := ⟨.program ⟨214⟩, ⟨7097⟩⟩
def rawTerms : List Term := Proof.Events418.exact107173RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 107173
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107173.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge107172.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge107172.frameStart)
    (transferEvent := 107171) (owner := owner)
    (leftResult := 27) (rightResult := 5873)
    (working := LeftOperatorMerge107172.working)
    (reconstruction := LeftOperatorMerge107172.reconstruction)
    (leftReference := .predecessor 0 107169 .coefficient) (rightReference := .predecessor 1 107170 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5873.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge107172.operationAgreement
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
end SemanticResult107173

namespace SemanticResult107177
def owner : Owner := ⟨.program ⟨214⟩, ⟨7731⟩⟩
def rawTerms : List Term := Proof.Events418.exact107177RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 107177
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107177.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 107174) (rightBinding := 107175)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7097⟩) (rightExpression := ⟨6623⟩)
    (transferEvent := 107176)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult107173.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult107168.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult107177

namespace SemanticResult107183
def owner : Owner := ⟨.program ⟨214⟩, ⟨7732⟩⟩
def rawTerms : List Term := Proof.Events418.exact107183RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 107183
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107183.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 107180) (survivorTransfer := 107181)
    (survivorEvent := 107182) (resultEvent := resultEvent)
    (rightCoefficientProducer := 20907)
    (owner := owner) (leftOwner := SemanticResult107177.owner)
    (rightOwner := SemanticResult20908.owner)
    (leftResult := 107177) (rightResult := 20908)
    (leftBinding := 107178) (rightBinding := 107179)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7731⟩) (rightExpression := ⟨74⟩)
    (leftActual := SemanticResult107177.actual selector witness)
    (rightActual := SemanticResult20908.actual selector witness)
    (leftRaw := SemanticResult107177.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨74⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound20907.actual selector witness)
    (survivorMagnitude := LeftBound107181.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult107177.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20908.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)
  · exact LeftBound107181.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult107183

namespace SemanticResult107190
def owner : Owner := ⟨.program ⟨214⟩, ⟨7805⟩⟩
def rawTerms : List Term := Proof.Events418.exact107190RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 107190
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107190.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge107187.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult107183.owner)
    (rightOwner := SemanticResult107183.owner)
    (leftResult := 107183) (rightResult := 107183)
    (leftActual := SemanticResult107183.actual selector witness)
    (rightActual := SemanticResult107183.actual selector witness)
    (leftRaw := SemanticResult107183.rawTerms)
    (rightRaw := SemanticResult107183.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 107184) (rightBinding := 107185)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7732⟩) (rightExpression := ⟨7732⟩)
    (coefficientTransfer := 107186) (summaryTransfer := 107189)
    (base := LeftOperatorMerge107187.base)
    (reconstruction := LeftOperatorMerge107187.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult107183.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult107183.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge107187.operationAgreement
  · rfl
  · decide
end SemanticResult107190

namespace SemanticResult107195
def owner : Owner := ⟨.program ⟨214⟩, ⟨26324⟩⟩
def rawTerms : List Term := Proof.Events418.exact107195RawTerms
def summary : Bound := (.finite 4741253940199267499646124084)
def resultEvent : Nat := 107195
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107195.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult107190.owner)
    (rightOwner := SemanticResult107163.owner)
    (leftResult := 107190) (rightResult := 107163)
    (leftActual := SemanticResult107190.actual selector witness)
    (rightActual := SemanticResult107163.actual selector witness)
    (leftRaw := SemanticResult107190.rawTerms)
    (rightRaw := SemanticResult107163.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 4741253940199267499646124032) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 107191) (rightBinding := 107192)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7805⟩) (rightExpression := ⟨26323⟩)
    (transferEvent := 107193) (summaryTransferEvent := 107194)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult107190.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult107163.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult107195

namespace SemanticResult107200
def owner : Owner := ⟨.program ⟨214⟩, ⟨26527⟩⟩
def rawTerms : List Term := Proof.Events418.exact107200RawTerms
def summary : Bound := (.finite 9482549007414447334737575988)
def resultEvent : Nat := 107200
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107200.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult107195.owner)
    (rightOwner := SemanticResult106975.owner)
    (leftResult := 107195) (rightResult := 106975)
    (leftActual := SemanticResult107195.actual selector witness)
    (rightActual := SemanticResult106975.actual selector witness)
    (leftRaw := SemanticResult107195.rawTerms)
    (rightRaw := SemanticResult106975.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4741253940199267499646124084)
    (rightMaximum := 4741295067215179835091451904) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 107196) (rightBinding := 107197)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26324⟩) (rightExpression := ⟨26526⟩)
    (transferEvent := 107198) (summaryTransferEvent := 107199)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult107195.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult106975.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult107200

namespace SemanticResult107205
def owner : Owner := ⟨.program ⟨214⟩, ⟨26744⟩⟩
def rawTerms : List Term := Proof.Events418.exact107205RawTerms
def summary : Bound := (.finite 14223885201645539505274355764)
def resultEvent : Nat := 107205
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107205.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult107200.owner)
    (rightOwner := SemanticResult106787.owner)
    (leftResult := 107200) (rightResult := 106787)
    (leftActual := SemanticResult107200.actual selector witness)
    (rightActual := SemanticResult106787.actual selector witness)
    (leftRaw := SemanticResult107200.rawTerms)
    (rightRaw := SemanticResult106787.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 9482549007414447334737575988)
    (rightMaximum := 4741336194231092170536779776) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 107201) (rightBinding := 107202)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26527⟩) (rightExpression := ⟨26743⟩)
    (transferEvent := 107203) (summaryTransferEvent := 107204)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult107200.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult106787.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult107205

namespace SemanticResult107210
def owner : Owner := ⟨.program ⟨214⟩, ⟨26961⟩⟩
def rawTerms : List Term := Proof.Events418.exact107210RawTerms
def summary : Bound := (.finite 18965303649908456346701791284)
def resultEvent : Nat := 107210
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107210.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult107205.owner)
    (rightOwner := SemanticResult106599.owner)
    (leftResult := 107205) (rightResult := 106599)
    (leftActual := SemanticResult107205.actual selector witness)
    (rightActual := SemanticResult106599.actual selector witness)
    (leftRaw := SemanticResult107205.rawTerms)
    (rightRaw := SemanticResult106599.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 14223885201645539505274355764)
    (rightMaximum := 4741418448262916841427435520) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 107206) (rightBinding := 107207)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26744⟩) (rightExpression := ⟨26960⟩)
    (transferEvent := 107208) (summaryTransferEvent := 107209)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult107205.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult106599.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult107210

namespace SemanticResult107215
def owner : Owner := ⟨.program ⟨214⟩, ⟨27178⟩⟩
def rawTerms : List Term := Proof.Events418.exact107215RawTerms
def summary : Bound := (.finite 23706886606235022529910538292)
def resultEvent : Nat := 107215
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107215.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult107210.owner)
    (rightOwner := SemanticResult106411.owner)
    (leftResult := 107210) (rightResult := 106411)
    (leftActual := SemanticResult107210.actual selector witness)
    (rightActual := SemanticResult106411.actual selector witness)
    (leftRaw := SemanticResult107210.rawTerms)
    (rightRaw := SemanticResult106411.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 18965303649908456346701791284)
    (rightMaximum := 4741582956326566183208747008) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 107211) (rightBinding := 107212)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26961⟩) (rightExpression := ⟨27177⟩)
    (transferEvent := 107213) (summaryTransferEvent := 107214)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult107210.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult106411.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult107215

namespace SemanticResult107220
def owner : Owner := ⟨.program ⟨214⟩, ⟨27395⟩⟩
def rawTerms : List Term := Proof.Events418.exact107220RawTerms
def summary : Bound := (.finite 28448551816593413384009941044)
def resultEvent : Nat := 107220
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107220.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult107215.owner)
    (rightOwner := SemanticResult106223.owner)
    (leftResult := 107215) (rightResult := 106223)
    (leftActual := SemanticResult107215.actual selector witness)
    (rightActual := SemanticResult106223.actual selector witness)
    (leftRaw := SemanticResult107215.rawTerms)
    (rightRaw := SemanticResult106223.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 23706886606235022529910538292)
    (rightMaximum := 4741665210358390854099402752) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 107216) (rightBinding := 107217)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27178⟩) (rightExpression := ⟨27394⟩)
    (transferEvent := 107218) (summaryTransferEvent := 107219)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult107215.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult106223.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult107220

namespace SemanticResult107225
def owner : Owner := ⟨.program ⟨214⟩, ⟨27612⟩⟩
def rawTerms : List Term := Proof.Events418.exact107225RawTerms
def summary : Bound := (.finite 33190381535015453579890655284)
def resultEvent : Nat := 107225
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107225.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult107220.owner)
    (rightOwner := SemanticResult106035.owner)
    (leftResult := 107220) (rightResult := 106035)
    (leftActual := SemanticResult107220.actual selector witness)
    (rightActual := SemanticResult106035.actual selector witness)
    (leftRaw := SemanticResult107220.rawTerms)
    (rightRaw := SemanticResult106035.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 28448551816593413384009941044)
    (rightMaximum := 4741829718422040195880714240) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 107221) (rightBinding := 107222)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27395⟩) (rightExpression := ⟨27611⟩)
    (transferEvent := 107223) (summaryTransferEvent := 107224)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult107220.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult106035.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult107225

namespace SemanticResult107230
def owner : Owner := ⟨.program ⟨214⟩, ⟨27829⟩⟩
def rawTerms : List Term := Proof.Events418.exact107230RawTerms
def summary : Bound := (.finite 37932293507469318446662025268)
def resultEvent : Nat := 107230
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107230.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult107225.owner)
    (rightOwner := SemanticResult105847.owner)
    (leftResult := 107225) (rightResult := 105847)
    (leftActual := SemanticResult107225.actual selector witness)
    (rightActual := SemanticResult105847.actual selector witness)
    (leftRaw := SemanticResult107225.rawTerms)
    (rightRaw := SemanticResult105847.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 33190381535015453579890655284)
    (rightMaximum := 4741911972453864866771369984) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 107226) (rightBinding := 107227)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27612⟩) (rightExpression := ⟨27828⟩)
    (transferEvent := 107228) (summaryTransferEvent := 107229)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult107225.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult105847.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult107230

namespace SemanticResult107235
def owner : Owner := ⟨.program ⟨214⟩, ⟨28046⟩⟩
def rawTerms : List Term := Proof.Events418.exact107235RawTerms
def summary : Bound := (.finite 42674369987986832655214706740)
def resultEvent : Nat := 107235
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107235.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult107230.owner)
    (rightOwner := SemanticResult105659.owner)
    (leftResult := 107230) (rightResult := 105659)
    (leftActual := SemanticResult107230.actual selector witness)
    (rightActual := SemanticResult105659.actual selector witness)
    (leftRaw := SemanticResult107230.rawTerms)
    (rightRaw := SemanticResult105659.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 37932293507469318446662025268)
    (rightMaximum := 4742076480517514208552681472) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 107231) (rightBinding := 107232)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27829⟩) (rightExpression := ⟨28045⟩)
    (transferEvent := 107233) (summaryTransferEvent := 107234)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult107230.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult105659.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult107235

namespace SemanticResult107240
def owner : Owner := ⟨.program ⟨214⟩, ⟨28263⟩⟩
def rawTerms : List Term := Proof.Events418.exact107240RawTerms
def summary : Bound := (.finite 47416693230599820876439355444)
def resultEvent : Nat := 107240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107240.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult107235.owner)
    (rightOwner := SemanticResult105471.owner)
    (leftResult := 107235) (rightResult := 105471)
    (leftActual := SemanticResult107235.actual selector witness)
    (rightActual := SemanticResult105471.actual selector witness)
    (leftRaw := SemanticResult107235.rawTerms)
    (rightRaw := SemanticResult105471.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 42674369987986832655214706740)
    (rightMaximum := 4742323242612988221224648704) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 107236) (rightBinding := 107237)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28046⟩) (rightExpression := ⟨28262⟩)
    (transferEvent := 107238) (summaryTransferEvent := 107239)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult107235.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult105471.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult107240

namespace SemanticResult107245
def owner : Owner := ⟨.program ⟨214⟩, ⟨28480⟩⟩
def rawTerms : List Term := Proof.Events418.exact107245RawTerms
def summary : Bound := (.finite 52159098727244633768554659892)
def resultEvent : Nat := 107245
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107245.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult107240.owner)
    (rightOwner := SemanticResult105283.owner)
    (leftResult := 107240) (rightResult := 105283)
    (leftActual := SemanticResult107240.actual selector witness)
    (rightActual := SemanticResult105283.actual selector witness)
    (leftRaw := SemanticResult107240.rawTerms)
    (rightRaw := SemanticResult105283.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 47416693230599820876439355444)
    (rightMaximum := 4742405496644812892115304448) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 107241) (rightBinding := 107242)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28263⟩) (rightExpression := ⟨28479⟩)
    (transferEvent := 107243) (summaryTransferEvent := 107244)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult107240.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult105283.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult107245

namespace SemanticResult107250
def owner : Owner := ⟨.program ⟨214⟩, ⟨28697⟩⟩
def rawTerms : List Term := Proof.Events418.exact107250RawTerms
def summary : Bound := (.finite 56901750985984920673341931572)
def resultEvent : Nat := 107250
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult107250.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult107245.owner)
    (rightOwner := SemanticResult105095.owner)
    (leftResult := 107245) (rightResult := 105095)
    (leftActual := SemanticResult107245.actual selector witness)
    (rightActual := SemanticResult105095.actual selector witness)
    (leftRaw := SemanticResult107245.rawTerms)
    (rightRaw := SemanticResult105095.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52159098727244633768554659892)
    (rightMaximum := 4742652258740286904787271680) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 107246) (rightBinding := 107247)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28480⟩) (rightExpression := ⟨28696⟩)
    (transferEvent := 107248) (summaryTransferEvent := 107249)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult107245.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult105095.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult107250

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
