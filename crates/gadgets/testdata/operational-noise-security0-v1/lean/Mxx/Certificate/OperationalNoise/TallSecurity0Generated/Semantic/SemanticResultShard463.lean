import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard463
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard161
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard444
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard446
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard447
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard449
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard450
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard451
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard453
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard454
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard455
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard457
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard458
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard460
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard461
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard462

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult64803
def owner : Owner := ⟨.program ⟨214⟩, ⟨7758⟩⟩
def rawTerms : List Term := Proof.Events253.exact64803RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 64803
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64803.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 64800) (survivorTransfer := 64801)
    (survivorEvent := 64802) (resultEvent := resultEvent)
    (rightCoefficientProducer := 20907)
    (owner := owner) (leftOwner := SemanticResult64797.owner)
    (rightOwner := SemanticResult20908.owner)
    (leftResult := 64797) (rightResult := 20908)
    (leftBinding := 64798) (rightBinding := 64799)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7757⟩) (rightExpression := ⟨74⟩)
    (leftActual := SemanticResult64797.actual selector witness)
    (rightActual := SemanticResult20908.actual selector witness)
    (leftRaw := SemanticResult64797.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨74⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound20907.actual selector witness)
    (survivorMagnitude := LeftBound64801.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64797.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20908.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)
  · exact LeftBound64801.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult64803

namespace SemanticResult64810
def owner : Owner := ⟨.program ⟨214⟩, ⟨7809⟩⟩
def rawTerms : List Term := Proof.Events253.exact64810RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 64810
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64810.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge64807.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult64803.owner)
    (rightOwner := SemanticResult64803.owner)
    (leftResult := 64803) (rightResult := 64803)
    (leftActual := SemanticResult64803.actual selector witness)
    (rightActual := SemanticResult64803.actual selector witness)
    (leftRaw := SemanticResult64803.rawTerms)
    (rightRaw := SemanticResult64803.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 64804) (rightBinding := 64805)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7758⟩) (rightExpression := ⟨7758⟩)
    (coefficientTransfer := 64806) (summaryTransfer := 64809)
    (base := LeftOperatorMerge64807.base)
    (reconstruction := LeftOperatorMerge64807.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64803.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult64803.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge64807.operationAgreement
  · rfl
  · decide
end SemanticResult64810

namespace SemanticResult64815
def owner : Owner := ⟨.program ⟨214⟩, ⟨26368⟩⟩
def rawTerms : List Term := Proof.Events253.exact64815RawTerms
def summary : Bound := (.finite 4741253940199267499646124084)
def resultEvent : Nat := 64815
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64815.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult64810.owner)
    (rightOwner := SemanticResult64783.owner)
    (leftResult := 64810) (rightResult := 64783)
    (leftActual := SemanticResult64810.actual selector witness)
    (rightActual := SemanticResult64783.actual selector witness)
    (leftRaw := SemanticResult64810.rawTerms)
    (rightRaw := SemanticResult64783.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 4741253940199267499646124032) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 64811) (rightBinding := 64812)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7809⟩) (rightExpression := ⟨26367⟩)
    (transferEvent := 64813) (summaryTransferEvent := 64814)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64810.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult64783.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult64815

namespace SemanticResult64820
def owner : Owner := ⟨.program ⟨214⟩, ⟨26575⟩⟩
def rawTerms : List Term := Proof.Events253.exact64820RawTerms
def summary : Bound := (.finite 9482549007414447334737575988)
def resultEvent : Nat := 64820
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64820.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult64815.owner)
    (rightOwner := SemanticResult64571.owner)
    (leftResult := 64815) (rightResult := 64571)
    (leftActual := SemanticResult64815.actual selector witness)
    (rightActual := SemanticResult64571.actual selector witness)
    (leftRaw := SemanticResult64815.rawTerms)
    (rightRaw := SemanticResult64571.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4741253940199267499646124084)
    (rightMaximum := 4741295067215179835091451904) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 64816) (rightBinding := 64817)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26368⟩) (rightExpression := ⟨26574⟩)
    (transferEvent := 64818) (summaryTransferEvent := 64819)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64815.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult64571.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult64820

namespace SemanticResult64825
def owner : Owner := ⟨.program ⟨214⟩, ⟨26792⟩⟩
def rawTerms : List Term := Proof.Events253.exact64825RawTerms
def summary : Bound := (.finite 14223885201645539505274355764)
def resultEvent : Nat := 64825
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64825.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult64820.owner)
    (rightOwner := SemanticResult64359.owner)
    (leftResult := 64820) (rightResult := 64359)
    (leftActual := SemanticResult64820.actual selector witness)
    (rightActual := SemanticResult64359.actual selector witness)
    (leftRaw := SemanticResult64820.rawTerms)
    (rightRaw := SemanticResult64359.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 9482549007414447334737575988)
    (rightMaximum := 4741336194231092170536779776) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 64821) (rightBinding := 64822)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26575⟩) (rightExpression := ⟨26791⟩)
    (transferEvent := 64823) (summaryTransferEvent := 64824)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64820.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult64359.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult64825

namespace SemanticResult64830
def owner : Owner := ⟨.program ⟨214⟩, ⟨27009⟩⟩
def rawTerms : List Term := Proof.Events253.exact64830RawTerms
def summary : Bound := (.finite 18965303649908456346701791284)
def resultEvent : Nat := 64830
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64830.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult64825.owner)
    (rightOwner := SemanticResult64147.owner)
    (leftResult := 64825) (rightResult := 64147)
    (leftActual := SemanticResult64825.actual selector witness)
    (rightActual := SemanticResult64147.actual selector witness)
    (leftRaw := SemanticResult64825.rawTerms)
    (rightRaw := SemanticResult64147.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 14223885201645539505274355764)
    (rightMaximum := 4741418448262916841427435520) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 64826) (rightBinding := 64827)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26792⟩) (rightExpression := ⟨27008⟩)
    (transferEvent := 64828) (summaryTransferEvent := 64829)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64825.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult64147.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult64830

namespace SemanticResult64835
def owner : Owner := ⟨.program ⟨214⟩, ⟨27226⟩⟩
def rawTerms : List Term := Proof.Events253.exact64835RawTerms
def summary : Bound := (.finite 23706886606235022529910538292)
def resultEvent : Nat := 64835
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64835.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult64830.owner)
    (rightOwner := SemanticResult63935.owner)
    (leftResult := 64830) (rightResult := 63935)
    (leftActual := SemanticResult64830.actual selector witness)
    (rightActual := SemanticResult63935.actual selector witness)
    (leftRaw := SemanticResult64830.rawTerms)
    (rightRaw := SemanticResult63935.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 18965303649908456346701791284)
    (rightMaximum := 4741582956326566183208747008) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 64831) (rightBinding := 64832)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27009⟩) (rightExpression := ⟨27225⟩)
    (transferEvent := 64833) (summaryTransferEvent := 64834)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64830.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult63935.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult64835

namespace SemanticResult64840
def owner : Owner := ⟨.program ⟨214⟩, ⟨27443⟩⟩
def rawTerms : List Term := Proof.Events253.exact64840RawTerms
def summary : Bound := (.finite 28448551816593413384009941044)
def resultEvent : Nat := 64840
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64840.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult64835.owner)
    (rightOwner := SemanticResult63723.owner)
    (leftResult := 64835) (rightResult := 63723)
    (leftActual := SemanticResult64835.actual selector witness)
    (rightActual := SemanticResult63723.actual selector witness)
    (leftRaw := SemanticResult64835.rawTerms)
    (rightRaw := SemanticResult63723.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 23706886606235022529910538292)
    (rightMaximum := 4741665210358390854099402752) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 64836) (rightBinding := 64837)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27226⟩) (rightExpression := ⟨27442⟩)
    (transferEvent := 64838) (summaryTransferEvent := 64839)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64835.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult63723.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult64840

namespace SemanticResult64845
def owner : Owner := ⟨.program ⟨214⟩, ⟨27660⟩⟩
def rawTerms : List Term := Proof.Events253.exact64845RawTerms
def summary : Bound := (.finite 33190381535015453579890655284)
def resultEvent : Nat := 64845
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64845.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult64840.owner)
    (rightOwner := SemanticResult63511.owner)
    (leftResult := 64840) (rightResult := 63511)
    (leftActual := SemanticResult64840.actual selector witness)
    (rightActual := SemanticResult63511.actual selector witness)
    (leftRaw := SemanticResult64840.rawTerms)
    (rightRaw := SemanticResult63511.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 28448551816593413384009941044)
    (rightMaximum := 4741829718422040195880714240) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 64841) (rightBinding := 64842)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27443⟩) (rightExpression := ⟨27659⟩)
    (transferEvent := 64843) (summaryTransferEvent := 64844)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64840.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult63511.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult64845

namespace SemanticResult64850
def owner : Owner := ⟨.program ⟨214⟩, ⟨27877⟩⟩
def rawTerms : List Term := Proof.Events253.exact64850RawTerms
def summary : Bound := (.finite 37932293507469318446662025268)
def resultEvent : Nat := 64850
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64850.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult64845.owner)
    (rightOwner := SemanticResult63299.owner)
    (leftResult := 64845) (rightResult := 63299)
    (leftActual := SemanticResult64845.actual selector witness)
    (rightActual := SemanticResult63299.actual selector witness)
    (leftRaw := SemanticResult64845.rawTerms)
    (rightRaw := SemanticResult63299.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 33190381535015453579890655284)
    (rightMaximum := 4741911972453864866771369984) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 64846) (rightBinding := 64847)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27660⟩) (rightExpression := ⟨27876⟩)
    (transferEvent := 64848) (summaryTransferEvent := 64849)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64845.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult63299.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult64850

namespace SemanticResult64855
def owner : Owner := ⟨.program ⟨214⟩, ⟨28094⟩⟩
def rawTerms : List Term := Proof.Events253.exact64855RawTerms
def summary : Bound := (.finite 42674369987986832655214706740)
def resultEvent : Nat := 64855
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64855.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult64850.owner)
    (rightOwner := SemanticResult63087.owner)
    (leftResult := 64850) (rightResult := 63087)
    (leftActual := SemanticResult64850.actual selector witness)
    (rightActual := SemanticResult63087.actual selector witness)
    (leftRaw := SemanticResult64850.rawTerms)
    (rightRaw := SemanticResult63087.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 37932293507469318446662025268)
    (rightMaximum := 4742076480517514208552681472) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 64851) (rightBinding := 64852)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27877⟩) (rightExpression := ⟨28093⟩)
    (transferEvent := 64853) (summaryTransferEvent := 64854)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64850.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult63087.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult64855

namespace SemanticResult64860
def owner : Owner := ⟨.program ⟨214⟩, ⟨28311⟩⟩
def rawTerms : List Term := Proof.Events253.exact64860RawTerms
def summary : Bound := (.finite 47416693230599820876439355444)
def resultEvent : Nat := 64860
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64860.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult64855.owner)
    (rightOwner := SemanticResult62875.owner)
    (leftResult := 64855) (rightResult := 62875)
    (leftActual := SemanticResult64855.actual selector witness)
    (rightActual := SemanticResult62875.actual selector witness)
    (leftRaw := SemanticResult64855.rawTerms)
    (rightRaw := SemanticResult62875.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 42674369987986832655214706740)
    (rightMaximum := 4742323242612988221224648704) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 64856) (rightBinding := 64857)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28094⟩) (rightExpression := ⟨28310⟩)
    (transferEvent := 64858) (summaryTransferEvent := 64859)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64855.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult62875.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult64860

namespace SemanticResult64865
def owner : Owner := ⟨.program ⟨214⟩, ⟨28528⟩⟩
def rawTerms : List Term := Proof.Events253.exact64865RawTerms
def summary : Bound := (.finite 52159098727244633768554659892)
def resultEvent : Nat := 64865
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64865.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult64860.owner)
    (rightOwner := SemanticResult62663.owner)
    (leftResult := 64860) (rightResult := 62663)
    (leftActual := SemanticResult64860.actual selector witness)
    (rightActual := SemanticResult62663.actual selector witness)
    (leftRaw := SemanticResult64860.rawTerms)
    (rightRaw := SemanticResult62663.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 47416693230599820876439355444)
    (rightMaximum := 4742405496644812892115304448) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 64861) (rightBinding := 64862)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28311⟩) (rightExpression := ⟨28527⟩)
    (transferEvent := 64863) (summaryTransferEvent := 64864)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64860.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult62663.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult64865

namespace SemanticResult64870
def owner : Owner := ⟨.program ⟨214⟩, ⟨28745⟩⟩
def rawTerms : List Term := Proof.Events253.exact64870RawTerms
def summary : Bound := (.finite 56901750985984920673341931572)
def resultEvent : Nat := 64870
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64870.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult64865.owner)
    (rightOwner := SemanticResult62451.owner)
    (leftResult := 64865) (rightResult := 62451)
    (leftActual := SemanticResult64865.actual selector witness)
    (rightActual := SemanticResult62451.actual selector witness)
    (leftRaw := SemanticResult64865.rawTerms)
    (rightRaw := SemanticResult62451.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52159098727244633768554659892)
    (rightMaximum := 4742652258740286904787271680) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 64866) (rightBinding := 64867)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28528⟩) (rightExpression := ⟨28744⟩)
    (transferEvent := 64868) (summaryTransferEvent := 64869)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64865.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult62451.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult64870

namespace SemanticResult64875
def owner : Owner := ⟨.program ⟨214⟩, ⟨28962⟩⟩
def rawTerms : List Term := Proof.Events253.exact64875RawTerms
def summary : Bound := (.finite 61644567752788856919910514740)
def resultEvent : Nat := 64875
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64875.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult64870.owner)
    (rightOwner := SemanticResult62239.owner)
    (leftResult := 64870) (rightResult := 62239)
    (leftActual := SemanticResult64870.actual selector witness)
    (rightActual := SemanticResult62239.actual selector witness)
    (leftRaw := SemanticResult64870.rawTerms)
    (rightRaw := SemanticResult62239.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 56901750985984920673341931572)
    (rightMaximum := 4742816766803936246568583168) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 64871) (rightBinding := 64872)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28745⟩) (rightExpression := ⟨28961⟩)
    (transferEvent := 64873) (summaryTransferEvent := 64874)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64870.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult62239.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult64875

namespace SemanticResult64880
def owner : Owner := ⟨.program ⟨214⟩, ⟨29179⟩⟩
def rawTerms : List Term := Proof.Events253.exact64880RawTerms
def summary : Bound := (.finite 66387466773624617837369753652)
def resultEvent : Nat := 64880
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult64880.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult64875.owner)
    (rightOwner := SemanticResult62027.owner)
    (leftResult := 64875) (rightResult := 62027)
    (leftActual := SemanticResult64875.actual selector witness)
    (rightActual := SemanticResult62027.actual selector witness)
    (leftRaw := SemanticResult64875.rawTerms)
    (rightRaw := SemanticResult62027.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 61644567752788856919910514740)
    (rightMaximum := 4742899020835760917459238912) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 64876) (rightBinding := 64877)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28962⟩) (rightExpression := ⟨29178⟩)
    (transferEvent := 64878) (summaryTransferEvent := 64879)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult64875.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult62027.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult64880

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
