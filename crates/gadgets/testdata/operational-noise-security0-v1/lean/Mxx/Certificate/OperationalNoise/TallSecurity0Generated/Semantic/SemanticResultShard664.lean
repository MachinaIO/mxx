import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard664
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard050
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard161
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard648
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard650
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard651
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard652
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard654
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard655
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard657
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard658
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard659
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard661
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard662
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard663

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult94007
def owner : Owner := ⟨.program ⟨214⟩, ⟨7216⟩⟩
def rawTerms : List Term := Proof.Events367.exact94007RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 94007
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94007.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge94006.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge94006.frameStart)
    (transferEvent := 94005) (owner := owner)
    (leftResult := 79790) (rightResult := 5873)
    (working := LeftOperatorMerge94006.working)
    (reconstruction := LeftOperatorMerge94006.reconstruction)
    (leftReference := .predecessor 0 94003 .coefficient) (rightReference := .predecessor 1 94004 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5873.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge94006.operationAgreement
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
end SemanticResult94007

namespace SemanticResult94011
def owner : Owner := ⟨.program ⟨214⟩, ⟨7753⟩⟩
def rawTerms : List Term := Proof.Events367.exact94011RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 94011
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94011.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 94008) (rightBinding := 94009)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7216⟩) (rightExpression := ⟨6626⟩)
    (transferEvent := 94010)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94007.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult94002.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult94011

namespace SemanticResult94017
def owner : Owner := ⟨.program ⟨214⟩, ⟨7754⟩⟩
def rawTerms : List Term := Proof.Events367.exact94017RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 94017
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94017.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 94014) (survivorTransfer := 94015)
    (survivorEvent := 94016) (resultEvent := resultEvent)
    (rightCoefficientProducer := 20907)
    (owner := owner) (leftOwner := SemanticResult94011.owner)
    (rightOwner := SemanticResult20908.owner)
    (leftResult := 94011) (rightResult := 20908)
    (leftBinding := 94012) (rightBinding := 94013)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7753⟩) (rightExpression := ⟨74⟩)
    (leftActual := SemanticResult94011.actual selector witness)
    (rightActual := SemanticResult20908.actual selector witness)
    (leftRaw := SemanticResult94011.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨74⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound20907.actual selector witness)
    (survivorMagnitude := LeftBound94015.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94011.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20908.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)
  · exact LeftBound94015.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult94017

namespace SemanticResult94024
def owner : Owner := ⟨.program ⟨214⟩, ⟨7808⟩⟩
def rawTerms : List Term := Proof.Events367.exact94024RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 94024
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94024.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge94021.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult94017.owner)
    (rightOwner := SemanticResult94017.owner)
    (leftResult := 94017) (rightResult := 94017)
    (leftActual := SemanticResult94017.actual selector witness)
    (rightActual := SemanticResult94017.actual selector witness)
    (leftRaw := SemanticResult94017.rawTerms)
    (rightRaw := SemanticResult94017.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 94018) (rightBinding := 94019)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7754⟩) (rightExpression := ⟨7754⟩)
    (coefficientTransfer := 94020) (summaryTransfer := 94023)
    (base := LeftOperatorMerge94021.base)
    (reconstruction := LeftOperatorMerge94021.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94017.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult94017.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge94021.operationAgreement
  · rfl
  · decide
end SemanticResult94024

namespace SemanticResult94029
def owner : Owner := ⟨.program ⟨214⟩, ⟨26356⟩⟩
def rawTerms : List Term := Proof.Events367.exact94029RawTerms
def summary : Bound := (.finite 4741253940199267499646124084)
def resultEvent : Nat := 94029
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94029.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult94024.owner)
    (rightOwner := SemanticResult93997.owner)
    (leftResult := 94024) (rightResult := 93997)
    (leftActual := SemanticResult94024.actual selector witness)
    (rightActual := SemanticResult93997.actual selector witness)
    (leftRaw := SemanticResult94024.rawTerms)
    (rightRaw := SemanticResult93997.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 4741253940199267499646124032) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 94025) (rightBinding := 94026)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7808⟩) (rightExpression := ⟨26355⟩)
    (transferEvent := 94027) (summaryTransferEvent := 94028)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94024.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult93997.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult94029

namespace SemanticResult94034
def owner : Owner := ⟨.program ⟨214⟩, ⟨26562⟩⟩
def rawTerms : List Term := Proof.Events367.exact94034RawTerms
def summary : Bound := (.finite 9482549007414447334737575988)
def resultEvent : Nat := 94034
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94034.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult94029.owner)
    (rightOwner := SemanticResult93785.owner)
    (leftResult := 94029) (rightResult := 93785)
    (leftActual := SemanticResult94029.actual selector witness)
    (rightActual := SemanticResult93785.actual selector witness)
    (leftRaw := SemanticResult94029.rawTerms)
    (rightRaw := SemanticResult93785.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4741253940199267499646124084)
    (rightMaximum := 4741295067215179835091451904) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 94030) (rightBinding := 94031)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26356⟩) (rightExpression := ⟨26561⟩)
    (transferEvent := 94032) (summaryTransferEvent := 94033)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94029.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult93785.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult94034

namespace SemanticResult94039
def owner : Owner := ⟨.program ⟨214⟩, ⟨26779⟩⟩
def rawTerms : List Term := Proof.Events367.exact94039RawTerms
def summary : Bound := (.finite 14223885201645539505274355764)
def resultEvent : Nat := 94039
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94039.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult94034.owner)
    (rightOwner := SemanticResult93573.owner)
    (leftResult := 94034) (rightResult := 93573)
    (leftActual := SemanticResult94034.actual selector witness)
    (rightActual := SemanticResult93573.actual selector witness)
    (leftRaw := SemanticResult94034.rawTerms)
    (rightRaw := SemanticResult93573.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 9482549007414447334737575988)
    (rightMaximum := 4741336194231092170536779776) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 94035) (rightBinding := 94036)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26562⟩) (rightExpression := ⟨26778⟩)
    (transferEvent := 94037) (summaryTransferEvent := 94038)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94034.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult93573.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult94039

namespace SemanticResult94044
def owner : Owner := ⟨.program ⟨214⟩, ⟨26996⟩⟩
def rawTerms : List Term := Proof.Events367.exact94044RawTerms
def summary : Bound := (.finite 18965303649908456346701791284)
def resultEvent : Nat := 94044
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94044.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult94039.owner)
    (rightOwner := SemanticResult93361.owner)
    (leftResult := 94039) (rightResult := 93361)
    (leftActual := SemanticResult94039.actual selector witness)
    (rightActual := SemanticResult93361.actual selector witness)
    (leftRaw := SemanticResult94039.rawTerms)
    (rightRaw := SemanticResult93361.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 14223885201645539505274355764)
    (rightMaximum := 4741418448262916841427435520) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 94040) (rightBinding := 94041)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26779⟩) (rightExpression := ⟨26995⟩)
    (transferEvent := 94042) (summaryTransferEvent := 94043)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94039.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult93361.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult94044

namespace SemanticResult94049
def owner : Owner := ⟨.program ⟨214⟩, ⟨27213⟩⟩
def rawTerms : List Term := Proof.Events367.exact94049RawTerms
def summary : Bound := (.finite 23706886606235022529910538292)
def resultEvent : Nat := 94049
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94049.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult94044.owner)
    (rightOwner := SemanticResult93149.owner)
    (leftResult := 94044) (rightResult := 93149)
    (leftActual := SemanticResult94044.actual selector witness)
    (rightActual := SemanticResult93149.actual selector witness)
    (leftRaw := SemanticResult94044.rawTerms)
    (rightRaw := SemanticResult93149.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 18965303649908456346701791284)
    (rightMaximum := 4741582956326566183208747008) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 94045) (rightBinding := 94046)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26996⟩) (rightExpression := ⟨27212⟩)
    (transferEvent := 94047) (summaryTransferEvent := 94048)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94044.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult93149.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult94049

namespace SemanticResult94054
def owner : Owner := ⟨.program ⟨214⟩, ⟨27430⟩⟩
def rawTerms : List Term := Proof.Events367.exact94054RawTerms
def summary : Bound := (.finite 28448551816593413384009941044)
def resultEvent : Nat := 94054
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94054.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult94049.owner)
    (rightOwner := SemanticResult92937.owner)
    (leftResult := 94049) (rightResult := 92937)
    (leftActual := SemanticResult94049.actual selector witness)
    (rightActual := SemanticResult92937.actual selector witness)
    (leftRaw := SemanticResult94049.rawTerms)
    (rightRaw := SemanticResult92937.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 23706886606235022529910538292)
    (rightMaximum := 4741665210358390854099402752) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 94050) (rightBinding := 94051)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27213⟩) (rightExpression := ⟨27429⟩)
    (transferEvent := 94052) (summaryTransferEvent := 94053)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94049.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult92937.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult94054

namespace SemanticResult94059
def owner : Owner := ⟨.program ⟨214⟩, ⟨27647⟩⟩
def rawTerms : List Term := Proof.Events367.exact94059RawTerms
def summary : Bound := (.finite 33190381535015453579890655284)
def resultEvent : Nat := 94059
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94059.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult94054.owner)
    (rightOwner := SemanticResult92725.owner)
    (leftResult := 94054) (rightResult := 92725)
    (leftActual := SemanticResult94054.actual selector witness)
    (rightActual := SemanticResult92725.actual selector witness)
    (leftRaw := SemanticResult94054.rawTerms)
    (rightRaw := SemanticResult92725.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 28448551816593413384009941044)
    (rightMaximum := 4741829718422040195880714240) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 94055) (rightBinding := 94056)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27430⟩) (rightExpression := ⟨27646⟩)
    (transferEvent := 94057) (summaryTransferEvent := 94058)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94054.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult92725.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult94059

namespace SemanticResult94064
def owner : Owner := ⟨.program ⟨214⟩, ⟨27864⟩⟩
def rawTerms : List Term := Proof.Events367.exact94064RawTerms
def summary : Bound := (.finite 37932293507469318446662025268)
def resultEvent : Nat := 94064
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94064.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult94059.owner)
    (rightOwner := SemanticResult92513.owner)
    (leftResult := 94059) (rightResult := 92513)
    (leftActual := SemanticResult94059.actual selector witness)
    (rightActual := SemanticResult92513.actual selector witness)
    (leftRaw := SemanticResult94059.rawTerms)
    (rightRaw := SemanticResult92513.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 33190381535015453579890655284)
    (rightMaximum := 4741911972453864866771369984) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 94060) (rightBinding := 94061)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27647⟩) (rightExpression := ⟨27863⟩)
    (transferEvent := 94062) (summaryTransferEvent := 94063)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94059.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult92513.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult94064

namespace SemanticResult94069
def owner : Owner := ⟨.program ⟨214⟩, ⟨28081⟩⟩
def rawTerms : List Term := Proof.Events367.exact94069RawTerms
def summary : Bound := (.finite 42674369987986832655214706740)
def resultEvent : Nat := 94069
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94069.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult94064.owner)
    (rightOwner := SemanticResult92301.owner)
    (leftResult := 94064) (rightResult := 92301)
    (leftActual := SemanticResult94064.actual selector witness)
    (rightActual := SemanticResult92301.actual selector witness)
    (leftRaw := SemanticResult94064.rawTerms)
    (rightRaw := SemanticResult92301.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 37932293507469318446662025268)
    (rightMaximum := 4742076480517514208552681472) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 94065) (rightBinding := 94066)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27864⟩) (rightExpression := ⟨28080⟩)
    (transferEvent := 94067) (summaryTransferEvent := 94068)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94064.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult92301.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult94069

namespace SemanticResult94074
def owner : Owner := ⟨.program ⟨214⟩, ⟨28298⟩⟩
def rawTerms : List Term := Proof.Events367.exact94074RawTerms
def summary : Bound := (.finite 47416693230599820876439355444)
def resultEvent : Nat := 94074
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94074.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult94069.owner)
    (rightOwner := SemanticResult92089.owner)
    (leftResult := 94069) (rightResult := 92089)
    (leftActual := SemanticResult94069.actual selector witness)
    (rightActual := SemanticResult92089.actual selector witness)
    (leftRaw := SemanticResult94069.rawTerms)
    (rightRaw := SemanticResult92089.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 42674369987986832655214706740)
    (rightMaximum := 4742323242612988221224648704) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 94070) (rightBinding := 94071)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28081⟩) (rightExpression := ⟨28297⟩)
    (transferEvent := 94072) (summaryTransferEvent := 94073)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94069.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult92089.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult94074

namespace SemanticResult94079
def owner : Owner := ⟨.program ⟨214⟩, ⟨28515⟩⟩
def rawTerms : List Term := Proof.Events367.exact94079RawTerms
def summary : Bound := (.finite 52159098727244633768554659892)
def resultEvent : Nat := 94079
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94079.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult94074.owner)
    (rightOwner := SemanticResult91877.owner)
    (leftResult := 94074) (rightResult := 91877)
    (leftActual := SemanticResult94074.actual selector witness)
    (rightActual := SemanticResult91877.actual selector witness)
    (leftRaw := SemanticResult94074.rawTerms)
    (rightRaw := SemanticResult91877.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 47416693230599820876439355444)
    (rightMaximum := 4742405496644812892115304448) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 94075) (rightBinding := 94076)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28298⟩) (rightExpression := ⟨28514⟩)
    (transferEvent := 94077) (summaryTransferEvent := 94078)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94074.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult91877.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult94079

namespace SemanticResult94084
def owner : Owner := ⟨.program ⟨214⟩, ⟨28732⟩⟩
def rawTerms : List Term := Proof.Events367.exact94084RawTerms
def summary : Bound := (.finite 56901750985984920673341931572)
def resultEvent : Nat := 94084
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult94084.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult94079.owner)
    (rightOwner := SemanticResult91665.owner)
    (leftResult := 94079) (rightResult := 91665)
    (leftActual := SemanticResult94079.actual selector witness)
    (rightActual := SemanticResult91665.actual selector witness)
    (leftRaw := SemanticResult94079.rawTerms)
    (rightRaw := SemanticResult91665.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52159098727244633768554659892)
    (rightMaximum := 4742652258740286904787271680) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 94080) (rightBinding := 94081)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28515⟩) (rightExpression := ⟨28731⟩)
    (transferEvent := 94082) (summaryTransferEvent := 94083)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94079.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult91665.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult94084

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
