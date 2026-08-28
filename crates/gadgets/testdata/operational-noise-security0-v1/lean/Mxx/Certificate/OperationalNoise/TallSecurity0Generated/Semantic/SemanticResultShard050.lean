import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard050
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard045
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard046
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard047
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard048
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard049

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult5873
def owner : Owner := ⟨.program ⟨214⟩, ⟨6760⟩⟩
def rawTerms : List Term := Proof.Events022.exact5873RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5873
def producerEvent : Nat := 5872
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5873.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 5871 .coefficient), 0, .large, .identity (.predecessor 0 5871 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult5873

namespace SemanticResult5878
def owner : Owner := ⟨.program ⟨214⟩, ⟨7650⟩⟩
def rawTerms : List Term := Proof.Events022.exact5878RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5878
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5878.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge5877.working .exactZero) := by
  apply operatorSubMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge5877.frameStart)
    (transferEvent := 5876) (owner := owner)
    (leftResult := 5873) (rightResult := 5873)
    (working := LeftOperatorMerge5877.working)
    (reconstruction := LeftOperatorMerge5877.reconstruction)
    (leftReference := .predecessor 0 5874 .coefficient) (rightReference := .predecessor 1 5875 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult5873.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5873.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge5877.operationAgreement
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
end SemanticResult5878

namespace SemanticResult5882
def owner : Owner := ⟨.program ⟨214⟩, ⟨7651⟩⟩
def rawTerms : List Term := Proof.Events022.exact5882RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5882
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5882.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5879) (rightBinding := 5880)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7650⟩) (rightExpression := ⟨7631⟩)
    (transferEvent := 5881)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5878.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5867.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5882

namespace SemanticResult5886
def owner : Owner := ⟨.program ⟨214⟩, ⟨7652⟩⟩
def rawTerms : List Term := Proof.Events022.exact5886RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5886
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5886.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5883) (rightBinding := 5884)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7651⟩) (rightExpression := ⟨7632⟩)
    (transferEvent := 5885)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5882.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5847.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5886

namespace SemanticResult5890
def owner : Owner := ⟨.program ⟨214⟩, ⟨7653⟩⟩
def rawTerms : List Term := Proof.Events023.exact5890RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5890
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5890.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5887) (rightBinding := 5888)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7652⟩) (rightExpression := ⟨7633⟩)
    (transferEvent := 5889)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5886.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5827.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5890

namespace SemanticResult5894
def owner : Owner := ⟨.program ⟨214⟩, ⟨7654⟩⟩
def rawTerms : List Term := Proof.Events023.exact5894RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5894
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5894.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5891) (rightBinding := 5892)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7653⟩) (rightExpression := ⟨7634⟩)
    (transferEvent := 5893)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5890.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5807.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5894

namespace SemanticResult5898
def owner : Owner := ⟨.program ⟨214⟩, ⟨7655⟩⟩
def rawTerms : List Term := Proof.Events023.exact5898RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5898
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5898.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5895) (rightBinding := 5896)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7654⟩) (rightExpression := ⟨7635⟩)
    (transferEvent := 5897)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5894.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5787.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5898

namespace SemanticResult5902
def owner : Owner := ⟨.program ⟨214⟩, ⟨7656⟩⟩
def rawTerms : List Term := Proof.Events023.exact5902RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5902
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5902.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5899) (rightBinding := 5900)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7655⟩) (rightExpression := ⟨7636⟩)
    (transferEvent := 5901)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5898.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5767.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5902

namespace SemanticResult5906
def owner : Owner := ⟨.program ⟨214⟩, ⟨7657⟩⟩
def rawTerms : List Term := Proof.Events023.exact5906RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5906
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5906.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5903) (rightBinding := 5904)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7656⟩) (rightExpression := ⟨7637⟩)
    (transferEvent := 5905)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5902.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5747.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5906

namespace SemanticResult5910
def owner : Owner := ⟨.program ⟨214⟩, ⟨7658⟩⟩
def rawTerms : List Term := Proof.Events023.exact5910RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5910
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5910.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5907) (rightBinding := 5908)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7657⟩) (rightExpression := ⟨7638⟩)
    (transferEvent := 5909)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5906.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5727.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5910

namespace SemanticResult5914
def owner : Owner := ⟨.program ⟨214⟩, ⟨7659⟩⟩
def rawTerms : List Term := Proof.Events023.exact5914RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5914
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5914.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5911) (rightBinding := 5912)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7658⟩) (rightExpression := ⟨7639⟩)
    (transferEvent := 5913)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5910.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5707.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5914

namespace SemanticResult5918
def owner : Owner := ⟨.program ⟨214⟩, ⟨7660⟩⟩
def rawTerms : List Term := Proof.Events023.exact5918RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5918
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5918.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5915) (rightBinding := 5916)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7659⟩) (rightExpression := ⟨7640⟩)
    (transferEvent := 5917)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5914.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5687.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5918

namespace SemanticResult5922
def owner : Owner := ⟨.program ⟨214⟩, ⟨7661⟩⟩
def rawTerms : List Term := Proof.Events023.exact5922RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5922
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5922.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5919) (rightBinding := 5920)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7660⟩) (rightExpression := ⟨7641⟩)
    (transferEvent := 5921)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5918.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5667.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5922

namespace SemanticResult5926
def owner : Owner := ⟨.program ⟨214⟩, ⟨7662⟩⟩
def rawTerms : List Term := Proof.Events023.exact5926RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5926
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5926.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5923) (rightBinding := 5924)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7661⟩) (rightExpression := ⟨7642⟩)
    (transferEvent := 5925)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5922.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5647.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5926

namespace SemanticResult5930
def owner : Owner := ⟨.program ⟨214⟩, ⟨7663⟩⟩
def rawTerms : List Term := Proof.Events023.exact5930RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5930
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5930.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5927) (rightBinding := 5928)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7662⟩) (rightExpression := ⟨7643⟩)
    (transferEvent := 5929)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5926.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5627.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5930

namespace SemanticResult5934
def owner : Owner := ⟨.program ⟨214⟩, ⟨7664⟩⟩
def rawTerms : List Term := Proof.Events023.exact5934RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5934
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5934.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5931) (rightBinding := 5932)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7663⟩) (rightExpression := ⟨7644⟩)
    (transferEvent := 5933)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5930.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5607.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5934

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
