import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard051
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard007
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard043
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard044
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard045
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard050

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult5938
def owner : Owner := ⟨.program ⟨214⟩, ⟨7665⟩⟩
def rawTerms : List Term := Proof.Events023.exact5938RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5938
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5938.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5935) (rightBinding := 5936)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7664⟩) (rightExpression := ⟨7645⟩)
    (transferEvent := 5937)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5934.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5587.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5938

namespace SemanticResult5942
def owner : Owner := ⟨.program ⟨214⟩, ⟨7666⟩⟩
def rawTerms : List Term := Proof.Events023.exact5942RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5942
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5942.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5939) (rightBinding := 5940)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7665⟩) (rightExpression := ⟨7646⟩)
    (transferEvent := 5941)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5938.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5567.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5942

namespace SemanticResult5946
def owner : Owner := ⟨.program ⟨214⟩, ⟨7667⟩⟩
def rawTerms : List Term := Proof.Events023.exact5946RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5946
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5946.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5943) (rightBinding := 5944)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7666⟩) (rightExpression := ⟨7647⟩)
    (transferEvent := 5945)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5942.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5547.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5946

namespace SemanticResult5950
def owner : Owner := ⟨.program ⟨214⟩, ⟨7668⟩⟩
def rawTerms : List Term := Proof.Events023.exact5950RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5950
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5950.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5947) (rightBinding := 5948)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7667⟩) (rightExpression := ⟨7648⟩)
    (transferEvent := 5949)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5946.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5527.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5950

namespace SemanticResult5954
def owner : Owner := ⟨.program ⟨214⟩, ⟨7795⟩⟩
def rawTerms : List Term := Proof.Events023.exact5954RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5954
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5954.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5951) (rightBinding := 5952)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7668⟩) (rightExpression := ⟨7649⟩)
    (transferEvent := 5953)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5950.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5507.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5954

namespace SemanticResult5957
def owner : Owner := ⟨.program ⟨214⟩, ⟨7885⟩⟩
def rawTerms : List Term := Proof.Events023.exact5957RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5957
def producerEvent : Nat := 5956
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5957.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult5957

namespace SemanticResult5961
def owner : Owner := ⟨.program ⟨214⟩, ⟨7886⟩⟩
def rawTerms : List Term := Proof.Events023.exact5961RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5961
def producerEvent : Nat := 5960
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5961.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 5958 .coefficient) (.value (.predecessor 1 5959 .coefficient)), 0, .finite 8192, .scale (.predecessor 0 5958 .coefficient) (.value (.predecessor 1 5959 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult5961

namespace SemanticResult5964
def owner : Owner := ⟨.program ⟨214⟩, ⟨6745⟩⟩
def rawTerms : List Term := Proof.Events023.exact5964RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5964
def producerEvent : Nat := 5963
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5964.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult5964

namespace SemanticResult5969
def owner : Owner := ⟨.program ⟨214⟩, ⟨7887⟩⟩
def rawTerms : List Term := Proof.Events023.exact5969RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5969
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5969.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge5968.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge5968.frameStart)
    (transferEvent := 5967) (owner := owner)
    (leftResult := 5964) (rightResult := 5961)
    (working := LeftOperatorMerge5968.working)
    (reconstruction := LeftOperatorMerge5968.reconstruction)
    (leftReference := .predecessor 0 5965 .coefficient) (rightReference := .predecessor 1 5966 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult5964.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5961.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge5968.operationAgreement
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
end SemanticResult5969

namespace SemanticResult5974
def owner : Owner := ⟨.program ⟨214⟩, ⟨7911⟩⟩
def rawTerms : List Term := Proof.Events023.exact5974RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5974
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5974.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge5973.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge5973.frameStart)
    (transferEvent := 5972) (owner := owner)
    (leftResult := 5969) (rightResult := 5487)
    (working := LeftOperatorMerge5973.working)
    (reconstruction := LeftOperatorMerge5973.reconstruction)
    (leftReference := .predecessor 0 5970 .coefficient) (rightReference := .predecessor 1 5971 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult5969.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5487.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge5973.operationAgreement
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
end SemanticResult5974

namespace SemanticResult5979
def owner : Owner := ⟨.program ⟨214⟩, ⟨7917⟩⟩
def rawTerms : List Term := Proof.Events023.exact5979RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5979
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5979.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge5978.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge5978.frameStart)
    (transferEvent := 5977) (owner := owner)
    (leftResult := 5974) (rightResult := 5476)
    (working := LeftOperatorMerge5978.working)
    (reconstruction := LeftOperatorMerge5978.reconstruction)
    (leftReference := .predecessor 0 5975 .coefficient) (rightReference := .predecessor 1 5976 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult5974.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5476.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge5978.operationAgreement
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
end SemanticResult5979

namespace SemanticResult5984
def owner : Owner := ⟨.program ⟨214⟩, ⟨6615⟩⟩
def rawTerms : List Term := Proof.Events023.exact5984RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5984
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5984.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge5983.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge5983.frameStart)
    (transferEvent := 5982) (owner := owner)
    (leftResult := 2) (rightResult := 829)
    (working := LeftOperatorMerge5983.working)
    (reconstruction := LeftOperatorMerge5983.reconstruction)
    (leftReference := .predecessor 0 5980 .coefficient) (rightReference := .predecessor 1 5981 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult2.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult829.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge5983.operationAgreement
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
end SemanticResult5984

namespace SemanticResult5987
def owner : Owner := ⟨.program ⟨214⟩, ⟨6687⟩⟩
def rawTerms : List Term := Proof.Events023.exact5987RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5987
def producerEvent : Nat := 5986
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5987.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult5987

namespace SemanticResult5991
def owner : Owner := ⟨.program ⟨214⟩, ⟨6688⟩⟩
def rawTerms : List Term := Proof.Events023.exact5991RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5991
def producerEvent : Nat := 5990
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5991.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 5988 .coefficient) (.value (.predecessor 1 5989 .coefficient)), 0, .finite 8192, .scale (.predecessor 0 5988 .coefficient) (.value (.predecessor 1 5989 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult5991

namespace SemanticResult5994
def owner : Owner := ⟨.program ⟨214⟩, ⟨6748⟩⟩
def rawTerms : List Term := Proof.Events023.exact5994RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5994
def producerEvent : Nat := 5993
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5994.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult5994

namespace SemanticResult5997
def owner : Owner := ⟨.program ⟨214⟩, ⟨7821⟩⟩
def rawTerms : List Term := Proof.Events023.exact5997RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5997
def producerEvent : Nat := 5996
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5997.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult5997

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
