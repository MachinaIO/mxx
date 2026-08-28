import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard637
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard636

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult89899
def owner : Owner := ⟨.program ⟨214⟩, ⟨6723⟩⟩
def rawTerms : List Term := Proof.Events351.exact89899RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89899
def producerEvent : Nat := 89898
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89899.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 89317, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult89899

namespace SemanticResult89902
def owner : Owner := ⟨.program ⟨214⟩, ⟨6721⟩⟩
def rawTerms : List Term := Proof.Events351.exact89902RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89902
def producerEvent : Nat := 89901
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89902.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 89317, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult89902

namespace SemanticResult89905
def owner : Owner := ⟨.program ⟨214⟩, ⟨6719⟩⟩
def rawTerms : List Term := Proof.Events351.exact89905RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89905
def producerEvent : Nat := 89904
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89905.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 89317, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult89905

namespace SemanticResult89908
def owner : Owner := ⟨.program ⟨214⟩, ⟨6717⟩⟩
def rawTerms : List Term := Proof.Events351.exact89908RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89908
def producerEvent : Nat := 89907
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89908.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 89317, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult89908

namespace SemanticResult89911
def owner : Owner := ⟨.program ⟨214⟩, ⟨6715⟩⟩
def rawTerms : List Term := Proof.Events351.exact89911RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89911
def producerEvent : Nat := 89910
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89911.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 89317, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult89911

namespace SemanticResult89914
def owner : Owner := ⟨.program ⟨214⟩, ⟨6713⟩⟩
def rawTerms : List Term := Proof.Events351.exact89914RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89914
def producerEvent : Nat := 89913
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89914.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 89317, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult89914

namespace SemanticResult89917
def owner : Owner := ⟨.program ⟨214⟩, ⟨6711⟩⟩
def rawTerms : List Term := Proof.Events351.exact89917RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89917
def producerEvent : Nat := 89916
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89917.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 89317, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult89917

namespace SemanticResult89920
def owner : Owner := ⟨.program ⟨214⟩, ⟨6709⟩⟩
def rawTerms : List Term := Proof.Events351.exact89920RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89920
def producerEvent : Nat := 89919
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89920.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 89317, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult89920

namespace SemanticResult89924
def owner : Owner := ⟨.program ⟨214⟩, ⟨6795⟩⟩
def rawTerms : List Term := Proof.Events351.exact89924RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89924
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89924.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89921) (rightBinding := 89922)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6709⟩) (rightExpression := ⟨6711⟩)
    (transferEvent := 89923)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89920.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89917.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89924

namespace SemanticResult89928
def owner : Owner := ⟨.program ⟨214⟩, ⟨6796⟩⟩
def rawTerms : List Term := Proof.Events351.exact89928RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89928
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89928.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89925) (rightBinding := 89926)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6795⟩) (rightExpression := ⟨6713⟩)
    (transferEvent := 89927)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89924.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89914.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89928

namespace SemanticResult89932
def owner : Owner := ⟨.program ⟨214⟩, ⟨6797⟩⟩
def rawTerms : List Term := Proof.Events351.exact89932RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89932
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89932.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89929) (rightBinding := 89930)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6796⟩) (rightExpression := ⟨6715⟩)
    (transferEvent := 89931)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89928.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89911.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89932

namespace SemanticResult89936
def owner : Owner := ⟨.program ⟨214⟩, ⟨6798⟩⟩
def rawTerms : List Term := Proof.Events351.exact89936RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89936
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89936.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89933) (rightBinding := 89934)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6797⟩) (rightExpression := ⟨6717⟩)
    (transferEvent := 89935)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89932.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89908.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89936

namespace SemanticResult89940
def owner : Owner := ⟨.program ⟨214⟩, ⟨6799⟩⟩
def rawTerms : List Term := Proof.Events351.exact89940RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89940
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89940.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89937) (rightBinding := 89938)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6798⟩) (rightExpression := ⟨6719⟩)
    (transferEvent := 89939)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89936.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89905.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89940

namespace SemanticResult89944
def owner : Owner := ⟨.program ⟨214⟩, ⟨6800⟩⟩
def rawTerms : List Term := Proof.Events351.exact89944RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89944
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89944.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89941) (rightBinding := 89942)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6799⟩) (rightExpression := ⟨6721⟩)
    (transferEvent := 89943)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89940.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89902.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89944

namespace SemanticResult89948
def owner : Owner := ⟨.program ⟨214⟩, ⟨6801⟩⟩
def rawTerms : List Term := Proof.Events351.exact89948RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89948
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89948.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89945) (rightBinding := 89946)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6800⟩) (rightExpression := ⟨6723⟩)
    (transferEvent := 89947)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89944.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89899.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89948

namespace SemanticResult89952
def owner : Owner := ⟨.program ⟨214⟩, ⟨6802⟩⟩
def rawTerms : List Term := Proof.Events351.exact89952RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89952
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89952.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89949) (rightBinding := 89950)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6801⟩) (rightExpression := ⟨6725⟩)
    (transferEvent := 89951)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89948.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89896.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89952

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
