import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard436
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard435

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult60691
def owner : Owner := ⟨.program ⟨214⟩, ⟨6719⟩⟩
def rawTerms : List Term := Proof.Events237.exact60691RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60691
def producerEvent : Nat := 60690
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60691.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 60103, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult60691

namespace SemanticResult60694
def owner : Owner := ⟨.program ⟨214⟩, ⟨6717⟩⟩
def rawTerms : List Term := Proof.Events237.exact60694RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60694
def producerEvent : Nat := 60693
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60694.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 60103, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult60694

namespace SemanticResult60697
def owner : Owner := ⟨.program ⟨214⟩, ⟨6715⟩⟩
def rawTerms : List Term := Proof.Events237.exact60697RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60697
def producerEvent : Nat := 60696
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60697.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 60103, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult60697

namespace SemanticResult60700
def owner : Owner := ⟨.program ⟨214⟩, ⟨6713⟩⟩
def rawTerms : List Term := Proof.Events237.exact60700RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60700
def producerEvent : Nat := 60699
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60700.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 60103, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult60700

namespace SemanticResult60703
def owner : Owner := ⟨.program ⟨214⟩, ⟨6711⟩⟩
def rawTerms : List Term := Proof.Events237.exact60703RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60703
def producerEvent : Nat := 60702
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60703.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 60103, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult60703

namespace SemanticResult60706
def owner : Owner := ⟨.program ⟨214⟩, ⟨6709⟩⟩
def rawTerms : List Term := Proof.Events237.exact60706RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60706
def producerEvent : Nat := 60705
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60706.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 60103, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult60706

namespace SemanticResult60710
def owner : Owner := ⟨.program ⟨214⟩, ⟨6795⟩⟩
def rawTerms : List Term := Proof.Events237.exact60710RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60710
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60710.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60707) (rightBinding := 60708)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6709⟩) (rightExpression := ⟨6711⟩)
    (transferEvent := 60709)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60706.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60703.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60710

namespace SemanticResult60714
def owner : Owner := ⟨.program ⟨214⟩, ⟨6796⟩⟩
def rawTerms : List Term := Proof.Events237.exact60714RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60714
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60714.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60711) (rightBinding := 60712)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6795⟩) (rightExpression := ⟨6713⟩)
    (transferEvent := 60713)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60710.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60700.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60714

namespace SemanticResult60718
def owner : Owner := ⟨.program ⟨214⟩, ⟨6797⟩⟩
def rawTerms : List Term := Proof.Events237.exact60718RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60718
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60718.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60715) (rightBinding := 60716)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6796⟩) (rightExpression := ⟨6715⟩)
    (transferEvent := 60717)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60714.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60697.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60718

namespace SemanticResult60722
def owner : Owner := ⟨.program ⟨214⟩, ⟨6798⟩⟩
def rawTerms : List Term := Proof.Events237.exact60722RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60722
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60722.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60719) (rightBinding := 60720)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6797⟩) (rightExpression := ⟨6717⟩)
    (transferEvent := 60721)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60718.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60694.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60722

namespace SemanticResult60726
def owner : Owner := ⟨.program ⟨214⟩, ⟨6799⟩⟩
def rawTerms : List Term := Proof.Events237.exact60726RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60726
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60726.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60723) (rightBinding := 60724)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6798⟩) (rightExpression := ⟨6719⟩)
    (transferEvent := 60725)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60722.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60691.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60726

namespace SemanticResult60730
def owner : Owner := ⟨.program ⟨214⟩, ⟨6800⟩⟩
def rawTerms : List Term := Proof.Events237.exact60730RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60730
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60730.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60727) (rightBinding := 60728)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6799⟩) (rightExpression := ⟨6721⟩)
    (transferEvent := 60729)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60726.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60688.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60730

namespace SemanticResult60734
def owner : Owner := ⟨.program ⟨214⟩, ⟨6801⟩⟩
def rawTerms : List Term := Proof.Events237.exact60734RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60734
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60734.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60731) (rightBinding := 60732)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6800⟩) (rightExpression := ⟨6723⟩)
    (transferEvent := 60733)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60730.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60685.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60734

namespace SemanticResult60738
def owner : Owner := ⟨.program ⟨214⟩, ⟨6802⟩⟩
def rawTerms : List Term := Proof.Events237.exact60738RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60738
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60738.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60735) (rightBinding := 60736)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6801⟩) (rightExpression := ⟨6725⟩)
    (transferEvent := 60737)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60734.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60682.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60738

namespace SemanticResult60742
def owner : Owner := ⟨.program ⟨214⟩, ⟨6803⟩⟩
def rawTerms : List Term := Proof.Events237.exact60742RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60742
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60742.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60739) (rightBinding := 60740)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6802⟩) (rightExpression := ⟨6727⟩)
    (transferEvent := 60741)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60738.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60679.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60742

namespace SemanticResult60746
def owner : Owner := ⟨.program ⟨214⟩, ⟨6804⟩⟩
def rawTerms : List Term := Proof.Events237.exact60746RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60746
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60746.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60743) (rightBinding := 60744)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6803⟩) (rightExpression := ⟨6729⟩)
    (transferEvent := 60745)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60742.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60676.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60746

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
