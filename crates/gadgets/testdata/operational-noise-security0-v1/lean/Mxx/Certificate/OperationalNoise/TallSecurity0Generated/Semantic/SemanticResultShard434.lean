import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard434
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard432
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard433

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult60549
def owner : Owner := ⟨.program ⟨214⟩, ⟨17337⟩⟩
def rawTerms : List Term := Proof.Events236.exact60549RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60549
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60549.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60546) (rightBinding := 60547)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15371⟩) (rightExpression := ⟨17336⟩)
    (transferEvent := 60548)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60545.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60468.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60549

namespace SemanticResult60553
def owner : Owner := ⟨.program ⟨214⟩, ⟨17338⟩⟩
def rawTerms : List Term := Proof.Events236.exact60553RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60553
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60553.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60550) (rightBinding := 60551)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17337⟩) (rightExpression := ⟨15632⟩)
    (transferEvent := 60552)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60549.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60445.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60553

namespace SemanticResult60557
def owner : Owner := ⟨.program ⟨214⟩, ⟨17339⟩⟩
def rawTerms : List Term := Proof.Events236.exact60557RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60557
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60557.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60554) (rightBinding := 60555)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17338⟩) (rightExpression := ⟨15751⟩)
    (transferEvent := 60556)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60553.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60422.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60557

namespace SemanticResult60561
def owner : Owner := ⟨.program ⟨214⟩, ⟨17340⟩⟩
def rawTerms : List Term := Proof.Events236.exact60561RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60561
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60561.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60558) (rightBinding := 60559)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17339⟩) (rightExpression := ⟨15870⟩)
    (transferEvent := 60560)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60557.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60399.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60561

namespace SemanticResult60565
def owner : Owner := ⟨.program ⟨214⟩, ⟨17341⟩⟩
def rawTerms : List Term := Proof.Events236.exact60565RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60565
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60565.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60562) (rightBinding := 60563)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17340⟩) (rightExpression := ⟨15989⟩)
    (transferEvent := 60564)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60561.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60376.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60565

namespace SemanticResult60569
def owner : Owner := ⟨.program ⟨214⟩, ⟨17342⟩⟩
def rawTerms : List Term := Proof.Events236.exact60569RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60569
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60569.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60566) (rightBinding := 60567)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17341⟩) (rightExpression := ⟨16108⟩)
    (transferEvent := 60568)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60565.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60353.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60569

namespace SemanticResult60573
def owner : Owner := ⟨.program ⟨214⟩, ⟨18354⟩⟩
def rawTerms : List Term := Proof.Events236.exact60573RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60573
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60573.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60570) (rightBinding := 60571)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17342⟩) (rightExpression := ⟨18353⟩)
    (transferEvent := 60572)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60569.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60330.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60573

namespace SemanticResult60577
def owner : Owner := ⟨.program ⟨214⟩, ⟨18355⟩⟩
def rawTerms : List Term := Proof.Events236.exact60577RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60577
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60577.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60574) (rightBinding := 60575)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18354⟩) (rightExpression := ⟨16311⟩)
    (transferEvent := 60576)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60573.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60307.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60577

namespace SemanticResult60581
def owner : Owner := ⟨.program ⟨214⟩, ⟨18356⟩⟩
def rawTerms : List Term := Proof.Events236.exact60581RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60581
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60581.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60578) (rightBinding := 60579)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18355⟩) (rightExpression := ⟨17123⟩)
    (transferEvent := 60580)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60577.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60284.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60581

namespace SemanticResult60585
def owner : Owner := ⟨.program ⟨214⟩, ⟨18357⟩⟩
def rawTerms : List Term := Proof.Events236.exact60585RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60585
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60585.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60582) (rightBinding := 60583)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18356⟩) (rightExpression := ⟨17907⟩)
    (transferEvent := 60584)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60581.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60261.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60585

namespace SemanticResult60589
def owner : Owner := ⟨.program ⟨214⟩, ⟨18358⟩⟩
def rawTerms : List Term := Proof.Events236.exact60589RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60589
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60589.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60586) (rightBinding := 60587)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18357⟩) (rightExpression := ⟨18208⟩)
    (transferEvent := 60588)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60585.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60238.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60589

namespace SemanticResult60593
def owner : Owner := ⟨.program ⟨214⟩, ⟨18359⟩⟩
def rawTerms : List Term := Proof.Events236.exact60593RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60593
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60593.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60590) (rightBinding := 60591)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18358⟩) (rightExpression := ⟨16682⟩)
    (transferEvent := 60592)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60589.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60215.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60593

namespace SemanticResult60597
def owner : Owner := ⟨.program ⟨214⟩, ⟨18360⟩⟩
def rawTerms : List Term := Proof.Events236.exact60597RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60597
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60597.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60594) (rightBinding := 60595)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18359⟩) (rightExpression := ⟨16801⟩)
    (transferEvent := 60596)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60593.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60192.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60597

namespace SemanticResult60601
def owner : Owner := ⟨.program ⟨214⟩, ⟨18361⟩⟩
def rawTerms : List Term := Proof.Events236.exact60601RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60601
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60601.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60598) (rightBinding := 60599)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18360⟩) (rightExpression := ⟨17088⟩)
    (transferEvent := 60600)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60597.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60169.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60601

namespace SemanticResult60605
def owner : Owner := ⟨.program ⟨214⟩, ⟨18362⟩⟩
def rawTerms : List Term := Proof.Events236.exact60605RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60605
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60605.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60602) (rightBinding := 60603)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18361⟩) (rightExpression := ⟨18173⟩)
    (transferEvent := 60604)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60601.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60146.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60605

namespace SemanticResult60616
def owner : Owner := ⟨.program ⟨214⟩, ⟨18620⟩⟩
def rawTerms : List Term := Proof.Events236.exact60616RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60616
def producerEvent : Nat := 60615
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult60616.actual selector witness
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
end SemanticResult60616

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
