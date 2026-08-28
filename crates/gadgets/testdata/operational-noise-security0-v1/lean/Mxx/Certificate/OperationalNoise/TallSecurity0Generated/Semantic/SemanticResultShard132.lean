import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard131

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult16636
def owner : Owner := ⟨.program ⟨214⟩, ⟨15326⟩⟩
def rawTerms : List Term := Proof.Events064.exact16636RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16636
def producerEvent : Nat := 16635
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16636.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 16225, .finite 48, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult16636

namespace SemanticResult16659
def owner : Owner := ⟨.program ⟨214⟩, ⟨15277⟩⟩
def rawTerms : List Term := Proof.Events065.exact16659RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16659
def producerEvent : Nat := 16658
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16659.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 16225, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult16659

namespace SemanticResult16663
def owner : Owner := ⟨.program ⟨214⟩, ⟨15327⟩⟩
def rawTerms : List Term := Proof.Events065.exact16663RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16663
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16663.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16660) (rightBinding := 16661)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15277⟩) (rightExpression := ⟨15326⟩)
    (transferEvent := 16662)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16659.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16636.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16663

namespace SemanticResult16667
def owner : Owner := ⟨.program ⟨214⟩, ⟨15383⟩⟩
def rawTerms : List Term := Proof.Events065.exact16667RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16667
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16667.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16664) (rightBinding := 16665)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15327⟩) (rightExpression := ⟨15382⟩)
    (transferEvent := 16666)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16663.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16613.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16667

namespace SemanticResult16671
def owner : Owner := ⟨.program ⟨214⟩, ⟨17364⟩⟩
def rawTerms : List Term := Proof.Events065.exact16671RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16671
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16671.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16668) (rightBinding := 16669)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15383⟩) (rightExpression := ⟨17363⟩)
    (transferEvent := 16670)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16667.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16590.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16671

namespace SemanticResult16675
def owner : Owner := ⟨.program ⟨214⟩, ⟨17365⟩⟩
def rawTerms : List Term := Proof.Events065.exact16675RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16675
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16675.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16672) (rightBinding := 16673)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17364⟩) (rightExpression := ⟨15641⟩)
    (transferEvent := 16674)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16671.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16567.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16675

namespace SemanticResult16679
def owner : Owner := ⟨.program ⟨214⟩, ⟨17366⟩⟩
def rawTerms : List Term := Proof.Events065.exact16679RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16679
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16679.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16676) (rightBinding := 16677)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17365⟩) (rightExpression := ⟨15760⟩)
    (transferEvent := 16678)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16675.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16544.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16679

namespace SemanticResult16683
def owner : Owner := ⟨.program ⟨214⟩, ⟨17367⟩⟩
def rawTerms : List Term := Proof.Events065.exact16683RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16683
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16683.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16680) (rightBinding := 16681)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17366⟩) (rightExpression := ⟨15879⟩)
    (transferEvent := 16682)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16679.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16521.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16683

namespace SemanticResult16687
def owner : Owner := ⟨.program ⟨214⟩, ⟨17368⟩⟩
def rawTerms : List Term := Proof.Events065.exact16687RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16687
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16687.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16684) (rightBinding := 16685)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17367⟩) (rightExpression := ⟨15998⟩)
    (transferEvent := 16686)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16683.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16498.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16687

namespace SemanticResult16691
def owner : Owner := ⟨.program ⟨214⟩, ⟨17369⟩⟩
def rawTerms : List Term := Proof.Events065.exact16691RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16691
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16691.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16688) (rightBinding := 16689)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17368⟩) (rightExpression := ⟨16117⟩)
    (transferEvent := 16690)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16687.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16475.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16691

namespace SemanticResult16695
def owner : Owner := ⟨.program ⟨214⟩, ⟨18393⟩⟩
def rawTerms : List Term := Proof.Events065.exact16695RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16695
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16695.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16692) (rightBinding := 16693)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17369⟩) (rightExpression := ⟨18392⟩)
    (transferEvent := 16694)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16691.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16452.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16695

namespace SemanticResult16699
def owner : Owner := ⟨.program ⟨214⟩, ⟨18394⟩⟩
def rawTerms : List Term := Proof.Events065.exact16699RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16699
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16699.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16696) (rightBinding := 16697)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18393⟩) (rightExpression := ⟨16320⟩)
    (transferEvent := 16698)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16695.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16429.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16699

namespace SemanticResult16703
def owner : Owner := ⟨.program ⟨214⟩, ⟨18395⟩⟩
def rawTerms : List Term := Proof.Events065.exact16703RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16703
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16703.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16700) (rightBinding := 16701)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18394⟩) (rightExpression := ⟨17132⟩)
    (transferEvent := 16702)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16699.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16406.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16703

namespace SemanticResult16707
def owner : Owner := ⟨.program ⟨214⟩, ⟨18396⟩⟩
def rawTerms : List Term := Proof.Events065.exact16707RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16707
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16707.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16704) (rightBinding := 16705)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18395⟩) (rightExpression := ⟨17916⟩)
    (transferEvent := 16706)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16703.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16383.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16707

namespace SemanticResult16711
def owner : Owner := ⟨.program ⟨214⟩, ⟨18397⟩⟩
def rawTerms : List Term := Proof.Events065.exact16711RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16711
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16711.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16708) (rightBinding := 16709)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18396⟩) (rightExpression := ⟨18217⟩)
    (transferEvent := 16710)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16707.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16360.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16711

namespace SemanticResult16715
def owner : Owner := ⟨.program ⟨214⟩, ⟨18398⟩⟩
def rawTerms : List Term := Proof.Events065.exact16715RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16715
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult16715.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16712) (rightBinding := 16713)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18397⟩) (rightExpression := ⟨16691⟩)
    (transferEvent := 16714)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16711.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16337.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16715

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
