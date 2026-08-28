import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard333
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard332

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult45820
def owner : Owner := ⟨.program ⟨214⟩, ⟨15635⟩⟩
def rawTerms : List Term := Proof.Events178.exact45820RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45820
def producerEvent : Nat := 45819
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45820.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 45478, .finite 58, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult45820

namespace SemanticResult45843
def owner : Owner := ⟨.program ⟨214⟩, ⟨17345⟩⟩
def rawTerms : List Term := Proof.Events179.exact45843RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45843
def producerEvent : Nat := 45842
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45843.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 45478, .finite 55, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult45843

namespace SemanticResult45866
def owner : Owner := ⟨.program ⟨214⟩, ⟨15374⟩⟩
def rawTerms : List Term := Proof.Events179.exact45866RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45866
def producerEvent : Nat := 45865
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45866.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 45478, .finite 51, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult45866

namespace SemanticResult45889
def owner : Owner := ⟨.program ⟨214⟩, ⟨15318⟩⟩
def rawTerms : List Term := Proof.Events179.exact45889RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45889
def producerEvent : Nat := 45888
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45889.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 45478, .finite 48, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult45889

namespace SemanticResult45912
def owner : Owner := ⟨.program ⟨214⟩, ⟨15271⟩⟩
def rawTerms : List Term := Proof.Events179.exact45912RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45912
def producerEvent : Nat := 45911
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45912.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 45478, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult45912

namespace SemanticResult45916
def owner : Owner := ⟨.program ⟨214⟩, ⟨15319⟩⟩
def rawTerms : List Term := Proof.Events179.exact45916RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45916
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45916.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 45913) (rightBinding := 45914)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15271⟩) (rightExpression := ⟨15318⟩)
    (transferEvent := 45915)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult45912.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45889.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult45916

namespace SemanticResult45920
def owner : Owner := ⟨.program ⟨214⟩, ⟨15375⟩⟩
def rawTerms : List Term := Proof.Events179.exact45920RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45920
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45920.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 45917) (rightBinding := 45918)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15319⟩) (rightExpression := ⟨15374⟩)
    (transferEvent := 45919)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult45916.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45866.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult45920

namespace SemanticResult45924
def owner : Owner := ⟨.program ⟨214⟩, ⟨17346⟩⟩
def rawTerms : List Term := Proof.Events179.exact45924RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45924
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45924.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 45921) (rightBinding := 45922)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15375⟩) (rightExpression := ⟨17345⟩)
    (transferEvent := 45923)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult45920.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45843.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult45924

namespace SemanticResult45928
def owner : Owner := ⟨.program ⟨214⟩, ⟨17347⟩⟩
def rawTerms : List Term := Proof.Events179.exact45928RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45928
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45928.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 45925) (rightBinding := 45926)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17346⟩) (rightExpression := ⟨15635⟩)
    (transferEvent := 45927)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult45924.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45820.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult45928

namespace SemanticResult45932
def owner : Owner := ⟨.program ⟨214⟩, ⟨17348⟩⟩
def rawTerms : List Term := Proof.Events179.exact45932RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45932
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45932.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 45929) (rightBinding := 45930)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17347⟩) (rightExpression := ⟨15754⟩)
    (transferEvent := 45931)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult45928.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45797.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult45932

namespace SemanticResult45936
def owner : Owner := ⟨.program ⟨214⟩, ⟨17349⟩⟩
def rawTerms : List Term := Proof.Events179.exact45936RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45936
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45936.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 45933) (rightBinding := 45934)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17348⟩) (rightExpression := ⟨15873⟩)
    (transferEvent := 45935)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult45932.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45774.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult45936

namespace SemanticResult45940
def owner : Owner := ⟨.program ⟨214⟩, ⟨17350⟩⟩
def rawTerms : List Term := Proof.Events179.exact45940RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45940
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45940.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 45937) (rightBinding := 45938)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17349⟩) (rightExpression := ⟨15992⟩)
    (transferEvent := 45939)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult45936.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45751.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult45940

namespace SemanticResult45944
def owner : Owner := ⟨.program ⟨214⟩, ⟨17351⟩⟩
def rawTerms : List Term := Proof.Events179.exact45944RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45944
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45944.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 45941) (rightBinding := 45942)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17350⟩) (rightExpression := ⟨16111⟩)
    (transferEvent := 45943)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult45940.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45728.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult45944

namespace SemanticResult45948
def owner : Owner := ⟨.program ⟨214⟩, ⟨18367⟩⟩
def rawTerms : List Term := Proof.Events179.exact45948RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45948
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45948.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 45945) (rightBinding := 45946)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17351⟩) (rightExpression := ⟨18366⟩)
    (transferEvent := 45947)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult45944.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45705.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult45948

namespace SemanticResult45952
def owner : Owner := ⟨.program ⟨214⟩, ⟨18368⟩⟩
def rawTerms : List Term := Proof.Events179.exact45952RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45952
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45952.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 45949) (rightBinding := 45950)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18367⟩) (rightExpression := ⟨16314⟩)
    (transferEvent := 45951)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult45948.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45682.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult45952

namespace SemanticResult45956
def owner : Owner := ⟨.program ⟨214⟩, ⟨18369⟩⟩
def rawTerms : List Term := Proof.Events179.exact45956RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 45956
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult45956.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 45953) (rightBinding := 45954)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18368⟩) (rightExpression := ⟨17126⟩)
    (transferEvent := 45955)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult45952.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45659.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult45956

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
