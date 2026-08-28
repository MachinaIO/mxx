import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard534
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard533

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult75024
def owner : Owner := ⟨.program ⟨214⟩, ⟨15864⟩⟩
def rawTerms : List Term := Proof.Events293.exact75024RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75024
def producerEvent : Nat := 75023
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75024.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 74728, .finite 60, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult75024

namespace SemanticResult75047
def owner : Owner := ⟨.program ⟨214⟩, ⟨15745⟩⟩
def rawTerms : List Term := Proof.Events293.exact75047RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75047
def producerEvent : Nat := 75046
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75047.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 74728, .finite 59, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult75047

namespace SemanticResult75070
def owner : Owner := ⟨.program ⟨214⟩, ⟨15626⟩⟩
def rawTerms : List Term := Proof.Events293.exact75070RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75070
def producerEvent : Nat := 75069
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75070.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 74728, .finite 58, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult75070

namespace SemanticResult75093
def owner : Owner := ⟨.program ⟨214⟩, ⟨17318⟩⟩
def rawTerms : List Term := Proof.Events293.exact75093RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75093
def producerEvent : Nat := 75092
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75093.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 74728, .finite 55, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult75093

namespace SemanticResult75116
def owner : Owner := ⟨.program ⟨214⟩, ⟨15362⟩⟩
def rawTerms : List Term := Proof.Events293.exact75116RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75116
def producerEvent : Nat := 75115
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75116.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 74728, .finite 51, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult75116

namespace SemanticResult75139
def owner : Owner := ⟨.program ⟨214⟩, ⟨15306⟩⟩
def rawTerms : List Term := Proof.Events293.exact75139RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75139
def producerEvent : Nat := 75138
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75139.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 74728, .finite 48, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult75139

namespace SemanticResult75162
def owner : Owner := ⟨.program ⟨214⟩, ⟨15262⟩⟩
def rawTerms : List Term := Proof.Events293.exact75162RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75162
def producerEvent : Nat := 75161
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75162.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 74728, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult75162

namespace SemanticResult75166
def owner : Owner := ⟨.program ⟨214⟩, ⟨15307⟩⟩
def rawTerms : List Term := Proof.Events293.exact75166RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75166
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75166.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75163) (rightBinding := 75164)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15262⟩) (rightExpression := ⟨15306⟩)
    (transferEvent := 75165)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75162.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75139.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75166

namespace SemanticResult75170
def owner : Owner := ⟨.program ⟨214⟩, ⟨15363⟩⟩
def rawTerms : List Term := Proof.Events293.exact75170RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75170
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75170.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75167) (rightBinding := 75168)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15307⟩) (rightExpression := ⟨15362⟩)
    (transferEvent := 75169)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75166.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75116.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75170

namespace SemanticResult75174
def owner : Owner := ⟨.program ⟨214⟩, ⟨17319⟩⟩
def rawTerms : List Term := Proof.Events293.exact75174RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75174
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75174.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75171) (rightBinding := 75172)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15363⟩) (rightExpression := ⟨17318⟩)
    (transferEvent := 75173)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75170.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75093.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75174

namespace SemanticResult75178
def owner : Owner := ⟨.program ⟨214⟩, ⟨17320⟩⟩
def rawTerms : List Term := Proof.Events293.exact75178RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75178
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75178.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75175) (rightBinding := 75176)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17319⟩) (rightExpression := ⟨15626⟩)
    (transferEvent := 75177)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75174.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75070.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75178

namespace SemanticResult75182
def owner : Owner := ⟨.program ⟨214⟩, ⟨17321⟩⟩
def rawTerms : List Term := Proof.Events293.exact75182RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75182
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75182.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75179) (rightBinding := 75180)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17320⟩) (rightExpression := ⟨15745⟩)
    (transferEvent := 75181)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75178.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75047.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75182

namespace SemanticResult75186
def owner : Owner := ⟨.program ⟨214⟩, ⟨17322⟩⟩
def rawTerms : List Term := Proof.Events293.exact75186RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75186
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75186.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75183) (rightBinding := 75184)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17321⟩) (rightExpression := ⟨15864⟩)
    (transferEvent := 75185)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75182.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75024.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75186

namespace SemanticResult75190
def owner : Owner := ⟨.program ⟨214⟩, ⟨17323⟩⟩
def rawTerms : List Term := Proof.Events293.exact75190RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75190
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75190.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75187) (rightBinding := 75188)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17322⟩) (rightExpression := ⟨15983⟩)
    (transferEvent := 75189)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75186.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75001.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75190

namespace SemanticResult75194
def owner : Owner := ⟨.program ⟨214⟩, ⟨17324⟩⟩
def rawTerms : List Term := Proof.Events293.exact75194RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75194
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75194.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75191) (rightBinding := 75192)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17323⟩) (rightExpression := ⟨16102⟩)
    (transferEvent := 75193)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75190.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult74978.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75194

namespace SemanticResult75198
def owner : Owner := ⟨.program ⟨214⟩, ⟨18328⟩⟩
def rawTerms : List Term := Proof.Events293.exact75198RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75198
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75198.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75195) (rightBinding := 75196)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17324⟩) (rightExpression := ⟨18327⟩)
    (transferEvent := 75197)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75194.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult74955.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75198

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
