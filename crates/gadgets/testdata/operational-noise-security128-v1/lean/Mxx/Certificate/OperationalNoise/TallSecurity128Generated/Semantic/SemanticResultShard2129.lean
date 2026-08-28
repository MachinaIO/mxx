import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard2129
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard2128

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult304013
def owner : Owner := ⟨.program ⟨257⟩, ⟨31916⟩⟩
def rawTerms : List Term := Proof.Events1187.exact304013RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304013
def producerEvent : Nat := 304012
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304013.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 303660, .finite 55, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult304013

namespace SemanticResult304036
def owner : Owner := ⟨.program ⟨257⟩, ⟨21896⟩⟩
def rawTerms : List Term := Proof.Events1187.exact304036RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304036
def producerEvent : Nat := 304035
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304036.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 303660, .finite 51, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult304036

namespace SemanticResult304059
def owner : Owner := ⟨.program ⟨257⟩, ⟨18676⟩⟩
def rawTerms : List Term := Proof.Events1187.exact304059RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304059
def producerEvent : Nat := 304058
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304059.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 303660, .finite 48, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult304059

namespace SemanticResult304082
def owner : Owner := ⟨.program ⟨257⟩, ⟨15875⟩⟩
def rawTerms : List Term := Proof.Events1187.exact304082RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304082
def producerEvent : Nat := 304081
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304082.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 303660, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult304082

namespace SemanticResult304086
def owner : Owner := ⟨.program ⟨257⟩, ⟨18677⟩⟩
def rawTerms : List Term := Proof.Events1187.exact304086RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304086
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304086.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 304083) (rightBinding := 304084)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15875⟩) (rightExpression := ⟨18676⟩)
    (transferEvent := 304085)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult304082.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult304059.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult304086

namespace SemanticResult304090
def owner : Owner := ⟨.program ⟨257⟩, ⟨21897⟩⟩
def rawTerms : List Term := Proof.Events1187.exact304090RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304090
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304090.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 304087) (rightBinding := 304088)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18677⟩) (rightExpression := ⟨21896⟩)
    (transferEvent := 304089)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult304086.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult304036.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult304090

namespace SemanticResult304094
def owner : Owner := ⟨.program ⟨257⟩, ⟨31917⟩⟩
def rawTerms : List Term := Proof.Events1187.exact304094RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304094
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304094.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 304091) (rightBinding := 304092)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21897⟩) (rightExpression := ⟨31916⟩)
    (transferEvent := 304093)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult304090.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult304013.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult304094

namespace SemanticResult304098
def owner : Owner := ⟨.program ⟨257⟩, ⟨50972⟩⟩
def rawTerms : List Term := Proof.Events1187.exact304098RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304098
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304098.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 304095) (rightBinding := 304096)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨31917⟩) (rightExpression := ⟨50971⟩)
    (transferEvent := 304097)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult304094.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult303990.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult304098

namespace SemanticResult304102
def owner : Owner := ⟨.program ⟨257⟩, ⟨53952⟩⟩
def rawTerms : List Term := Proof.Events1187.exact304102RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304102
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304102.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 304099) (rightBinding := 304100)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨50972⟩) (rightExpression := ⟨53951⟩)
    (transferEvent := 304101)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult304098.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult303967.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult304102

namespace SemanticResult304106
def owner : Owner := ⟨.program ⟨257⟩, ⟨56932⟩⟩
def rawTerms : List Term := Proof.Events1187.exact304106RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304106
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304106.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 304103) (rightBinding := 304104)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53952⟩) (rightExpression := ⟨56931⟩)
    (transferEvent := 304105)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult304102.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult303944.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult304106

namespace SemanticResult304110
def owner : Owner := ⟨.program ⟨257⟩, ⟨59912⟩⟩
def rawTerms : List Term := Proof.Events1187.exact304110RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304110
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304110.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 304107) (rightBinding := 304108)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56932⟩) (rightExpression := ⟨59911⟩)
    (transferEvent := 304109)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult304106.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult303921.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult304110

namespace SemanticResult304114
def owner : Owner := ⟨.program ⟨257⟩, ⟨62892⟩⟩
def rawTerms : List Term := Proof.Events1187.exact304114RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304114
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304114.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 304111) (rightBinding := 304112)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59912⟩) (rightExpression := ⟨62891⟩)
    (transferEvent := 304113)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult304110.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult303898.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult304114

namespace SemanticResult304118
def owner : Owner := ⟨.program ⟨257⟩, ⟨65902⟩⟩
def rawTerms : List Term := Proof.Events1187.exact304118RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304118
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304118.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 304115) (rightBinding := 304116)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62892⟩) (rightExpression := ⟨65901⟩)
    (transferEvent := 304117)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult304114.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult303875.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult304118

namespace SemanticResult304122
def owner : Owner := ⟨.program ⟨257⟩, ⟨65903⟩⟩
def rawTerms : List Term := Proof.Events1187.exact304122RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304122
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304122.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 304119) (rightBinding := 304120)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65902⟩) (rightExpression := ⟨26489⟩)
    (transferEvent := 304121)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult304118.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult303852.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult304122

namespace SemanticResult304126
def owner : Owner := ⟨.program ⟨257⟩, ⟨65904⟩⟩
def rawTerms : List Term := Proof.Events1187.exact304126RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304126
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304126.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 304123) (rightBinding := 304124)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65903⟩) (rightExpression := ⟨29169⟩)
    (transferEvent := 304125)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult304122.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult303829.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult304126

namespace SemanticResult304130
def owner : Owner := ⟨.program ⟨257⟩, ⟨65905⟩⟩
def rawTerms : List Term := Proof.Events1188.exact304130RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 304130
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult304130.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 304127) (rightBinding := 304128)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65904⟩) (rightExpression := ⟨34833⟩)
    (transferEvent := 304129)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult304126.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult303806.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult304130

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
