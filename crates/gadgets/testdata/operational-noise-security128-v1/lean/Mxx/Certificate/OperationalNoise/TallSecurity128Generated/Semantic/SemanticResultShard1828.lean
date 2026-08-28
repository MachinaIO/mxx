import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1828
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1827

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult261247
def owner : Owner := ⟨.program ⟨257⟩, ⟨18771⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261247RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261247
def producerEvent : Nat := 261246
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261247.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 260836, .finite 48, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult261247

namespace SemanticResult261270
def owner : Owner := ⟨.program ⟨257⟩, ⟨15955⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261270RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261270
def producerEvent : Nat := 261269
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261270.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 260836, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult261270

namespace SemanticResult261274
def owner : Owner := ⟨.program ⟨257⟩, ⟨18772⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261274RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261274
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261274.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 261271) (rightBinding := 261272)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15955⟩) (rightExpression := ⟨18771⟩)
    (transferEvent := 261273)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult261270.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult261247.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult261274

namespace SemanticResult261278
def owner : Owner := ⟨.program ⟨257⟩, ⟨21992⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261278RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261278
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261278.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 261275) (rightBinding := 261276)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18772⟩) (rightExpression := ⟨21991⟩)
    (transferEvent := 261277)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult261274.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult261224.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult261278

namespace SemanticResult261282
def owner : Owner := ⟨.program ⟨257⟩, ⟨32012⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261282RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261282
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261282.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 261279) (rightBinding := 261280)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21992⟩) (rightExpression := ⟨32011⟩)
    (transferEvent := 261281)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult261278.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult261201.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult261282

namespace SemanticResult261286
def owner : Owner := ⟨.program ⟨257⟩, ⟨51067⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261286RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261286
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261286.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 261283) (rightBinding := 261284)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32012⟩) (rightExpression := ⟨51066⟩)
    (transferEvent := 261285)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult261282.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult261178.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult261286

namespace SemanticResult261290
def owner : Owner := ⟨.program ⟨257⟩, ⟨54047⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261290RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261290
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261290.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 261287) (rightBinding := 261288)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51067⟩) (rightExpression := ⟨54046⟩)
    (transferEvent := 261289)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult261286.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult261155.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult261290

namespace SemanticResult261294
def owner : Owner := ⟨.program ⟨257⟩, ⟨57027⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261294RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261294
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261294.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 261291) (rightBinding := 261292)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54047⟩) (rightExpression := ⟨57026⟩)
    (transferEvent := 261293)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult261290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult261132.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult261294

namespace SemanticResult261298
def owner : Owner := ⟨.program ⟨257⟩, ⟨60007⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261298RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261298
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261298.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 261295) (rightBinding := 261296)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57027⟩) (rightExpression := ⟨60006⟩)
    (transferEvent := 261297)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult261294.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult261109.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult261298

namespace SemanticResult261302
def owner : Owner := ⟨.program ⟨257⟩, ⟨62987⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261302RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261302
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261302.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 261299) (rightBinding := 261300)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60007⟩) (rightExpression := ⟨62986⟩)
    (transferEvent := 261301)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult261298.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult261086.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult261302

namespace SemanticResult261306
def owner : Owner := ⟨.program ⟨257⟩, ⟨66252⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261306RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261306
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261306.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 261303) (rightBinding := 261304)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62987⟩) (rightExpression := ⟨66251⟩)
    (transferEvent := 261305)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult261302.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult261063.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult261306

namespace SemanticResult261310
def owner : Owner := ⟨.program ⟨257⟩, ⟨66253⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261310RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261310
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261310.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 261307) (rightBinding := 261308)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66252⟩) (rightExpression := ⟨26554⟩)
    (transferEvent := 261309)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult261306.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult261040.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult261310

namespace SemanticResult261314
def owner : Owner := ⟨.program ⟨257⟩, ⟨66254⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261314RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261314
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261314.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 261311) (rightBinding := 261312)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66253⟩) (rightExpression := ⟨29234⟩)
    (transferEvent := 261313)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult261310.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult261017.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult261314

namespace SemanticResult261318
def owner : Owner := ⟨.program ⟨257⟩, ⟨66255⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261318RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261318
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261318.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 261315) (rightBinding := 261316)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66254⟩) (rightExpression := ⟨34898⟩)
    (transferEvent := 261317)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult261314.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult260994.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult261318

namespace SemanticResult261322
def owner : Owner := ⟨.program ⟨257⟩, ⟨66256⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261322RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261322
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261322.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 261319) (rightBinding := 261320)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66255⟩) (rightExpression := ⟨37578⟩)
    (transferEvent := 261321)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult261318.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult260971.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult261322

namespace SemanticResult261326
def owner : Owner := ⟨.program ⟨257⟩, ⟨66257⟩⟩
def rawTerms : List Term := Proof.Events1020.exact261326RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 261326
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult261326.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 261323) (rightBinding := 261324)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66256⟩) (rightExpression := ⟨40254⟩)
    (transferEvent := 261325)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult261322.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult260948.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult261326

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
