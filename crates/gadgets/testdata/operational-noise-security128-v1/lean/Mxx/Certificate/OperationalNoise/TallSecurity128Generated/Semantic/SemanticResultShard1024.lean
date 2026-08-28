import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1024
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1022
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1023

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult144298
def owner : Owner := ⟨.program ⟨257⟩, ⟨59969⟩⟩
def rawTerms : List Term := Proof.Events563.exact144298RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144298
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144298.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 144295) (rightBinding := 144296)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56989⟩) (rightExpression := ⟨59968⟩)
    (transferEvent := 144297)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult144294.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult144109.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult144298

namespace SemanticResult144302
def owner : Owner := ⟨.program ⟨257⟩, ⟨62949⟩⟩
def rawTerms : List Term := Proof.Events563.exact144302RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144302
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144302.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 144299) (rightBinding := 144300)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59969⟩) (rightExpression := ⟨62948⟩)
    (transferEvent := 144301)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult144298.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult144086.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult144302

namespace SemanticResult144306
def owner : Owner := ⟨.program ⟨257⟩, ⟨66112⟩⟩
def rawTerms : List Term := Proof.Events563.exact144306RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144306
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144306.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 144303) (rightBinding := 144304)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62949⟩) (rightExpression := ⟨66111⟩)
    (transferEvent := 144305)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult144302.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult144063.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult144306

namespace SemanticResult144310
def owner : Owner := ⟨.program ⟨257⟩, ⟨66113⟩⟩
def rawTerms : List Term := Proof.Events563.exact144310RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144310
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144310.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 144307) (rightBinding := 144308)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66112⟩) (rightExpression := ⟨26528⟩)
    (transferEvent := 144309)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult144306.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult144040.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult144310

namespace SemanticResult144314
def owner : Owner := ⟨.program ⟨257⟩, ⟨66114⟩⟩
def rawTerms : List Term := Proof.Events563.exact144314RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144314
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144314.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 144311) (rightBinding := 144312)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66113⟩) (rightExpression := ⟨29208⟩)
    (transferEvent := 144313)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult144310.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult144017.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult144314

namespace SemanticResult144318
def owner : Owner := ⟨.program ⟨257⟩, ⟨66115⟩⟩
def rawTerms : List Term := Proof.Events563.exact144318RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144318
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144318.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 144315) (rightBinding := 144316)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66114⟩) (rightExpression := ⟨34872⟩)
    (transferEvent := 144317)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult144314.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult143994.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult144318

namespace SemanticResult144322
def owner : Owner := ⟨.program ⟨257⟩, ⟨66116⟩⟩
def rawTerms : List Term := Proof.Events563.exact144322RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144322
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144322.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 144319) (rightBinding := 144320)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66115⟩) (rightExpression := ⟨37552⟩)
    (transferEvent := 144321)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult144318.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult143971.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult144322

namespace SemanticResult144326
def owner : Owner := ⟨.program ⟨257⟩, ⟨66117⟩⟩
def rawTerms : List Term := Proof.Events563.exact144326RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144326
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144326.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 144323) (rightBinding := 144324)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66116⟩) (rightExpression := ⟨40228⟩)
    (transferEvent := 144325)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult144322.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult143948.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult144326

namespace SemanticResult144330
def owner : Owner := ⟨.program ⟨257⟩, ⟨66118⟩⟩
def rawTerms : List Term := Proof.Events563.exact144330RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144330
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144330.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 144327) (rightBinding := 144328)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66117⟩) (rightExpression := ⟨42908⟩)
    (transferEvent := 144329)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult144326.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult143925.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult144330

namespace SemanticResult144334
def owner : Owner := ⟨.program ⟨257⟩, ⟨66119⟩⟩
def rawTerms : List Term := Proof.Events563.exact144334RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144334
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144334.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 144331) (rightBinding := 144332)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66118⟩) (rightExpression := ⟨45592⟩)
    (transferEvent := 144333)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult144330.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult143902.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult144334

namespace SemanticResult144338
def owner : Owner := ⟨.program ⟨257⟩, ⟨66120⟩⟩
def rawTerms : List Term := Proof.Events563.exact144338RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144338
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144338.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 144335) (rightBinding := 144336)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66119⟩) (rightExpression := ⟨48272⟩)
    (transferEvent := 144337)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult144334.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult143879.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult144338

namespace SemanticResult144349
def owner : Owner := ⟨.program ⟨257⟩, ⟨68788⟩⟩
def rawTerms : List Term := Proof.Events563.exact144349RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144349
def producerEvent : Nat := 144348
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144349.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 143836, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult144349

namespace SemanticResult144352
def owner : Owner := ⟨.program ⟨257⟩, ⟨71017⟩⟩
def rawTerms : List Term := Proof.Events563.exact144352RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144352
def producerEvent : Nat := 144351
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144352.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 143836, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult144352

namespace SemanticResult144361
def owner : Owner := ⟨.program ⟨257⟩, ⟨69060⟩⟩
def rawTerms : List Term := Proof.Events563.exact144361RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144361
def producerEvent : Nat := 144360
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144361.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 144359 .coefficient), 143836, .finite 1059, .identity (.predecessor 0 144359 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult144361

namespace SemanticResult144363
def owner : Owner := ⟨.program ⟨257⟩, ⟨6908⟩⟩
def rawTerms : List Term := Proof.Events563.exact144363RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144363
def producerEvent : Nat := 144362
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144363.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 143836, .large, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult144363

namespace SemanticResult144385
def owner : Owner := ⟨.program ⟨257⟩, ⟨69061⟩⟩
def rawTerms : List Term := Proof.Events564.exact144385RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 144385
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult144385.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge144367.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge144367.frameStart)
    (transferEvent := 144366) (owner := owner)
    (leftResult := 144363) (rightResult := 144361)
    (working := LeftOperatorMerge144367.working)
    (reconstruction := LeftOperatorMerge144367.reconstruction)
    (leftReference := .predecessor 0 144364 .coefficient) (rightReference := .predecessor 1 144365 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult144363.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult144361.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge144367.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult144385

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
