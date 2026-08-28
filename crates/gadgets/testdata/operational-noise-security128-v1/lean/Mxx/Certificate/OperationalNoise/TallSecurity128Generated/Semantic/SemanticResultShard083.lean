import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard083
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard005
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard082

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult10428
def owner : Owner := ⟨.program ⟨257⟩, ⟨32102⟩⟩
def rawTerms : List Term := Proof.Events040.exact10428RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10428
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10428.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge10427.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge10427.frameStart)
    (transferEvent := 10426) (owner := owner)
    (leftResult := 10423) (rightResult := 683)
    (working := LeftOperatorMerge10427.working)
    (reconstruction := LeftOperatorMerge10427.reconstruction)
    (leftReference := .predecessor 0 10424 .coefficient) (rightReference := .predecessor 1 10425 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult10423.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult683.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge10427.operationAgreement
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
end SemanticResult10428

namespace SemanticResult10431
def owner : Owner := ⟨.program ⟨257⟩, ⟨22081⟩⟩
def rawTerms : List Term := Proof.Events040.exact10431RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10431
def producerEvent : Nat := 10430
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10431.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 0, .finite 4, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult10431

namespace SemanticResult10436
def owner : Owner := ⟨.program ⟨257⟩, ⟨22082⟩⟩
def rawTerms : List Term := Proof.Events040.exact10436RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10436
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10436.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge10435.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge10435.frameStart)
    (transferEvent := 10434) (owner := owner)
    (leftResult := 10431) (rightResult := 693)
    (working := LeftOperatorMerge10435.working)
    (reconstruction := LeftOperatorMerge10435.reconstruction)
    (leftReference := .predecessor 0 10432 .coefficient) (rightReference := .predecessor 1 10433 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult10431.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult693.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge10435.operationAgreement
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
end SemanticResult10436

namespace SemanticResult10439
def owner : Owner := ⟨.program ⟨257⟩, ⟨18861⟩⟩
def rawTerms : List Term := Proof.Events040.exact10439RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10439
def producerEvent : Nat := 10438
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10439.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 0, .finite 3, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult10439

namespace SemanticResult10444
def owner : Owner := ⟨.program ⟨257⟩, ⟨18862⟩⟩
def rawTerms : List Term := Proof.Events040.exact10444RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10444
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10444.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge10443.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge10443.frameStart)
    (transferEvent := 10442) (owner := owner)
    (leftResult := 10439) (rightResult := 703)
    (working := LeftOperatorMerge10443.working)
    (reconstruction := LeftOperatorMerge10443.reconstruction)
    (leftReference := .predecessor 0 10440 .coefficient) (rightReference := .predecessor 1 10441 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult10439.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult703.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge10443.operationAgreement
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
end SemanticResult10444

namespace SemanticResult10447
def owner : Owner := ⟨.program ⟨257⟩, ⟨16030⟩⟩
def rawTerms : List Term := Proof.Events040.exact10447RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10447
def producerEvent : Nat := 10446
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10447.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 0, .finite 2, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult10447

namespace SemanticResult10452
def owner : Owner := ⟨.program ⟨257⟩, ⟨16031⟩⟩
def rawTerms : List Term := Proof.Events040.exact10452RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10452
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10452.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge10451.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge10451.frameStart)
    (transferEvent := 10450) (owner := owner)
    (leftResult := 10447) (rightResult := 713)
    (working := LeftOperatorMerge10451.working)
    (reconstruction := LeftOperatorMerge10451.reconstruction)
    (leftReference := .predecessor 0 10448 .coefficient) (rightReference := .predecessor 1 10449 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult10447.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult713.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge10451.operationAgreement
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
end SemanticResult10452

namespace SemanticResult10456
def owner : Owner := ⟨.program ⟨257⟩, ⟨16032⟩⟩
def rawTerms : List Term := Proof.Events040.exact10456RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10456
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10456.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10453) (rightBinding := 10454)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6728⟩) (rightExpression := ⟨16031⟩)
    (transferEvent := 10455)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10452.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10456

namespace SemanticResult10460
def owner : Owner := ⟨.program ⟨257⟩, ⟨18863⟩⟩
def rawTerms : List Term := Proof.Events040.exact10460RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10460
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10460.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10457) (rightBinding := 10458)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16032⟩) (rightExpression := ⟨18862⟩)
    (transferEvent := 10459)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10456.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10444.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10460

namespace SemanticResult10464
def owner : Owner := ⟨.program ⟨257⟩, ⟨22083⟩⟩
def rawTerms : List Term := Proof.Events040.exact10464RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10464
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10464.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10461) (rightBinding := 10462)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18863⟩) (rightExpression := ⟨22082⟩)
    (transferEvent := 10463)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10460.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10436.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10464

namespace SemanticResult10468
def owner : Owner := ⟨.program ⟨257⟩, ⟨32103⟩⟩
def rawTerms : List Term := Proof.Events040.exact10468RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10468
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10468.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10465) (rightBinding := 10466)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22083⟩) (rightExpression := ⟨32102⟩)
    (transferEvent := 10467)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10464.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10428.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10468

namespace SemanticResult10472
def owner : Owner := ⟨.program ⟨257⟩, ⟨51167⟩⟩
def rawTerms : List Term := Proof.Events040.exact10472RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10472
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10472.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10469) (rightBinding := 10470)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32103⟩) (rightExpression := ⟨51166⟩)
    (transferEvent := 10471)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10468.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10420.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10472

namespace SemanticResult10476
def owner : Owner := ⟨.program ⟨257⟩, ⟨54147⟩⟩
def rawTerms : List Term := Proof.Events040.exact10476RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10476
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10476.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10473) (rightBinding := 10474)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51167⟩) (rightExpression := ⟨54146⟩)
    (transferEvent := 10475)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10472.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10412.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10476

namespace SemanticResult10480
def owner : Owner := ⟨.program ⟨257⟩, ⟨57127⟩⟩
def rawTerms : List Term := Proof.Events040.exact10480RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10480
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10480.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10477) (rightBinding := 10478)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54147⟩) (rightExpression := ⟨57126⟩)
    (transferEvent := 10479)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10476.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10404.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10480

namespace SemanticResult10484
def owner : Owner := ⟨.program ⟨257⟩, ⟨60107⟩⟩
def rawTerms : List Term := Proof.Events040.exact10484RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10484
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10484.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10481) (rightBinding := 10482)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57127⟩) (rightExpression := ⟨60106⟩)
    (transferEvent := 10483)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10480.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10396.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10484

namespace SemanticResult10488
def owner : Owner := ⟨.program ⟨257⟩, ⟨63087⟩⟩
def rawTerms : List Term := Proof.Events040.exact10488RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10488
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10488.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10485) (rightBinding := 10486)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60107⟩) (rightExpression := ⟨63086⟩)
    (transferEvent := 10487)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10484.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10388.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10488

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
