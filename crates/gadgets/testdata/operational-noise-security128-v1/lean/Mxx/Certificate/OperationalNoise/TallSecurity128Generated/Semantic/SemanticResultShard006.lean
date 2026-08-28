import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard004
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard005

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult716
def owner : Owner := ⟨.program ⟨257⟩, ⟨15890⟩⟩
def rawTerms : List Term := Proof.Events002.exact716RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 716
def producerEvent : Nat := 715
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult716.actual selector witness
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
end SemanticResult716

namespace SemanticResult721
def owner : Owner := ⟨.program ⟨257⟩, ⟨15891⟩⟩
def rawTerms : List Term := Proof.Events002.exact721RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 721
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult721.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge720.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge720.frameStart)
    (transferEvent := 719) (owner := owner)
    (leftResult := 716) (rightResult := 713)
    (working := LeftOperatorMerge720.working)
    (reconstruction := LeftOperatorMerge720.reconstruction)
    (leftReference := .predecessor 0 717 .coefficient) (rightReference := .predecessor 1 718 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult716.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult713.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge720.operationAgreement
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
end SemanticResult721

namespace SemanticResult723
def owner : Owner := ⟨.program ⟨257⟩, ⟨6727⟩⟩
def rawTerms : List Term := Proof.Events002.exact723RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 723
def producerEvent : Nat := 722
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult723.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 0, .finite 1, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult723

namespace SemanticResult728
def owner : Owner := ⟨.program ⟨257⟩, ⟨6728⟩⟩
def rawTerms : List Term := Proof.Events002.exact728RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 728
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult728.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge727.working .exactZero) := by
  apply operatorSubMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge727.frameStart)
    (transferEvent := 726) (owner := owner)
    (leftResult := 723) (rightResult := 723)
    (working := LeftOperatorMerge727.working)
    (reconstruction := LeftOperatorMerge727.reconstruction)
    (leftReference := .predecessor 0 724 .coefficient) (rightReference := .predecessor 1 725 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult723.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult723.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge727.operationAgreement
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
end SemanticResult728

namespace SemanticResult732
def owner : Owner := ⟨.program ⟨257⟩, ⟨15892⟩⟩
def rawTerms : List Term := Proof.Events002.exact732RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 732
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult732.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 729) (rightBinding := 730)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6728⟩) (rightExpression := ⟨15891⟩)
    (transferEvent := 731)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult721.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult732

namespace SemanticResult736
def owner : Owner := ⟨.program ⟨257⟩, ⟨18697⟩⟩
def rawTerms : List Term := Proof.Events002.exact736RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 736
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult736.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 733) (rightBinding := 734)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15892⟩) (rightExpression := ⟨18696⟩)
    (transferEvent := 735)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult732.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult711.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult736

namespace SemanticResult740
def owner : Owner := ⟨.program ⟨257⟩, ⟨21917⟩⟩
def rawTerms : List Term := Proof.Events002.exact740RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 740
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult740.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 737) (rightBinding := 738)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18697⟩) (rightExpression := ⟨21916⟩)
    (transferEvent := 739)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult736.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult701.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult740

namespace SemanticResult744
def owner : Owner := ⟨.program ⟨257⟩, ⟨31937⟩⟩
def rawTerms : List Term := Proof.Events002.exact744RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 744
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult744.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 741) (rightBinding := 742)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21917⟩) (rightExpression := ⟨31936⟩)
    (transferEvent := 743)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult740.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult691.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult744

namespace SemanticResult748
def owner : Owner := ⟨.program ⟨257⟩, ⟨51001⟩⟩
def rawTerms : List Term := Proof.Events002.exact748RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 748
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult748.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 745) (rightBinding := 746)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨31937⟩) (rightExpression := ⟨51000⟩)
    (transferEvent := 747)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult744.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult681.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult748

namespace SemanticResult752
def owner : Owner := ⟨.program ⟨257⟩, ⟨53981⟩⟩
def rawTerms : List Term := Proof.Events002.exact752RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 752
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult752.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 749) (rightBinding := 750)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51001⟩) (rightExpression := ⟨53980⟩)
    (transferEvent := 751)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult748.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult671.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult752

namespace SemanticResult756
def owner : Owner := ⟨.program ⟨257⟩, ⟨56961⟩⟩
def rawTerms : List Term := Proof.Events002.exact756RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 756
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult756.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 753) (rightBinding := 754)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53981⟩) (rightExpression := ⟨56960⟩)
    (transferEvent := 755)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult752.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult661.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult756

namespace SemanticResult760
def owner : Owner := ⟨.program ⟨257⟩, ⟨59941⟩⟩
def rawTerms : List Term := Proof.Events002.exact760RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 760
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult760.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 757) (rightBinding := 758)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56961⟩) (rightExpression := ⟨59940⟩)
    (transferEvent := 759)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult756.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult651.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult760

namespace SemanticResult764
def owner : Owner := ⟨.program ⟨257⟩, ⟨62921⟩⟩
def rawTerms : List Term := Proof.Events002.exact764RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 764
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult764.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 761) (rightBinding := 762)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59941⟩) (rightExpression := ⟨62920⟩)
    (transferEvent := 763)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult760.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult641.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult764

namespace SemanticResult768
def owner : Owner := ⟨.program ⟨257⟩, ⟨65982⟩⟩
def rawTerms : List Term := Proof.Events003.exact768RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 768
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult768.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 765) (rightBinding := 766)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62921⟩) (rightExpression := ⟨65981⟩)
    (transferEvent := 767)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult764.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult631.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult768

namespace SemanticResult772
def owner : Owner := ⟨.program ⟨257⟩, ⟨65983⟩⟩
def rawTerms : List Term := Proof.Events003.exact772RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 772
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult772.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 769) (rightBinding := 770)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65982⟩) (rightExpression := ⟨26509⟩)
    (transferEvent := 771)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult768.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult621.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult772

namespace SemanticResult776
def owner : Owner := ⟨.program ⟨257⟩, ⟨65984⟩⟩
def rawTerms : List Term := Proof.Events003.exact776RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 776
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult776.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 773) (rightBinding := 774)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65983⟩) (rightExpression := ⟨29189⟩)
    (transferEvent := 775)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult772.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult611.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult776

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
