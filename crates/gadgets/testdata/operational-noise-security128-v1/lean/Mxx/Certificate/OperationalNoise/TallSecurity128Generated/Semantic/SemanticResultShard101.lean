import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard101
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard005
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard099
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard100

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult12683
def owner : Owner := ⟨.program ⟨257⟩, ⟨18766⟩⟩
def rawTerms : List Term := Proof.Events049.exact12683RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12683
def producerEvent : Nat := 12682
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12683.actual selector witness
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
end SemanticResult12683

namespace SemanticResult12688
def owner : Owner := ⟨.program ⟨257⟩, ⟨18767⟩⟩
def rawTerms : List Term := Proof.Events049.exact12688RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12688
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12688.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge12687.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge12687.frameStart)
    (transferEvent := 12686) (owner := owner)
    (leftResult := 12683) (rightResult := 703)
    (working := LeftOperatorMerge12687.working)
    (reconstruction := LeftOperatorMerge12687.reconstruction)
    (leftReference := .predecessor 0 12684 .coefficient) (rightReference := .predecessor 1 12685 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult12683.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult703.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge12687.operationAgreement
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
end SemanticResult12688

namespace SemanticResult12691
def owner : Owner := ⟨.program ⟨257⟩, ⟨15950⟩⟩
def rawTerms : List Term := Proof.Events049.exact12691RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12691
def producerEvent : Nat := 12690
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12691.actual selector witness
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
end SemanticResult12691

namespace SemanticResult12696
def owner : Owner := ⟨.program ⟨257⟩, ⟨15951⟩⟩
def rawTerms : List Term := Proof.Events049.exact12696RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12696
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12696.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge12695.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge12695.frameStart)
    (transferEvent := 12694) (owner := owner)
    (leftResult := 12691) (rightResult := 713)
    (working := LeftOperatorMerge12695.working)
    (reconstruction := LeftOperatorMerge12695.reconstruction)
    (leftReference := .predecessor 0 12692 .coefficient) (rightReference := .predecessor 1 12693 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult12691.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult713.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge12695.operationAgreement
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
end SemanticResult12696

namespace SemanticResult12700
def owner : Owner := ⟨.program ⟨257⟩, ⟨15952⟩⟩
def rawTerms : List Term := Proof.Events049.exact12700RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12700
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12700.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 12697) (rightBinding := 12698)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6728⟩) (rightExpression := ⟨15951⟩)
    (transferEvent := 12699)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12696.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult12700

namespace SemanticResult12704
def owner : Owner := ⟨.program ⟨257⟩, ⟨18768⟩⟩
def rawTerms : List Term := Proof.Events049.exact12704RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12704
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12704.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 12701) (rightBinding := 12702)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15952⟩) (rightExpression := ⟨18767⟩)
    (transferEvent := 12703)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12700.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12688.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult12704

namespace SemanticResult12708
def owner : Owner := ⟨.program ⟨257⟩, ⟨21988⟩⟩
def rawTerms : List Term := Proof.Events049.exact12708RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12708
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12708.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 12705) (rightBinding := 12706)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18768⟩) (rightExpression := ⟨21987⟩)
    (transferEvent := 12707)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12704.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12680.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult12708

namespace SemanticResult12712
def owner : Owner := ⟨.program ⟨257⟩, ⟨32008⟩⟩
def rawTerms : List Term := Proof.Events049.exact12712RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12712
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12712.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 12709) (rightBinding := 12710)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21988⟩) (rightExpression := ⟨32007⟩)
    (transferEvent := 12711)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12708.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12672.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult12712

namespace SemanticResult12716
def owner : Owner := ⟨.program ⟨257⟩, ⟨51072⟩⟩
def rawTerms : List Term := Proof.Events049.exact12716RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12716
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12716.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 12713) (rightBinding := 12714)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32008⟩) (rightExpression := ⟨51071⟩)
    (transferEvent := 12715)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12712.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12664.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult12716

namespace SemanticResult12720
def owner : Owner := ⟨.program ⟨257⟩, ⟨54052⟩⟩
def rawTerms : List Term := Proof.Events049.exact12720RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12720
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12720.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 12717) (rightBinding := 12718)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51072⟩) (rightExpression := ⟨54051⟩)
    (transferEvent := 12719)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12716.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12656.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult12720

namespace SemanticResult12724
def owner : Owner := ⟨.program ⟨257⟩, ⟨57032⟩⟩
def rawTerms : List Term := Proof.Events049.exact12724RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12724
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12724.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 12721) (rightBinding := 12722)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54052⟩) (rightExpression := ⟨57031⟩)
    (transferEvent := 12723)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12720.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12648.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult12724

namespace SemanticResult12728
def owner : Owner := ⟨.program ⟨257⟩, ⟨60012⟩⟩
def rawTerms : List Term := Proof.Events049.exact12728RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12728
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12728.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 12725) (rightBinding := 12726)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57032⟩) (rightExpression := ⟨60011⟩)
    (transferEvent := 12727)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12724.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12640.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult12728

namespace SemanticResult12732
def owner : Owner := ⟨.program ⟨257⟩, ⟨62992⟩⟩
def rawTerms : List Term := Proof.Events049.exact12732RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12732
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12732.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 12729) (rightBinding := 12730)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60012⟩) (rightExpression := ⟨62991⟩)
    (transferEvent := 12731)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12632.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult12732

namespace SemanticResult12736
def owner : Owner := ⟨.program ⟨257⟩, ⟨66240⟩⟩
def rawTerms : List Term := Proof.Events049.exact12736RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12736
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12736.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 12733) (rightBinding := 12734)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62992⟩) (rightExpression := ⟨66239⟩)
    (transferEvent := 12735)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12732.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12624.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult12736

namespace SemanticResult12740
def owner : Owner := ⟨.program ⟨257⟩, ⟨66241⟩⟩
def rawTerms : List Term := Proof.Events049.exact12740RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12740
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12740.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 12737) (rightBinding := 12738)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66240⟩) (rightExpression := ⟨26558⟩)
    (transferEvent := 12739)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12736.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12616.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult12740

namespace SemanticResult12744
def owner : Owner := ⟨.program ⟨257⟩, ⟨66242⟩⟩
def rawTerms : List Term := Proof.Events049.exact12744RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12744
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult12744.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 12741) (rightBinding := 12742)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66241⟩) (rightExpression := ⟨29238⟩)
    (transferEvent := 12743)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12740.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12608.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult12744

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
