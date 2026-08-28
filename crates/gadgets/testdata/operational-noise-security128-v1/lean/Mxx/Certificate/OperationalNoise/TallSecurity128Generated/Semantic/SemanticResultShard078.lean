import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard078
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard072
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard075
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard076
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard077

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult9740
def owner : Owner := ⟨.program ⟨257⟩, ⟨63125⟩⟩
def rawTerms : List Term := Proof.Events038.exact9740RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9740
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9740.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9737) (rightBinding := 9738)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60145⟩) (rightExpression := ⟨63124⟩)
    (transferEvent := 9739)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9736.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9640.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9740

namespace SemanticResult9744
def owner : Owner := ⟨.program ⟨257⟩, ⟨66730⟩⟩
def rawTerms : List Term := Proof.Events038.exact9744RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9744
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9744.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9741) (rightBinding := 9742)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63125⟩) (rightExpression := ⟨66729⟩)
    (transferEvent := 9743)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9740.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9632.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9744

namespace SemanticResult9748
def owner : Owner := ⟨.program ⟨257⟩, ⟨66731⟩⟩
def rawTerms : List Term := Proof.Events038.exact9748RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9748
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9748.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9745) (rightBinding := 9746)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66730⟩) (rightExpression := ⟨26649⟩)
    (transferEvent := 9747)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9744.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9624.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9748

namespace SemanticResult9752
def owner : Owner := ⟨.program ⟨257⟩, ⟨66732⟩⟩
def rawTerms : List Term := Proof.Events038.exact9752RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9752
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9752.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9749) (rightBinding := 9750)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66731⟩) (rightExpression := ⟨29329⟩)
    (transferEvent := 9751)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9748.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9616.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9752

namespace SemanticResult9756
def owner : Owner := ⟨.program ⟨257⟩, ⟨66733⟩⟩
def rawTerms : List Term := Proof.Events038.exact9756RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9756
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9756.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9753) (rightBinding := 9754)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66732⟩) (rightExpression := ⟨34986⟩)
    (transferEvent := 9755)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9752.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9608.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9756

namespace SemanticResult9760
def owner : Owner := ⟨.program ⟨257⟩, ⟨66734⟩⟩
def rawTerms : List Term := Proof.Events038.exact9760RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9760
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9760.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9757) (rightBinding := 9758)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66733⟩) (rightExpression := ⟨37666⟩)
    (transferEvent := 9759)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9756.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9600.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9760

namespace SemanticResult9764
def owner : Owner := ⟨.program ⟨257⟩, ⟨66735⟩⟩
def rawTerms : List Term := Proof.Events038.exact9764RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9764
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9764.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9761) (rightBinding := 9762)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66734⟩) (rightExpression := ⟨40349⟩)
    (transferEvent := 9763)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9760.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9592.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9764

namespace SemanticResult9768
def owner : Owner := ⟨.program ⟨257⟩, ⟨66736⟩⟩
def rawTerms : List Term := Proof.Events038.exact9768RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9768
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9768.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9765) (rightBinding := 9766)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66735⟩) (rightExpression := ⟨43029⟩)
    (transferEvent := 9767)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9764.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9584.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9768

namespace SemanticResult9772
def owner : Owner := ⟨.program ⟨257⟩, ⟨66737⟩⟩
def rawTerms : List Term := Proof.Events038.exact9772RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9772
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9772.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9769) (rightBinding := 9770)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66736⟩) (rightExpression := ⟨45706⟩)
    (transferEvent := 9771)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9768.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9576.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9772

namespace SemanticResult9776
def owner : Owner := ⟨.program ⟨257⟩, ⟨66738⟩⟩
def rawTerms : List Term := Proof.Events038.exact9776RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9776
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9776.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9773) (rightBinding := 9774)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66737⟩) (rightExpression := ⟨48386⟩)
    (transferEvent := 9775)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9772.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9568.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9776

namespace SemanticResult9780
def owner : Owner := ⟨.program ⟨257⟩, ⟨67497⟩⟩
def rawTerms : List Term := Proof.Events038.exact9780RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9780
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9780.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9777) (rightBinding := 9778)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66738⟩) (rightExpression := ⟨67495⟩)
    (transferEvent := 9779)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9776.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9560.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9780

namespace SemanticResult9803
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def rawTerms : List Term := Proof.Events038.exact9803RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9803
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9803.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge9784.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge9784.frameStart)
    (transferEvent := 9783) (owner := owner)
    (leftResult := 9780) (rightResult := 9057)
    (working := LeftOperatorMerge9784.working)
    (reconstruction := LeftOperatorMerge9784.reconstruction)
    (leftReference := .predecessor 0 9781 .coefficient) (rightReference := .predecessor 1 9782 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult9780.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9057.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge9784.operationAgreement
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
end SemanticResult9803

namespace SemanticResult9805
def owner : Owner := ⟨.program ⟨257⟩, ⟨6770⟩⟩
def rawTerms : List Term := Proof.Events038.exact9805RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9805
def producerEvent : Nat := 9804
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9805.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 0, .finite 358505090762917939594344689123238600163555732303085698506375543281967191979063893391262832719097606360565606629795213856633053217224339593841464235295657933522017033812, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult9805

namespace SemanticResult9818
def owner : Owner := ⟨.program ⟨257⟩, ⟨47834⟩⟩
def rawTerms : List Term := Proof.Events038.exact9818RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9818
def producerEvent : Nat := 9817
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9818.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 0, .finite 60, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult9818

namespace SemanticResult9821
def owner : Owner := ⟨.program ⟨257⟩, ⟨15081⟩⟩
def rawTerms : List Term := Proof.Events038.exact9821RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9821
def producerEvent : Nat := 9820
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9821.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 0, .finite 60, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult9821

namespace SemanticResult9841
def owner : Owner := ⟨.program ⟨257⟩, ⟨45154⟩⟩
def rawTerms : List Term := Proof.Events038.exact9841RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9841
def producerEvent : Nat := 9840
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9841.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 0, .finite 58, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult9841

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
