import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard923
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard922

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult129645
def owner : Owner := ⟨.program ⟨257⟩, ⟨15971⟩⟩
def rawTerms : List Term := Proof.Events506.exact129645RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129645
def producerEvent : Nat := 129644
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129645.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 129211, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult129645

namespace SemanticResult129649
def owner : Owner := ⟨.program ⟨257⟩, ⟨18791⟩⟩
def rawTerms : List Term := Proof.Events506.exact129649RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129649
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129649.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 129646) (rightBinding := 129647)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15971⟩) (rightExpression := ⟨18790⟩)
    (transferEvent := 129648)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult129645.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult129622.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult129649

namespace SemanticResult129653
def owner : Owner := ⟨.program ⟨257⟩, ⟨22011⟩⟩
def rawTerms : List Term := Proof.Events506.exact129653RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129653
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129653.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 129650) (rightBinding := 129651)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18791⟩) (rightExpression := ⟨22010⟩)
    (transferEvent := 129652)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult129649.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult129599.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult129653

namespace SemanticResult129657
def owner : Owner := ⟨.program ⟨257⟩, ⟨32031⟩⟩
def rawTerms : List Term := Proof.Events506.exact129657RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129657
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129657.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 129654) (rightBinding := 129655)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22011⟩) (rightExpression := ⟨32030⟩)
    (transferEvent := 129656)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult129653.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult129576.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult129657

namespace SemanticResult129661
def owner : Owner := ⟨.program ⟨257⟩, ⟨51086⟩⟩
def rawTerms : List Term := Proof.Events506.exact129661RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129661
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129661.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 129658) (rightBinding := 129659)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32031⟩) (rightExpression := ⟨51085⟩)
    (transferEvent := 129660)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult129657.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult129553.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult129661

namespace SemanticResult129665
def owner : Owner := ⟨.program ⟨257⟩, ⟨54066⟩⟩
def rawTerms : List Term := Proof.Events506.exact129665RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129665
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129665.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 129662) (rightBinding := 129663)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51086⟩) (rightExpression := ⟨54065⟩)
    (transferEvent := 129664)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult129661.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult129530.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult129665

namespace SemanticResult129669
def owner : Owner := ⟨.program ⟨257⟩, ⟨57046⟩⟩
def rawTerms : List Term := Proof.Events506.exact129669RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129669
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129669.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 129666) (rightBinding := 129667)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54066⟩) (rightExpression := ⟨57045⟩)
    (transferEvent := 129668)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult129665.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult129507.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult129669

namespace SemanticResult129673
def owner : Owner := ⟨.program ⟨257⟩, ⟨60026⟩⟩
def rawTerms : List Term := Proof.Events506.exact129673RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129673
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129673.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 129670) (rightBinding := 129671)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57046⟩) (rightExpression := ⟨60025⟩)
    (transferEvent := 129672)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult129669.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult129484.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult129673

namespace SemanticResult129677
def owner : Owner := ⟨.program ⟨257⟩, ⟨63006⟩⟩
def rawTerms : List Term := Proof.Events506.exact129677RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129677
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129677.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 129674) (rightBinding := 129675)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60026⟩) (rightExpression := ⟨63005⟩)
    (transferEvent := 129676)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult129673.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult129461.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult129677

namespace SemanticResult129681
def owner : Owner := ⟨.program ⟨257⟩, ⟨66322⟩⟩
def rawTerms : List Term := Proof.Events506.exact129681RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129681
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129681.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 129678) (rightBinding := 129679)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63006⟩) (rightExpression := ⟨66321⟩)
    (transferEvent := 129680)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult129677.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult129438.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult129681

namespace SemanticResult129685
def owner : Owner := ⟨.program ⟨257⟩, ⟨66323⟩⟩
def rawTerms : List Term := Proof.Events506.exact129685RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129685
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129685.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 129682) (rightBinding := 129683)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66322⟩) (rightExpression := ⟨26567⟩)
    (transferEvent := 129684)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult129681.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult129415.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult129685

namespace SemanticResult129689
def owner : Owner := ⟨.program ⟨257⟩, ⟨66324⟩⟩
def rawTerms : List Term := Proof.Events506.exact129689RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129689
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129689.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 129686) (rightBinding := 129687)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66323⟩) (rightExpression := ⟨29247⟩)
    (transferEvent := 129688)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult129685.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult129392.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult129689

namespace SemanticResult129693
def owner : Owner := ⟨.program ⟨257⟩, ⟨66325⟩⟩
def rawTerms : List Term := Proof.Events506.exact129693RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129693
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129693.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 129690) (rightBinding := 129691)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66324⟩) (rightExpression := ⟨34911⟩)
    (transferEvent := 129692)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult129689.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult129369.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult129693

namespace SemanticResult129697
def owner : Owner := ⟨.program ⟨257⟩, ⟨66326⟩⟩
def rawTerms : List Term := Proof.Events506.exact129697RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129697
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129697.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 129694) (rightBinding := 129695)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66325⟩) (rightExpression := ⟨37591⟩)
    (transferEvent := 129696)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult129693.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult129346.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult129697

namespace SemanticResult129701
def owner : Owner := ⟨.program ⟨257⟩, ⟨66327⟩⟩
def rawTerms : List Term := Proof.Events506.exact129701RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129701
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129701.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 129698) (rightBinding := 129699)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66326⟩) (rightExpression := ⟨40267⟩)
    (transferEvent := 129700)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult129697.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult129323.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult129701

namespace SemanticResult129705
def owner : Owner := ⟨.program ⟨257⟩, ⟨66328⟩⟩
def rawTerms : List Term := Proof.Events506.exact129705RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 129705
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult129705.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 129702) (rightBinding := 129703)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66327⟩) (rightExpression := ⟨42947⟩)
    (transferEvent := 129704)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult129701.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult129300.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult129705

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
