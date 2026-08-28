import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1426
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1424
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1425

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult202782
def owner : Owner := ⟨.program ⟨257⟩, ⟨32145⟩⟩
def rawTerms : List Term := Proof.Events792.exact202782RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202782
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202782.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 202779) (rightBinding := 202780)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22125⟩) (rightExpression := ⟨32144⟩)
    (transferEvent := 202781)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult202778.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult202701.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult202782

namespace SemanticResult202786
def owner : Owner := ⟨.program ⟨257⟩, ⟨51200⟩⟩
def rawTerms : List Term := Proof.Events792.exact202786RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202786
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202786.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 202783) (rightBinding := 202784)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32145⟩) (rightExpression := ⟨51199⟩)
    (transferEvent := 202785)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult202782.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult202678.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult202786

namespace SemanticResult202790
def owner : Owner := ⟨.program ⟨257⟩, ⟨54180⟩⟩
def rawTerms : List Term := Proof.Events792.exact202790RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202790
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202790.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 202787) (rightBinding := 202788)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51200⟩) (rightExpression := ⟨54179⟩)
    (transferEvent := 202789)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult202786.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult202655.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult202790

namespace SemanticResult202794
def owner : Owner := ⟨.program ⟨257⟩, ⟨57160⟩⟩
def rawTerms : List Term := Proof.Events792.exact202794RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202794
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202794.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 202791) (rightBinding := 202792)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54180⟩) (rightExpression := ⟨57159⟩)
    (transferEvent := 202793)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult202790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult202632.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult202794

namespace SemanticResult202798
def owner : Owner := ⟨.program ⟨257⟩, ⟨60140⟩⟩
def rawTerms : List Term := Proof.Events792.exact202798RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202798
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202798.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 202795) (rightBinding := 202796)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57160⟩) (rightExpression := ⟨60139⟩)
    (transferEvent := 202797)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult202794.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult202609.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult202798

namespace SemanticResult202802
def owner : Owner := ⟨.program ⟨257⟩, ⟨63120⟩⟩
def rawTerms : List Term := Proof.Events792.exact202802RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202802
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202802.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 202799) (rightBinding := 202800)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60140⟩) (rightExpression := ⟨63119⟩)
    (transferEvent := 202801)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult202798.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult202586.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult202802

namespace SemanticResult202806
def owner : Owner := ⟨.program ⟨257⟩, ⟨66742⟩⟩
def rawTerms : List Term := Proof.Events792.exact202806RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202806
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202806.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 202803) (rightBinding := 202804)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63120⟩) (rightExpression := ⟨66741⟩)
    (transferEvent := 202805)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult202802.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult202563.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult202806

namespace SemanticResult202810
def owner : Owner := ⟨.program ⟨257⟩, ⟨66743⟩⟩
def rawTerms : List Term := Proof.Events792.exact202810RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202810
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202810.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 202807) (rightBinding := 202808)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66742⟩) (rightExpression := ⟨26645⟩)
    (transferEvent := 202809)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult202806.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult202540.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult202810

namespace SemanticResult202814
def owner : Owner := ⟨.program ⟨257⟩, ⟨66744⟩⟩
def rawTerms : List Term := Proof.Events792.exact202814RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202814
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202814.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 202811) (rightBinding := 202812)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66743⟩) (rightExpression := ⟨29325⟩)
    (transferEvent := 202813)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult202810.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult202517.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult202814

namespace SemanticResult202818
def owner : Owner := ⟨.program ⟨257⟩, ⟨66745⟩⟩
def rawTerms : List Term := Proof.Events792.exact202818RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202818
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202818.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 202815) (rightBinding := 202816)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66744⟩) (rightExpression := ⟨34989⟩)
    (transferEvent := 202817)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult202814.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult202494.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult202818

namespace SemanticResult202822
def owner : Owner := ⟨.program ⟨257⟩, ⟨66746⟩⟩
def rawTerms : List Term := Proof.Events792.exact202822RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202822
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202822.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 202819) (rightBinding := 202820)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66745⟩) (rightExpression := ⟨37669⟩)
    (transferEvent := 202821)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult202818.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult202471.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult202822

namespace SemanticResult202826
def owner : Owner := ⟨.program ⟨257⟩, ⟨66747⟩⟩
def rawTerms : List Term := Proof.Events792.exact202826RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202826
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202826.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 202823) (rightBinding := 202824)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66746⟩) (rightExpression := ⟨40345⟩)
    (transferEvent := 202825)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult202822.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult202448.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult202826

namespace SemanticResult202830
def owner : Owner := ⟨.program ⟨257⟩, ⟨66748⟩⟩
def rawTerms : List Term := Proof.Events792.exact202830RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202830
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202830.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 202827) (rightBinding := 202828)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66747⟩) (rightExpression := ⟨43025⟩)
    (transferEvent := 202829)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult202826.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult202425.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult202830

namespace SemanticResult202834
def owner : Owner := ⟨.program ⟨257⟩, ⟨66749⟩⟩
def rawTerms : List Term := Proof.Events792.exact202834RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202834
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202834.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 202831) (rightBinding := 202832)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66748⟩) (rightExpression := ⟨45709⟩)
    (transferEvent := 202833)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult202830.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult202402.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult202834

namespace SemanticResult202838
def owner : Owner := ⟨.program ⟨257⟩, ⟨66750⟩⟩
def rawTerms : List Term := Proof.Events792.exact202838RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202838
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202838.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 202835) (rightBinding := 202836)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66749⟩) (rightExpression := ⟨48389⟩)
    (transferEvent := 202837)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult202834.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult202379.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult202838

namespace SemanticResult202849
def owner : Owner := ⟨.program ⟨257⟩, ⟨68842⟩⟩
def rawTerms : List Term := Proof.Events792.exact202849RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 202849
def producerEvent : Nat := 202848
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult202849.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 202336, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult202849

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
