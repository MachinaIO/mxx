import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard060
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard055
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard057
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard058
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard059

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult7484
def owner : Owner := ⟨.program ⟨257⟩, ⟨54090⟩⟩
def rawTerms : List Term := Proof.Events029.exact7484RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7484
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7484.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7481) (rightBinding := 7482)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51110⟩) (rightExpression := ⟨54089⟩)
    (transferEvent := 7483)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7480.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7420.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7484

namespace SemanticResult7488
def owner : Owner := ⟨.program ⟨257⟩, ⟨57070⟩⟩
def rawTerms : List Term := Proof.Events029.exact7488RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7488
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7488.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7485) (rightBinding := 7486)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54090⟩) (rightExpression := ⟨57069⟩)
    (transferEvent := 7487)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7484.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7412.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7488

namespace SemanticResult7492
def owner : Owner := ⟨.program ⟨257⟩, ⟨60050⟩⟩
def rawTerms : List Term := Proof.Events029.exact7492RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7492
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7492.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7489) (rightBinding := 7490)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57070⟩) (rightExpression := ⟨60049⟩)
    (transferEvent := 7491)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7488.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7404.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7492

namespace SemanticResult7496
def owner : Owner := ⟨.program ⟨257⟩, ⟨63030⟩⟩
def rawTerms : List Term := Proof.Events029.exact7496RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7496
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7496.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7493) (rightBinding := 7494)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60050⟩) (rightExpression := ⟨63029⟩)
    (transferEvent := 7495)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7492.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7396.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7496

namespace SemanticResult7500
def owner : Owner := ⟨.program ⟨257⟩, ⟨66380⟩⟩
def rawTerms : List Term := Proof.Events029.exact7500RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7500
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7500.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7497) (rightBinding := 7498)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63030⟩) (rightExpression := ⟨66379⟩)
    (transferEvent := 7499)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7496.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7388.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7500

namespace SemanticResult7504
def owner : Owner := ⟨.program ⟨257⟩, ⟨66381⟩⟩
def rawTerms : List Term := Proof.Events029.exact7504RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7504
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7504.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7501) (rightBinding := 7502)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66380⟩) (rightExpression := ⟨26584⟩)
    (transferEvent := 7503)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7500.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7380.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7504

namespace SemanticResult7508
def owner : Owner := ⟨.program ⟨257⟩, ⟨66382⟩⟩
def rawTerms : List Term := Proof.Events029.exact7508RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7508
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7508.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7505) (rightBinding := 7506)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66381⟩) (rightExpression := ⟨29264⟩)
    (transferEvent := 7507)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7504.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7372.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7508

namespace SemanticResult7512
def owner : Owner := ⟨.program ⟨257⟩, ⟨66383⟩⟩
def rawTerms : List Term := Proof.Events029.exact7512RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7512
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7512.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7509) (rightBinding := 7510)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66382⟩) (rightExpression := ⟨34921⟩)
    (transferEvent := 7511)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7508.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7364.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7512

namespace SemanticResult7516
def owner : Owner := ⟨.program ⟨257⟩, ⟨66384⟩⟩
def rawTerms : List Term := Proof.Events029.exact7516RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7516
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7516.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7513) (rightBinding := 7514)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66383⟩) (rightExpression := ⟨37601⟩)
    (transferEvent := 7515)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7512.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7356.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7516

namespace SemanticResult7520
def owner : Owner := ⟨.program ⟨257⟩, ⟨66385⟩⟩
def rawTerms : List Term := Proof.Events029.exact7520RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7520
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7520.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7517) (rightBinding := 7518)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66384⟩) (rightExpression := ⟨40284⟩)
    (transferEvent := 7519)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7516.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7348.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7520

namespace SemanticResult7524
def owner : Owner := ⟨.program ⟨257⟩, ⟨66386⟩⟩
def rawTerms : List Term := Proof.Events029.exact7524RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7524
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7524.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7521) (rightBinding := 7522)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66385⟩) (rightExpression := ⟨42964⟩)
    (transferEvent := 7523)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7520.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7340.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7524

namespace SemanticResult7528
def owner : Owner := ⟨.program ⟨257⟩, ⟨66387⟩⟩
def rawTerms : List Term := Proof.Events029.exact7528RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7528
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7528.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7525) (rightBinding := 7526)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66386⟩) (rightExpression := ⟨45641⟩)
    (transferEvent := 7527)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7524.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7332.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7528

namespace SemanticResult7532
def owner : Owner := ⟨.program ⟨257⟩, ⟨66388⟩⟩
def rawTerms : List Term := Proof.Events029.exact7532RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7532
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7532.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7529) (rightBinding := 7530)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66387⟩) (rightExpression := ⟨48321⟩)
    (transferEvent := 7531)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7528.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7324.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7532

namespace SemanticResult7536
def owner : Owner := ⟨.program ⟨257⟩, ⟨67402⟩⟩
def rawTerms : List Term := Proof.Events029.exact7536RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7536
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7536.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7533) (rightBinding := 7534)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66388⟩) (rightExpression := ⟨67400⟩)
    (transferEvent := 7535)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7532.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7316.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7536

namespace SemanticResult7559
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def rawTerms : List Term := Proof.Events029.exact7559RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7559
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7559.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge7540.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge7540.frameStart)
    (transferEvent := 7539) (owner := owner)
    (leftResult := 7536) (rightResult := 6813)
    (working := LeftOperatorMerge7540.working)
    (reconstruction := LeftOperatorMerge7540.reconstruction)
    (leftReference := .predecessor 0 7537 .coefficient) (rightReference := .predecessor 1 7538 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult7536.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6813.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge7540.operationAgreement
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
end SemanticResult7559

namespace SemanticResult7561
def owner : Owner := ⟨.program ⟨257⟩, ⟨6765⟩⟩
def rawTerms : List Term := Proof.Events029.exact7561RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7561
def producerEvent : Nat := 7560
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult7561.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 0, .finite 209547688210549055471147046111004916489331190890252620496502021405337735671870380095231105730177606312631955343380640763509911328536630738066641741668496568757831236150, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult7561

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
