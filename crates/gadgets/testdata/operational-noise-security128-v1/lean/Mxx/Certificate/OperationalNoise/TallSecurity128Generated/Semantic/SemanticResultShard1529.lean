import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1529
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1527
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1528

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult217572
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def rawTerms : List Term := Proof.Events849.exact217572RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217572
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217572.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217569) (rightBinding := 217570)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7309⟩) (rightExpression := ⟨7202⟩)
    (transferEvent := 217571)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217568.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217558.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217572

namespace SemanticResult217576
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def rawTerms : List Term := Proof.Events849.exact217576RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217576
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217576.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217573) (rightBinding := 217574)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7310⟩) (rightExpression := ⟨7204⟩)
    (transferEvent := 217575)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217572.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217555.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217576

namespace SemanticResult217580
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def rawTerms : List Term := Proof.Events849.exact217580RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217580
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217580.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217577) (rightBinding := 217578)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7311⟩) (rightExpression := ⟨7206⟩)
    (transferEvent := 217579)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217576.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217552.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217580

namespace SemanticResult217584
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def rawTerms : List Term := Proof.Events849.exact217584RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217584
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217584.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217581) (rightBinding := 217582)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7312⟩) (rightExpression := ⟨7208⟩)
    (transferEvent := 217583)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217580.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217549.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217584

namespace SemanticResult217588
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def rawTerms : List Term := Proof.Events849.exact217588RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217588
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217588.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217585) (rightBinding := 217586)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7313⟩) (rightExpression := ⟨7210⟩)
    (transferEvent := 217587)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217584.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217546.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217588

namespace SemanticResult217592
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def rawTerms : List Term := Proof.Events849.exact217592RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217592
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217592.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217589) (rightBinding := 217590)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7314⟩) (rightExpression := ⟨7212⟩)
    (transferEvent := 217591)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217588.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217543.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217592

namespace SemanticResult217596
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def rawTerms : List Term := Proof.Events849.exact217596RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217596
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217596.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217593) (rightBinding := 217594)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7315⟩) (rightExpression := ⟨7214⟩)
    (transferEvent := 217595)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217592.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217540.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217596

namespace SemanticResult217600
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def rawTerms : List Term := Proof.Events850.exact217600RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217600
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217600.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217597) (rightBinding := 217598)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7316⟩) (rightExpression := ⟨7216⟩)
    (transferEvent := 217599)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217596.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217537.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217600

namespace SemanticResult217604
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def rawTerms : List Term := Proof.Events850.exact217604RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217604
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217604.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217601) (rightBinding := 217602)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7317⟩) (rightExpression := ⟨7218⟩)
    (transferEvent := 217603)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217600.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217534.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217604

namespace SemanticResult217608
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def rawTerms : List Term := Proof.Events850.exact217608RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217608
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217608.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217605) (rightBinding := 217606)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7318⟩) (rightExpression := ⟨7220⟩)
    (transferEvent := 217607)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217604.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217531.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217608

namespace SemanticResult217612
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def rawTerms : List Term := Proof.Events850.exact217612RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217612
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217612.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217609) (rightBinding := 217610)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7319⟩) (rightExpression := ⟨7222⟩)
    (transferEvent := 217611)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217608.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217528.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217612

namespace SemanticResult217616
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def rawTerms : List Term := Proof.Events850.exact217616RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217616
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217616.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217613) (rightBinding := 217614)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7320⟩) (rightExpression := ⟨7224⟩)
    (transferEvent := 217615)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217612.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217525.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217616

namespace SemanticResult217620
def owner : Owner := ⟨.program ⟨257⟩, ⟨7322⟩⟩
def rawTerms : List Term := Proof.Events850.exact217620RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217620
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217620.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217617) (rightBinding := 217618)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7321⟩) (rightExpression := ⟨7226⟩)
    (transferEvent := 217619)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217616.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217522.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217620

namespace SemanticResult217624
def owner : Owner := ⟨.program ⟨257⟩, ⟨7323⟩⟩
def rawTerms : List Term := Proof.Events850.exact217624RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217624
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217624.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217621) (rightBinding := 217622)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7322⟩) (rightExpression := ⟨7228⟩)
    (transferEvent := 217623)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217620.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217519.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217624

namespace SemanticResult217628
def owner : Owner := ⟨.program ⟨257⟩, ⟨7324⟩⟩
def rawTerms : List Term := Proof.Events850.exact217628RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217628
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217628.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217625) (rightBinding := 217626)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7323⟩) (rightExpression := ⟨7230⟩)
    (transferEvent := 217627)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217624.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217516.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217628

namespace SemanticResult217632
def owner : Owner := ⟨.program ⟨257⟩, ⟨7325⟩⟩
def rawTerms : List Term := Proof.Events850.exact217632RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217632
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217632.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217629) (rightBinding := 217630)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7324⟩) (rightExpression := ⟨7232⟩)
    (transferEvent := 217631)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217628.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217513.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217632

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
