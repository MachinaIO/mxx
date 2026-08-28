import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard121
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard007
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard013
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard019
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard025
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard031
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard037
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard043
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard049
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard054
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard120

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult15071
def owner : Owner := ⟨.program ⟨257⟩, ⟨67545⟩⟩
def rawTerms : List Term := Proof.Events058.exact15071RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15071
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15071.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15068) (rightBinding := 15069)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67544⟩) (rightExpression := ⟨67326⟩)
    (transferEvent := 15070)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15067.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6811.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15071

namespace SemanticResult15075
def owner : Owner := ⟨.program ⟨257⟩, ⟨67546⟩⟩
def rawTerms : List Term := Proof.Events058.exact15075RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15075
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15075.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15072) (rightBinding := 15073)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67545⟩) (rightExpression := ⟨67386⟩)
    (transferEvent := 15074)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15071.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6063.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15075

namespace SemanticResult15079
def owner : Owner := ⟨.program ⟨257⟩, ⟨67547⟩⟩
def rawTerms : List Term := Proof.Events058.exact15079RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15079
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15079.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15076) (rightBinding := 15077)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67546⟩) (rightExpression := ⟨67480⟩)
    (transferEvent := 15078)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15075.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5315.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15079

namespace SemanticResult15083
def owner : Owner := ⟨.program ⟨257⟩, ⟨67571⟩⟩
def rawTerms : List Term := Proof.Events058.exact15083RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15083
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15083.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15080) (rightBinding := 15081)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67547⟩) (rightExpression := ⟨67570⟩)
    (transferEvent := 15082)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15079.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4567.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15083

namespace SemanticResult15087
def owner : Owner := ⟨.program ⟨257⟩, ⟨67591⟩⟩
def rawTerms : List Term := Proof.Events058.exact15087RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15087
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15087.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15084) (rightBinding := 15085)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67571⟩) (rightExpression := ⟨67590⟩)
    (transferEvent := 15086)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15083.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3819.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15087

namespace SemanticResult15091
def owner : Owner := ⟨.program ⟨257⟩, ⟨67611⟩⟩
def rawTerms : List Term := Proof.Events058.exact15091RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15091
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15091.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15088) (rightBinding := 15089)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67591⟩) (rightExpression := ⟨67610⟩)
    (transferEvent := 15090)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15087.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3071.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15091

namespace SemanticResult15095
def owner : Owner := ⟨.program ⟨257⟩, ⟨67631⟩⟩
def rawTerms : List Term := Proof.Events058.exact15095RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15095
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15095.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15092) (rightBinding := 15093)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67611⟩) (rightExpression := ⟨67630⟩)
    (transferEvent := 15094)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15091.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2323.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15095

namespace SemanticResult15099
def owner : Owner := ⟨.program ⟨257⟩, ⟨67652⟩⟩
def rawTerms : List Term := Proof.Events058.exact15099RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15099
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15099.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15096) (rightBinding := 15097)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67631⟩) (rightExpression := ⟨67651⟩)
    (transferEvent := 15098)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15095.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1575.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15099

namespace SemanticResult15103
def owner : Owner := ⟨.program ⟨257⟩, ⟨67653⟩⟩
def rawTerms : List Term := Proof.Events058.exact15103RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15103
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15103.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15100) (rightBinding := 15101)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67652⟩) (rightExpression := ⟨67297⟩)
    (transferEvent := 15102)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15099.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult827.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15103

namespace SemanticResult15487
def owner : Owner := ⟨.program ⟨257⟩, ⟨67655⟩⟩
def rawTerms : List Term := Proof.Events060.exact15487RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15487
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15487.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge15107.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge15107.frameStart)
    (transferEvent := 15106) (owner := owner)
    (leftResult := 15103) (rightResult := 32)
    (working := LeftOperatorMerge15107.working)
    (reconstruction := LeftOperatorMerge15107.reconstruction)
    (leftReference := .predecessor 0 15104 .coefficient) (rightReference := .predecessor 1 15105 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult15103.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge15107.operationAgreement
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
end SemanticResult15487

namespace SemanticResult15492
def owner : Owner := ⟨.program ⟨257⟩, ⟨7030⟩⟩
def rawTerms : List Term := Proof.Events060.exact15492RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15492
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15492.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge15491.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge15491.frameStart)
    (transferEvent := 15490) (owner := owner)
    (leftResult := 2) (rightResult := 34)
    (working := LeftOperatorMerge15491.working)
    (reconstruction := LeftOperatorMerge15491.reconstruction)
    (leftReference := .predecessor 0 15488 .coefficient) (rightReference := .predecessor 1 15489 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult2.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult34.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge15491.operationAgreement
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
end SemanticResult15492

namespace SemanticResult15495
def owner : Owner := ⟨.program ⟨257⟩, ⟨7129⟩⟩
def rawTerms : List Term := Proof.Events060.exact15495RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15495
def producerEvent : Nat := 15494
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15495.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult15495

namespace SemanticResult15499
def owner : Owner := ⟨.program ⟨257⟩, ⟨7130⟩⟩
def rawTerms : List Term := Proof.Events060.exact15499RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15499
def producerEvent : Nat := 15498
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15499.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 15496 .coefficient) (.value (.predecessor 1 15497 .coefficient)), 0, .finite 8192, .scale (.predecessor 0 15496 .coefficient) (.value (.predecessor 1 15497 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult15499

namespace SemanticResult15503
def owner : Owner := ⟨.program ⟨257⟩, ⟨7235⟩⟩
def rawTerms : List Term := Proof.Events060.exact15503RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15503
def producerEvent : Nat := 15502
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15503.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult15503

namespace SemanticResult15506
def owner : Owner := ⟨.program ⟨257⟩, ⟨9491⟩⟩
def rawTerms : List Term := Proof.Events060.exact15506RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15506
def producerEvent : Nat := 15505
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15506.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult15506

namespace SemanticResult15510
def owner : Owner := ⟨.program ⟨257⟩, ⟨9492⟩⟩
def rawTerms : List Term := Proof.Events060.exact15510RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15510
def producerEvent : Nat := 15509
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15510.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 15507 .coefficient) (.value (.predecessor 1 15508 .coefficient)), 0, .finite 8192, .scale (.predecessor 0 15507 .coefficient) (.value (.predecessor 1 15508 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult15510

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
