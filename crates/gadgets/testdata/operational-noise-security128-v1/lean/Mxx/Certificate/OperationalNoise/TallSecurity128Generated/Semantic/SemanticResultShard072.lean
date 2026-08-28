import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard072
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard066
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard069
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard070
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard071

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult8988
def owner : Owner := ⟨.program ⟨257⟩, ⟨60164⟩⟩
def rawTerms : List Term := Proof.Events035.exact8988RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8988
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8988.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8985) (rightBinding := 8986)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57184⟩) (rightExpression := ⟨60163⟩)
    (transferEvent := 8987)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8984.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8900.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8988

namespace SemanticResult8992
def owner : Owner := ⟨.program ⟨257⟩, ⟨63144⟩⟩
def rawTerms : List Term := Proof.Events035.exact8992RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8992
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8992.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8989) (rightBinding := 8990)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60164⟩) (rightExpression := ⟨63143⟩)
    (transferEvent := 8991)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8988.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8892.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8992

namespace SemanticResult8996
def owner : Owner := ⟨.program ⟨257⟩, ⟨66800⟩⟩
def rawTerms : List Term := Proof.Events035.exact8996RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8996
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8996.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8993) (rightBinding := 8994)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63144⟩) (rightExpression := ⟨66799⟩)
    (transferEvent := 8995)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8992.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8884.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8996

namespace SemanticResult9000
def owner : Owner := ⟨.program ⟨257⟩, ⟨66801⟩⟩
def rawTerms : List Term := Proof.Events035.exact9000RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9000
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9000.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8997) (rightBinding := 8998)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66800⟩) (rightExpression := ⟨26662⟩)
    (transferEvent := 8999)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8996.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8876.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9000

namespace SemanticResult9004
def owner : Owner := ⟨.program ⟨257⟩, ⟨66802⟩⟩
def rawTerms : List Term := Proof.Events035.exact9004RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9004
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9004.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9001) (rightBinding := 9002)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66801⟩) (rightExpression := ⟨29342⟩)
    (transferEvent := 9003)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9000.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8868.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9004

namespace SemanticResult9008
def owner : Owner := ⟨.program ⟨257⟩, ⟨66803⟩⟩
def rawTerms : List Term := Proof.Events035.exact9008RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9008
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9008.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9005) (rightBinding := 9006)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66802⟩) (rightExpression := ⟨34999⟩)
    (transferEvent := 9007)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9004.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8860.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9008

namespace SemanticResult9012
def owner : Owner := ⟨.program ⟨257⟩, ⟨66804⟩⟩
def rawTerms : List Term := Proof.Events035.exact9012RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9012
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9012.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9009) (rightBinding := 9010)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66803⟩) (rightExpression := ⟨37679⟩)
    (transferEvent := 9011)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9008.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8852.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9012

namespace SemanticResult9016
def owner : Owner := ⟨.program ⟨257⟩, ⟨66805⟩⟩
def rawTerms : List Term := Proof.Events035.exact9016RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9016
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9016.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9013) (rightBinding := 9014)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66804⟩) (rightExpression := ⟨40362⟩)
    (transferEvent := 9015)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9012.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8844.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9016

namespace SemanticResult9020
def owner : Owner := ⟨.program ⟨257⟩, ⟨66806⟩⟩
def rawTerms : List Term := Proof.Events035.exact9020RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9020
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9020.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9017) (rightBinding := 9018)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66805⟩) (rightExpression := ⟨43042⟩)
    (transferEvent := 9019)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9016.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8836.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9020

namespace SemanticResult9024
def owner : Owner := ⟨.program ⟨257⟩, ⟨66807⟩⟩
def rawTerms : List Term := Proof.Events035.exact9024RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9024
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9024.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9021) (rightBinding := 9022)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66806⟩) (rightExpression := ⟨45719⟩)
    (transferEvent := 9023)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9020.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8828.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9024

namespace SemanticResult9028
def owner : Owner := ⟨.program ⟨257⟩, ⟨66808⟩⟩
def rawTerms : List Term := Proof.Events035.exact9028RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9028
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9028.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9025) (rightBinding := 9026)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66807⟩) (rightExpression := ⟨48399⟩)
    (transferEvent := 9027)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9024.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8820.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9028

namespace SemanticResult9032
def owner : Owner := ⟨.program ⟨257⟩, ⟨67517⟩⟩
def rawTerms : List Term := Proof.Events035.exact9032RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9032
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9032.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9029) (rightBinding := 9030)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66808⟩) (rightExpression := ⟨67515⟩)
    (transferEvent := 9031)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9028.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8812.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9032

namespace SemanticResult9055
def owner : Owner := ⟨.program ⟨257⟩, ⟨67518⟩⟩
def rawTerms : List Term := Proof.Events035.exact9055RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9055
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9055.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge9036.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge9036.frameStart)
    (transferEvent := 9035) (owner := owner)
    (leftResult := 9032) (rightResult := 8309)
    (working := LeftOperatorMerge9036.working)
    (reconstruction := LeftOperatorMerge9036.reconstruction)
    (leftReference := .predecessor 0 9033 .coefficient) (rightReference := .predecessor 1 9034 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult9032.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8309.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge9036.operationAgreement
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
end SemanticResult9055

namespace SemanticResult9057
def owner : Owner := ⟨.program ⟨257⟩, ⟨6907⟩⟩
def rawTerms : List Term := Proof.Events035.exact9057RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9057
def producerEvent : Nat := 9056
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9057.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 0, .finite 949765472837786621461281086895049655309960562397560588181162721740365167011484274077568270110122507580996980746643175131859041239136843301439062583529674884680451583842, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult9057

namespace SemanticResult9070
def owner : Owner := ⟨.program ⟨257⟩, ⟨47882⟩⟩
def rawTerms : List Term := Proof.Events035.exact9070RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9070
def producerEvent : Nat := 9069
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9070.actual selector witness
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
end SemanticResult9070

namespace SemanticResult9073
def owner : Owner := ⟨.program ⟨257⟩, ⟨15111⟩⟩
def rawTerms : List Term := Proof.Events035.exact9073RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9073
def producerEvent : Nat := 9072
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult9073.actual selector witness
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
end SemanticResult9073

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
