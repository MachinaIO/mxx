import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard120
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard060
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard066
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard072
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard078
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard084
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard090
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard096
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard102
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard108
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard114
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard116
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard117
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard119

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult14988
def owner : Owner := ⟨.program ⟨257⟩, ⟨65896⟩⟩
def rawTerms : List Term := Proof.Events058.exact14988RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14988
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14988.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14985) (rightBinding := 14986)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65895⟩) (rightExpression := ⟨42873⟩)
    (transferEvent := 14987)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14984.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14804.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14988

namespace SemanticResult14992
def owner : Owner := ⟨.program ⟨257⟩, ⟨65897⟩⟩
def rawTerms : List Term := Proof.Events058.exact14992RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14992
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14992.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14989) (rightBinding := 14990)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65896⟩) (rightExpression := ⟨45550⟩)
    (transferEvent := 14991)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14988.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14796.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14992

namespace SemanticResult14996
def owner : Owner := ⟨.program ⟨257⟩, ⟨65898⟩⟩
def rawTerms : List Term := Proof.Events058.exact14996RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14996
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14996.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14993) (rightBinding := 14994)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65897⟩) (rightExpression := ⟨48230⟩)
    (transferEvent := 14995)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14992.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14788.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14996

namespace SemanticResult15000
def owner : Owner := ⟨.program ⟨257⟩, ⟨67274⟩⟩
def rawTerms : List Term := Proof.Events058.exact15000RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15000
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15000.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14997) (rightBinding := 14998)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65898⟩) (rightExpression := ⟨67272⟩)
    (transferEvent := 14999)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14996.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14780.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15000

namespace SemanticResult15023
def owner : Owner := ⟨.program ⟨257⟩, ⟨67275⟩⟩
def rawTerms : List Term := Proof.Events058.exact15023RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15023
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15023.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge15004.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge15004.frameStart)
    (transferEvent := 15003) (owner := owner)
    (leftResult := 15000) (rightResult := 14287)
    (working := LeftOperatorMerge15004.working)
    (reconstruction := LeftOperatorMerge15004.reconstruction)
    (leftReference := .predecessor 0 15001 .coefficient) (rightReference := .predecessor 1 15002 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult15000.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14287.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge15004.operationAgreement
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
end SemanticResult15023

namespace SemanticResult15027
def owner : Owner := ⟨.program ⟨257⟩, ⟨67276⟩⟩
def rawTerms : List Term := Proof.Events058.exact15027RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15027
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15027.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15024) (rightBinding := 15025)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6728⟩) (rightExpression := ⟨67275⟩)
    (transferEvent := 15026)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15023.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15027

namespace SemanticResult15031
def owner : Owner := ⟨.program ⟨257⟩, ⟨67346⟩⟩
def rawTerms : List Term := Proof.Events058.exact15031RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15031
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15031.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15028) (rightBinding := 15029)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67276⟩) (rightExpression := ⟨67345⟩)
    (transferEvent := 15030)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15027.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14285.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15031

namespace SemanticResult15035
def owner : Owner := ⟨.program ⟨257⟩, ⟨67347⟩⟩
def rawTerms : List Term := Proof.Events058.exact15035RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15035
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15035.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15032) (rightBinding := 15033)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67346⟩) (rightExpression := ⟨67304⟩)
    (transferEvent := 15034)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15031.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13543.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15035

namespace SemanticResult15039
def owner : Owner := ⟨.program ⟨257⟩, ⟨67368⟩⟩
def rawTerms : List Term := Proof.Events058.exact15039RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15039
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15039.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15036) (rightBinding := 15037)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67347⟩) (rightExpression := ⟨67367⟩)
    (transferEvent := 15038)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15035.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12795.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15039

namespace SemanticResult15043
def owner : Owner := ⟨.program ⟨257⟩, ⟨67422⟩⟩
def rawTerms : List Term := Proof.Events058.exact15043RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15043
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15043.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15040) (rightBinding := 15041)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67368⟩) (rightExpression := ⟨67421⟩)
    (transferEvent := 15042)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15039.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12047.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15043

namespace SemanticResult15047
def owner : Owner := ⟨.program ⟨257⟩, ⟨67442⟩⟩
def rawTerms : List Term := Proof.Events058.exact15047RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15047
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15047.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15044) (rightBinding := 15045)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67422⟩) (rightExpression := ⟨67441⟩)
    (transferEvent := 15046)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15043.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11299.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15047

namespace SemanticResult15051
def owner : Owner := ⟨.program ⟨257⟩, ⟨67462⟩⟩
def rawTerms : List Term := Proof.Events058.exact15051RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15051
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15051.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15048) (rightBinding := 15049)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67442⟩) (rightExpression := ⟨67461⟩)
    (transferEvent := 15050)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15047.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10551.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15051

namespace SemanticResult15055
def owner : Owner := ⟨.program ⟨257⟩, ⟨67499⟩⟩
def rawTerms : List Term := Proof.Events058.exact15055RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15055
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15055.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15052) (rightBinding := 15053)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67462⟩) (rightExpression := ⟨67498⟩)
    (transferEvent := 15054)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15051.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9803.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15055

namespace SemanticResult15059
def owner : Owner := ⟨.program ⟨257⟩, ⟨67519⟩⟩
def rawTerms : List Term := Proof.Events058.exact15059RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15059
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15059.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15056) (rightBinding := 15057)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67499⟩) (rightExpression := ⟨67518⟩)
    (transferEvent := 15058)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15055.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9055.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15059

namespace SemanticResult15063
def owner : Owner := ⟨.program ⟨257⟩, ⟨67543⟩⟩
def rawTerms : List Term := Proof.Events058.exact15063RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15063
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15063.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15060) (rightBinding := 15061)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67519⟩) (rightExpression := ⟨67542⟩)
    (transferEvent := 15062)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15059.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8307.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15063

namespace SemanticResult15067
def owner : Owner := ⟨.program ⟨257⟩, ⟨67544⟩⟩
def rawTerms : List Term := Proof.Events058.exact15067RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15067
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15067.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15064) (rightBinding := 15065)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67543⟩) (rightExpression := ⟨67403⟩)
    (transferEvent := 15066)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15063.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7559.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15067

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
