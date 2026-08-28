import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard119
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard005
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard117
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard118

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult14924
def owner : Owner := ⟨.program ⟨257⟩, ⟨15871⟩⟩
def rawTerms : List Term := Proof.Events058.exact14924RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14924
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14924.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge14923.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge14923.frameStart)
    (transferEvent := 14922) (owner := owner)
    (leftResult := 14919) (rightResult := 713)
    (working := LeftOperatorMerge14923.working)
    (reconstruction := LeftOperatorMerge14923.reconstruction)
    (leftReference := .predecessor 0 14920 .coefficient) (rightReference := .predecessor 1 14921 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult14919.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult713.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge14923.operationAgreement
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
end SemanticResult14924

namespace SemanticResult14928
def owner : Owner := ⟨.program ⟨257⟩, ⟨15872⟩⟩
def rawTerms : List Term := Proof.Events058.exact14928RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14928
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14928.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14925) (rightBinding := 14926)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6728⟩) (rightExpression := ⟨15871⟩)
    (transferEvent := 14927)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14924.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14928

namespace SemanticResult14932
def owner : Owner := ⟨.program ⟨257⟩, ⟨18673⟩⟩
def rawTerms : List Term := Proof.Events058.exact14932RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14932
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14932.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14929) (rightBinding := 14930)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15872⟩) (rightExpression := ⟨18672⟩)
    (transferEvent := 14931)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14928.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14916.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14932

namespace SemanticResult14936
def owner : Owner := ⟨.program ⟨257⟩, ⟨21893⟩⟩
def rawTerms : List Term := Proof.Events058.exact14936RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14936
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14936.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14933) (rightBinding := 14934)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18673⟩) (rightExpression := ⟨21892⟩)
    (transferEvent := 14935)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14932.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14908.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14936

namespace SemanticResult14940
def owner : Owner := ⟨.program ⟨257⟩, ⟨31913⟩⟩
def rawTerms : List Term := Proof.Events058.exact14940RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14940
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14940.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14937) (rightBinding := 14938)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21893⟩) (rightExpression := ⟨31912⟩)
    (transferEvent := 14939)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14936.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14900.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14940

namespace SemanticResult14944
def owner : Owner := ⟨.program ⟨257⟩, ⟨50977⟩⟩
def rawTerms : List Term := Proof.Events058.exact14944RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14944
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14944.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14941) (rightBinding := 14942)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨31913⟩) (rightExpression := ⟨50976⟩)
    (transferEvent := 14943)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14940.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14892.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14944

namespace SemanticResult14948
def owner : Owner := ⟨.program ⟨257⟩, ⟨53957⟩⟩
def rawTerms : List Term := Proof.Events058.exact14948RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14948
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14948.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14945) (rightBinding := 14946)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨50977⟩) (rightExpression := ⟨53956⟩)
    (transferEvent := 14947)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14944.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14884.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14948

namespace SemanticResult14952
def owner : Owner := ⟨.program ⟨257⟩, ⟨56937⟩⟩
def rawTerms : List Term := Proof.Events058.exact14952RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14952
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14952.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14949) (rightBinding := 14950)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53957⟩) (rightExpression := ⟨56936⟩)
    (transferEvent := 14951)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14948.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14876.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14952

namespace SemanticResult14956
def owner : Owner := ⟨.program ⟨257⟩, ⟨59917⟩⟩
def rawTerms : List Term := Proof.Events058.exact14956RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14956
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14956.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14953) (rightBinding := 14954)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56937⟩) (rightExpression := ⟨59916⟩)
    (transferEvent := 14955)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14952.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14868.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14956

namespace SemanticResult14960
def owner : Owner := ⟨.program ⟨257⟩, ⟨62897⟩⟩
def rawTerms : List Term := Proof.Events058.exact14960RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14960
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14960.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14957) (rightBinding := 14958)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59917⟩) (rightExpression := ⟨62896⟩)
    (transferEvent := 14959)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14956.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14860.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14960

namespace SemanticResult14964
def owner : Owner := ⟨.program ⟨257⟩, ⟨65890⟩⟩
def rawTerms : List Term := Proof.Events058.exact14964RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14964
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14964.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14961) (rightBinding := 14962)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62897⟩) (rightExpression := ⟨65889⟩)
    (transferEvent := 14963)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14960.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14852.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14964

namespace SemanticResult14968
def owner : Owner := ⟨.program ⟨257⟩, ⟨65891⟩⟩
def rawTerms : List Term := Proof.Events058.exact14968RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14968
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14968.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14965) (rightBinding := 14966)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65890⟩) (rightExpression := ⟨26493⟩)
    (transferEvent := 14967)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14964.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14844.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14968

namespace SemanticResult14972
def owner : Owner := ⟨.program ⟨257⟩, ⟨65892⟩⟩
def rawTerms : List Term := Proof.Events058.exact14972RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14972
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14972.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14969) (rightBinding := 14970)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65891⟩) (rightExpression := ⟨29173⟩)
    (transferEvent := 14971)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14968.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14836.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14972

namespace SemanticResult14976
def owner : Owner := ⟨.program ⟨257⟩, ⟨65893⟩⟩
def rawTerms : List Term := Proof.Events058.exact14976RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14976
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14976.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14973) (rightBinding := 14974)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65892⟩) (rightExpression := ⟨34830⟩)
    (transferEvent := 14975)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14972.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14828.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14976

namespace SemanticResult14980
def owner : Owner := ⟨.program ⟨257⟩, ⟨65894⟩⟩
def rawTerms : List Term := Proof.Events058.exact14980RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14980
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14980.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14977) (rightBinding := 14978)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65893⟩) (rightExpression := ⟨37510⟩)
    (transferEvent := 14979)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14976.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14820.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14980

namespace SemanticResult14984
def owner : Owner := ⟨.program ⟨257⟩, ⟨65895⟩⟩
def rawTerms : List Term := Proof.Events058.exact14984RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14984
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14984.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14981) (rightBinding := 14982)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65894⟩) (rightExpression := ⟨40193⟩)
    (transferEvent := 14983)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14980.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14812.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14984

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
