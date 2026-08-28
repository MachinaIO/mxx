import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard095
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard005
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard094

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult11932
def owner : Owner := ⟨.program ⟨257⟩, ⟨22044⟩⟩
def rawTerms : List Term := Proof.Events046.exact11932RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11932
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11932.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge11931.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge11931.frameStart)
    (transferEvent := 11930) (owner := owner)
    (leftResult := 11927) (rightResult := 693)
    (working := LeftOperatorMerge11931.working)
    (reconstruction := LeftOperatorMerge11931.reconstruction)
    (leftReference := .predecessor 0 11928 .coefficient) (rightReference := .predecessor 1 11929 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult11927.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult693.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge11931.operationAgreement
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
end SemanticResult11932

namespace SemanticResult11935
def owner : Owner := ⟨.program ⟨257⟩, ⟨18823⟩⟩
def rawTerms : List Term := Proof.Events046.exact11935RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11935
def producerEvent : Nat := 11934
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11935.actual selector witness
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
end SemanticResult11935

namespace SemanticResult11940
def owner : Owner := ⟨.program ⟨257⟩, ⟨18824⟩⟩
def rawTerms : List Term := Proof.Events046.exact11940RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11940
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11940.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge11939.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge11939.frameStart)
    (transferEvent := 11938) (owner := owner)
    (leftResult := 11935) (rightResult := 703)
    (working := LeftOperatorMerge11939.working)
    (reconstruction := LeftOperatorMerge11939.reconstruction)
    (leftReference := .predecessor 0 11936 .coefficient) (rightReference := .predecessor 1 11937 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult11935.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult703.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge11939.operationAgreement
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
end SemanticResult11940

namespace SemanticResult11943
def owner : Owner := ⟨.program ⟨257⟩, ⟨15998⟩⟩
def rawTerms : List Term := Proof.Events046.exact11943RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11943
def producerEvent : Nat := 11942
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11943.actual selector witness
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
end SemanticResult11943

namespace SemanticResult11948
def owner : Owner := ⟨.program ⟨257⟩, ⟨15999⟩⟩
def rawTerms : List Term := Proof.Events046.exact11948RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11948
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11948.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge11947.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge11947.frameStart)
    (transferEvent := 11946) (owner := owner)
    (leftResult := 11943) (rightResult := 713)
    (working := LeftOperatorMerge11947.working)
    (reconstruction := LeftOperatorMerge11947.reconstruction)
    (leftReference := .predecessor 0 11944 .coefficient) (rightReference := .predecessor 1 11945 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult11943.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult713.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge11947.operationAgreement
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
end SemanticResult11948

namespace SemanticResult11952
def owner : Owner := ⟨.program ⟨257⟩, ⟨16000⟩⟩
def rawTerms : List Term := Proof.Events046.exact11952RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11952
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11952.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 11949) (rightBinding := 11950)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6728⟩) (rightExpression := ⟨15999⟩)
    (transferEvent := 11951)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11948.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult11952

namespace SemanticResult11956
def owner : Owner := ⟨.program ⟨257⟩, ⟨18825⟩⟩
def rawTerms : List Term := Proof.Events046.exact11956RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11956
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11956.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 11953) (rightBinding := 11954)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16000⟩) (rightExpression := ⟨18824⟩)
    (transferEvent := 11955)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult11952.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11940.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult11956

namespace SemanticResult11960
def owner : Owner := ⟨.program ⟨257⟩, ⟨22045⟩⟩
def rawTerms : List Term := Proof.Events046.exact11960RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11960
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11960.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 11957) (rightBinding := 11958)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18825⟩) (rightExpression := ⟨22044⟩)
    (transferEvent := 11959)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult11956.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11932.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult11960

namespace SemanticResult11964
def owner : Owner := ⟨.program ⟨257⟩, ⟨32065⟩⟩
def rawTerms : List Term := Proof.Events046.exact11964RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11964
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11964.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 11961) (rightBinding := 11962)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22045⟩) (rightExpression := ⟨32064⟩)
    (transferEvent := 11963)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult11960.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11924.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult11964

namespace SemanticResult11968
def owner : Owner := ⟨.program ⟨257⟩, ⟨51129⟩⟩
def rawTerms : List Term := Proof.Events046.exact11968RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11968
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11968.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 11965) (rightBinding := 11966)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32065⟩) (rightExpression := ⟨51128⟩)
    (transferEvent := 11967)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult11964.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11916.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult11968

namespace SemanticResult11972
def owner : Owner := ⟨.program ⟨257⟩, ⟨54109⟩⟩
def rawTerms : List Term := Proof.Events046.exact11972RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11972
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11972.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 11969) (rightBinding := 11970)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51129⟩) (rightExpression := ⟨54108⟩)
    (transferEvent := 11971)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult11968.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11908.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult11972

namespace SemanticResult11976
def owner : Owner := ⟨.program ⟨257⟩, ⟨57089⟩⟩
def rawTerms : List Term := Proof.Events046.exact11976RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11976
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11976.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 11973) (rightBinding := 11974)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54109⟩) (rightExpression := ⟨57088⟩)
    (transferEvent := 11975)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult11972.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11900.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult11976

namespace SemanticResult11980
def owner : Owner := ⟨.program ⟨257⟩, ⟨60069⟩⟩
def rawTerms : List Term := Proof.Events046.exact11980RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11980
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11980.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 11977) (rightBinding := 11978)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57089⟩) (rightExpression := ⟨60068⟩)
    (transferEvent := 11979)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult11976.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11892.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult11980

namespace SemanticResult11984
def owner : Owner := ⟨.program ⟨257⟩, ⟨63049⟩⟩
def rawTerms : List Term := Proof.Events046.exact11984RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11984
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11984.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 11981) (rightBinding := 11982)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60069⟩) (rightExpression := ⟨63048⟩)
    (transferEvent := 11983)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult11980.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11884.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult11984

namespace SemanticResult11988
def owner : Owner := ⟨.program ⟨257⟩, ⟨66450⟩⟩
def rawTerms : List Term := Proof.Events046.exact11988RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11988
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11988.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 11985) (rightBinding := 11986)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63049⟩) (rightExpression := ⟨66449⟩)
    (transferEvent := 11987)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult11984.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11876.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult11988

namespace SemanticResult11992
def owner : Owner := ⟨.program ⟨257⟩, ⟨66451⟩⟩
def rawTerms : List Term := Proof.Events046.exact11992RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11992
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult11992.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 11989) (rightBinding := 11990)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66450⟩) (rightExpression := ⟨26597⟩)
    (transferEvent := 11991)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult11988.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11868.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult11992

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
