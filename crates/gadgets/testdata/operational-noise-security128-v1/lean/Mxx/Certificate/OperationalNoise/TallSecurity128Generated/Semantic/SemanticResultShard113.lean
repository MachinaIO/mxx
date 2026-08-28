import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard113
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard005
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard111
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard112

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult14181
def owner : Owner := ⟨.program ⟨257⟩, ⟨15934⟩⟩
def rawTerms : List Term := Proof.Events055.exact14181RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14181
def producerEvent : Nat := 14180
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14181.actual selector witness
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
end SemanticResult14181

namespace SemanticResult14186
def owner : Owner := ⟨.program ⟨257⟩, ⟨15935⟩⟩
def rawTerms : List Term := Proof.Events055.exact14186RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14186
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14186.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge14185.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge14185.frameStart)
    (transferEvent := 14184) (owner := owner)
    (leftResult := 14181) (rightResult := 713)
    (working := LeftOperatorMerge14185.working)
    (reconstruction := LeftOperatorMerge14185.reconstruction)
    (leftReference := .predecessor 0 14182 .coefficient) (rightReference := .predecessor 1 14183 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult14181.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult713.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge14185.operationAgreement
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
end SemanticResult14186

namespace SemanticResult14190
def owner : Owner := ⟨.program ⟨257⟩, ⟨15936⟩⟩
def rawTerms : List Term := Proof.Events055.exact14190RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14190
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14190.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14187) (rightBinding := 14188)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6728⟩) (rightExpression := ⟨15935⟩)
    (transferEvent := 14189)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14186.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14190

namespace SemanticResult14194
def owner : Owner := ⟨.program ⟨257⟩, ⟨18749⟩⟩
def rawTerms : List Term := Proof.Events055.exact14194RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14194
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14194.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14191) (rightBinding := 14192)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15936⟩) (rightExpression := ⟨18748⟩)
    (transferEvent := 14193)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14190.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14178.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14194

namespace SemanticResult14198
def owner : Owner := ⟨.program ⟨257⟩, ⟨21969⟩⟩
def rawTerms : List Term := Proof.Events055.exact14198RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14198
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14198.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14195) (rightBinding := 14196)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18749⟩) (rightExpression := ⟨21968⟩)
    (transferEvent := 14197)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14194.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14170.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14198

namespace SemanticResult14202
def owner : Owner := ⟨.program ⟨257⟩, ⟨31989⟩⟩
def rawTerms : List Term := Proof.Events055.exact14202RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14202
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14202.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14199) (rightBinding := 14200)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21969⟩) (rightExpression := ⟨31988⟩)
    (transferEvent := 14201)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14198.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14162.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14202

namespace SemanticResult14206
def owner : Owner := ⟨.program ⟨257⟩, ⟨51053⟩⟩
def rawTerms : List Term := Proof.Events055.exact14206RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14206
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14206.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14203) (rightBinding := 14204)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨31989⟩) (rightExpression := ⟨51052⟩)
    (transferEvent := 14205)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14202.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14154.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14206

namespace SemanticResult14210
def owner : Owner := ⟨.program ⟨257⟩, ⟨54033⟩⟩
def rawTerms : List Term := Proof.Events055.exact14210RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14210
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14210.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14207) (rightBinding := 14208)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51053⟩) (rightExpression := ⟨54032⟩)
    (transferEvent := 14209)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14206.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14146.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14210

namespace SemanticResult14214
def owner : Owner := ⟨.program ⟨257⟩, ⟨57013⟩⟩
def rawTerms : List Term := Proof.Events055.exact14214RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14214
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14214.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14211) (rightBinding := 14212)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54033⟩) (rightExpression := ⟨57012⟩)
    (transferEvent := 14213)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14210.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14138.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14214

namespace SemanticResult14218
def owner : Owner := ⟨.program ⟨257⟩, ⟨59993⟩⟩
def rawTerms : List Term := Proof.Events055.exact14218RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14218
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14218.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14215) (rightBinding := 14216)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57013⟩) (rightExpression := ⟨59992⟩)
    (transferEvent := 14217)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14214.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14130.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14218

namespace SemanticResult14222
def owner : Owner := ⟨.program ⟨257⟩, ⟨62973⟩⟩
def rawTerms : List Term := Proof.Events055.exact14222RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14222
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14222.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14219) (rightBinding := 14220)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59993⟩) (rightExpression := ⟨62972⟩)
    (transferEvent := 14221)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14218.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14122.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14222

namespace SemanticResult14226
def owner : Owner := ⟨.program ⟨257⟩, ⟨66170⟩⟩
def rawTerms : List Term := Proof.Events055.exact14226RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14226
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14226.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14223) (rightBinding := 14224)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62973⟩) (rightExpression := ⟨66169⟩)
    (transferEvent := 14225)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14222.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14114.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14226

namespace SemanticResult14230
def owner : Owner := ⟨.program ⟨257⟩, ⟨66171⟩⟩
def rawTerms : List Term := Proof.Events055.exact14230RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14230
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14230.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14227) (rightBinding := 14228)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66170⟩) (rightExpression := ⟨26545⟩)
    (transferEvent := 14229)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14226.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14106.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14230

namespace SemanticResult14234
def owner : Owner := ⟨.program ⟨257⟩, ⟨66172⟩⟩
def rawTerms : List Term := Proof.Events055.exact14234RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14234
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14234.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14231) (rightBinding := 14232)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66171⟩) (rightExpression := ⟨29225⟩)
    (transferEvent := 14233)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14230.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14098.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14234

namespace SemanticResult14238
def owner : Owner := ⟨.program ⟨257⟩, ⟨66173⟩⟩
def rawTerms : List Term := Proof.Events055.exact14238RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14238
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14238.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14235) (rightBinding := 14236)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66172⟩) (rightExpression := ⟨34882⟩)
    (transferEvent := 14237)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14234.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14090.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14238

namespace SemanticResult14242
def owner : Owner := ⟨.program ⟨257⟩, ⟨66174⟩⟩
def rawTerms : List Term := Proof.Events055.exact14242RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14242
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult14242.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14239) (rightBinding := 14240)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66173⟩) (rightExpression := ⟨37562⟩)
    (transferEvent := 14241)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14238.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14082.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14242

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
