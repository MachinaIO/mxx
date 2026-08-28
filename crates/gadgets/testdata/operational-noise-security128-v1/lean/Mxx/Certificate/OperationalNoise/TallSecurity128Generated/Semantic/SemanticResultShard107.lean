import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard107
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard005
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard105
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard106

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult13436
def owner : Owner := ⟨.program ⟨257⟩, ⟨18705⟩⟩
def rawTerms : List Term := Proof.Events052.exact13436RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13436
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13436.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge13435.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge13435.frameStart)
    (transferEvent := 13434) (owner := owner)
    (leftResult := 13431) (rightResult := 703)
    (working := LeftOperatorMerge13435.working)
    (reconstruction := LeftOperatorMerge13435.reconstruction)
    (leftReference := .predecessor 0 13432 .coefficient) (rightReference := .predecessor 1 13433 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult13431.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult703.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge13435.operationAgreement
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
end SemanticResult13436

namespace SemanticResult13439
def owner : Owner := ⟨.program ⟨257⟩, ⟨15898⟩⟩
def rawTerms : List Term := Proof.Events052.exact13439RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13439
def producerEvent : Nat := 13438
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13439.actual selector witness
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
end SemanticResult13439

namespace SemanticResult13444
def owner : Owner := ⟨.program ⟨257⟩, ⟨15899⟩⟩
def rawTerms : List Term := Proof.Events052.exact13444RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13444
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13444.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge13443.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge13443.frameStart)
    (transferEvent := 13442) (owner := owner)
    (leftResult := 13439) (rightResult := 713)
    (working := LeftOperatorMerge13443.working)
    (reconstruction := LeftOperatorMerge13443.reconstruction)
    (leftReference := .predecessor 0 13440 .coefficient) (rightReference := .predecessor 1 13441 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult13439.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult713.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge13443.operationAgreement
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
end SemanticResult13444

namespace SemanticResult13448
def owner : Owner := ⟨.program ⟨257⟩, ⟨15900⟩⟩
def rawTerms : List Term := Proof.Events052.exact13448RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13448
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13448.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13445) (rightBinding := 13446)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6728⟩) (rightExpression := ⟨15899⟩)
    (transferEvent := 13447)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13444.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13448

namespace SemanticResult13452
def owner : Owner := ⟨.program ⟨257⟩, ⟨18706⟩⟩
def rawTerms : List Term := Proof.Events052.exact13452RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13452
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13452.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13449) (rightBinding := 13450)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15900⟩) (rightExpression := ⟨18705⟩)
    (transferEvent := 13451)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13448.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13436.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13452

namespace SemanticResult13456
def owner : Owner := ⟨.program ⟨257⟩, ⟨21926⟩⟩
def rawTerms : List Term := Proof.Events052.exact13456RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13456
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13456.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13453) (rightBinding := 13454)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18706⟩) (rightExpression := ⟨21925⟩)
    (transferEvent := 13455)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13452.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13428.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13456

namespace SemanticResult13460
def owner : Owner := ⟨.program ⟨257⟩, ⟨31946⟩⟩
def rawTerms : List Term := Proof.Events052.exact13460RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13460
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13460.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13457) (rightBinding := 13458)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21926⟩) (rightExpression := ⟨31945⟩)
    (transferEvent := 13459)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13456.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13420.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13460

namespace SemanticResult13464
def owner : Owner := ⟨.program ⟨257⟩, ⟨51010⟩⟩
def rawTerms : List Term := Proof.Events052.exact13464RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13464
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13464.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13461) (rightBinding := 13462)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨31946⟩) (rightExpression := ⟨51009⟩)
    (transferEvent := 13463)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13460.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13412.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13464

namespace SemanticResult13468
def owner : Owner := ⟨.program ⟨257⟩, ⟨53990⟩⟩
def rawTerms : List Term := Proof.Events052.exact13468RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13468
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13468.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13465) (rightBinding := 13466)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51010⟩) (rightExpression := ⟨53989⟩)
    (transferEvent := 13467)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13464.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13404.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13468

namespace SemanticResult13472
def owner : Owner := ⟨.program ⟨257⟩, ⟨56970⟩⟩
def rawTerms : List Term := Proof.Events052.exact13472RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13472
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13472.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13469) (rightBinding := 13470)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53990⟩) (rightExpression := ⟨56969⟩)
    (transferEvent := 13471)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13468.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13396.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13472

namespace SemanticResult13476
def owner : Owner := ⟨.program ⟨257⟩, ⟨59950⟩⟩
def rawTerms : List Term := Proof.Events052.exact13476RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13476
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13476.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13473) (rightBinding := 13474)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56970⟩) (rightExpression := ⟨59949⟩)
    (transferEvent := 13475)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13472.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13388.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13476

namespace SemanticResult13480
def owner : Owner := ⟨.program ⟨257⟩, ⟨62930⟩⟩
def rawTerms : List Term := Proof.Events052.exact13480RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13480
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13480.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13477) (rightBinding := 13478)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59950⟩) (rightExpression := ⟨62929⟩)
    (transferEvent := 13479)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13476.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13380.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13480

namespace SemanticResult13484
def owner : Owner := ⟨.program ⟨257⟩, ⟨66008⟩⟩
def rawTerms : List Term := Proof.Events052.exact13484RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13484
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13484.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13481) (rightBinding := 13482)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62930⟩) (rightExpression := ⟨66007⟩)
    (transferEvent := 13483)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13480.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13372.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13484

namespace SemanticResult13488
def owner : Owner := ⟨.program ⟨257⟩, ⟨66009⟩⟩
def rawTerms : List Term := Proof.Events052.exact13488RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13488
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13488.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13485) (rightBinding := 13486)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66008⟩) (rightExpression := ⟨26516⟩)
    (transferEvent := 13487)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13484.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13364.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13488

namespace SemanticResult13492
def owner : Owner := ⟨.program ⟨257⟩, ⟨66010⟩⟩
def rawTerms : List Term := Proof.Events052.exact13492RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13492
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13492.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13489) (rightBinding := 13490)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66009⟩) (rightExpression := ⟨29196⟩)
    (transferEvent := 13491)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13488.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13356.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13492

namespace SemanticResult13496
def owner : Owner := ⟨.program ⟨257⟩, ⟨66011⟩⟩
def rawTerms : List Term := Proof.Events052.exact13496RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13496
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult13496.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13493) (rightBinding := 13494)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66010⟩) (rightExpression := ⟨34853⟩)
    (transferEvent := 13495)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13492.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13348.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13496

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
