import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard084
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard078
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard081
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard082
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard083

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult10492
def owner : Owner := ⟨.program ⟨257⟩, ⟨66590⟩⟩
def rawTerms : List Term := Proof.Events040.exact10492RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10492
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10492.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10489) (rightBinding := 10490)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63087⟩) (rightExpression := ⟨66589⟩)
    (transferEvent := 10491)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10488.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10380.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10492

namespace SemanticResult10496
def owner : Owner := ⟨.program ⟨257⟩, ⟨66591⟩⟩
def rawTerms : List Term := Proof.Events041.exact10496RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10496
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10496.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10493) (rightBinding := 10494)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66590⟩) (rightExpression := ⟨26623⟩)
    (transferEvent := 10495)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10492.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10372.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10496

namespace SemanticResult10500
def owner : Owner := ⟨.program ⟨257⟩, ⟨66592⟩⟩
def rawTerms : List Term := Proof.Events041.exact10500RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10500
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10500.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10497) (rightBinding := 10498)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66591⟩) (rightExpression := ⟨29303⟩)
    (transferEvent := 10499)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10496.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10364.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10500

namespace SemanticResult10504
def owner : Owner := ⟨.program ⟨257⟩, ⟨66593⟩⟩
def rawTerms : List Term := Proof.Events041.exact10504RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10504
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10504.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10501) (rightBinding := 10502)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66592⟩) (rightExpression := ⟨34960⟩)
    (transferEvent := 10503)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10500.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10356.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10504

namespace SemanticResult10508
def owner : Owner := ⟨.program ⟨257⟩, ⟨66594⟩⟩
def rawTerms : List Term := Proof.Events041.exact10508RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10508
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10508.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10505) (rightBinding := 10506)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66593⟩) (rightExpression := ⟨37640⟩)
    (transferEvent := 10507)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10504.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10348.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10508

namespace SemanticResult10512
def owner : Owner := ⟨.program ⟨257⟩, ⟨66595⟩⟩
def rawTerms : List Term := Proof.Events041.exact10512RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10512
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10512.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10509) (rightBinding := 10510)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66594⟩) (rightExpression := ⟨40323⟩)
    (transferEvent := 10511)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10508.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10340.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10512

namespace SemanticResult10516
def owner : Owner := ⟨.program ⟨257⟩, ⟨66596⟩⟩
def rawTerms : List Term := Proof.Events041.exact10516RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10516
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10516.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10513) (rightBinding := 10514)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66595⟩) (rightExpression := ⟨43003⟩)
    (transferEvent := 10515)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10512.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10332.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10516

namespace SemanticResult10520
def owner : Owner := ⟨.program ⟨257⟩, ⟨66597⟩⟩
def rawTerms : List Term := Proof.Events041.exact10520RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10520
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10520.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10517) (rightBinding := 10518)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66596⟩) (rightExpression := ⟨45680⟩)
    (transferEvent := 10519)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10516.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10324.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10520

namespace SemanticResult10524
def owner : Owner := ⟨.program ⟨257⟩, ⟨66598⟩⟩
def rawTerms : List Term := Proof.Events041.exact10524RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10524
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10524.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10521) (rightBinding := 10522)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66597⟩) (rightExpression := ⟨48360⟩)
    (transferEvent := 10523)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10520.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10316.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10524

namespace SemanticResult10528
def owner : Owner := ⟨.program ⟨257⟩, ⟨67460⟩⟩
def rawTerms : List Term := Proof.Events041.exact10528RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10528
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10528.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 10525) (rightBinding := 10526)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66598⟩) (rightExpression := ⟨67458⟩)
    (transferEvent := 10527)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult10524.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10308.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult10528

namespace SemanticResult10551
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def rawTerms : List Term := Proof.Events041.exact10551RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10551
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10551.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge10532.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge10532.frameStart)
    (transferEvent := 10531) (owner := owner)
    (leftResult := 10528) (rightResult := 9805)
    (working := LeftOperatorMerge10532.working)
    (reconstruction := LeftOperatorMerge10532.reconstruction)
    (leftReference := .predecessor 0 10529 .coefficient) (rightReference := .predecessor 1 10530 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult10528.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9805.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge10532.operationAgreement
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
end SemanticResult10551

namespace SemanticResult10553
def owner : Owner := ⟨.program ⟨257⟩, ⟨6748⟩⟩
def rawTerms : List Term := Proof.Events041.exact10553RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10553
def producerEvent : Nat := 10552
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10553.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 0, .finite 926165433236564883034880022152199146955988691593218821555396268135235105694352221602317227330655430115567635851440095727930594311256248010260869261711131158601001588347, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult10553

namespace SemanticResult10566
def owner : Owner := ⟨.program ⟨257⟩, ⟨47810⟩⟩
def rawTerms : List Term := Proof.Events041.exact10566RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10566
def producerEvent : Nat := 10565
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10566.actual selector witness
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
end SemanticResult10566

namespace SemanticResult10569
def owner : Owner := ⟨.program ⟨257⟩, ⟨15066⟩⟩
def rawTerms : List Term := Proof.Events041.exact10569RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10569
def producerEvent : Nat := 10568
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10569.actual selector witness
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
end SemanticResult10569

namespace SemanticResult10589
def owner : Owner := ⟨.program ⟨257⟩, ⟨45130⟩⟩
def rawTerms : List Term := Proof.Events041.exact10589RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10589
def producerEvent : Nat := 10588
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10589.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 0, .finite 58, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult10589

namespace SemanticResult10592
def owner : Owner := ⟨.program ⟨257⟩, ⟨14766⟩⟩
def rawTerms : List Term := Proof.Events041.exact10592RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10592
def producerEvent : Nat := 10591
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult10592.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 0, .finite 58, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult10592

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
