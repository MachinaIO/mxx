import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1584
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard085
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard172
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1557
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1558
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1582
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1583

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult225518
def owner : Owner := ⟨.program ⟨257⟩, ⟨30947⟩⟩
def rawTerms : List Term := Proof.Events880.exact225518RawTerms
def summary : Bound := (.finite 32192146870060392302605751287808)
def resultEvent : Nat := 225518
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225518.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge225515.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult225511.owner)
    (rightOwner := SemanticResult225333.owner)
    (leftResult := 225511) (rightResult := 225333)
    (leftActual := SemanticResult225511.actual selector witness)
    (rightActual := SemanticResult225333.actual selector witness)
    (leftRaw := SemanticResult225511.rawTerms)
    (rightRaw := SemanticResult225333.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32192146870060190229763897425920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 225512) (rightBinding := 225513)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29819⟩) (rightExpression := ⟨30946⟩)
    (coefficientTransfer := 225514) (summaryTransfer := 225517)
    (base := LeftOperatorMerge225515.base)
    (reconstruction := LeftOperatorMerge225515.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult225511.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult225333.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge225515.operationAgreement
  · rfl
  · decide
end SemanticResult225518

namespace SemanticResult225525
def owner : Owner := ⟨.program ⟨257⟩, ⟨27552⟩⟩
def rawTerms : List Term := Proof.Events880.exact225525RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 225525
def producerEvent : Nat := 225524
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225525.actual selector witness
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
end SemanticResult225525

namespace SemanticResult225528
def owner : Owner := ⟨.program ⟨257⟩, ⟨28264⟩⟩
def rawTerms : List Term := Proof.Events880.exact225528RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 225528
def producerEvent : Nat := 225527
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225528.actual selector witness
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
end SemanticResult225528

namespace SemanticResult225535
def owner : Owner := ⟨.program ⟨257⟩, ⟨27403⟩⟩
def rawTerms : List Term := Proof.Events880.exact225535RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 225535
def producerEvent : Nat := 225534
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225535.actual selector witness
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
end SemanticResult225535

namespace SemanticResult225538
def owner : Owner := ⟨.program ⟨257⟩, ⟨27908⟩⟩
def rawTerms : List Term := Proof.Events881.exact225538RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 225538
def producerEvent : Nat := 225537
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225538.actual selector witness
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
end SemanticResult225538

namespace SemanticResult225543
def owner : Owner := ⟨.program ⟨257⟩, ⟨26073⟩⟩
def rawTerms : List Term := Proof.Events881.exact225543RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 225543
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225543.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge225542.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge225542.frameStart)
    (transferEvent := 225541) (owner := owner)
    (leftResult := 10727) (rightResult := 222153)
    (working := LeftOperatorMerge225542.working)
    (reconstruction := LeftOperatorMerge225542.reconstruction)
    (leftReference := .predecessor 0 225539 .coefficient) (rightReference := .predecessor 1 225540 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult10727.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult222153.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge225542.operationAgreement
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
end SemanticResult225543

namespace SemanticResult225548
def owner : Owner := ⟨.program ⟨257⟩, ⟨8470⟩⟩
def rawTerms : List Term := Proof.Events881.exact225548RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 225548
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225548.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge225547.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge225547.frameStart)
    (transferEvent := 225546) (owner := owner)
    (leftResult := 222023) (rightResult := 20587)
    (working := LeftOperatorMerge225547.working)
    (reconstruction := LeftOperatorMerge225547.reconstruction)
    (leftReference := .predecessor 0 225544 .coefficient) (rightReference := .predecessor 1 225545 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult222023.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20587.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge225547.operationAgreement
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
end SemanticResult225548

namespace SemanticResult225552
def owner : Owner := ⟨.program ⟨257⟩, ⟨26074⟩⟩
def rawTerms : List Term := Proof.Events881.exact225552RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 225552
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225552.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 225549) (rightBinding := 225550)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨8470⟩) (rightExpression := ⟨26073⟩)
    (transferEvent := 225551)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult225548.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult225543.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult225552

namespace SemanticResult225558
def owner : Owner := ⟨.program ⟨257⟩, ⟨26075⟩⟩
def rawTerms : List Term := Proof.Events881.exact225558RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 225558
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225558.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 225555) (survivorTransfer := 225556)
    (survivorEvent := 225557) (resultEvent := resultEvent)
    (rightCoefficientProducer := 20578)
    (owner := owner) (leftOwner := SemanticResult225552.owner)
    (rightOwner := SemanticResult20579.owner)
    (leftResult := 225552) (rightResult := 20579)
    (leftBinding := 225553) (rightBinding := 225554)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26074⟩) (rightExpression := ⟨104⟩)
    (leftActual := SemanticResult225552.actual selector witness)
    (rightActual := SemanticResult20579.actual selector witness)
    (leftRaw := SemanticResult225552.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨104⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound20578.actual selector witness)
    (survivorMagnitude := LeftBound225556.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult225552.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20579.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20578.derived selector witness)
  · exact LeftBound225556.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult225558

namespace SemanticResult225566
def owner : Owner := ⟨.program ⟨257⟩, ⟨26076⟩⟩
def rawTerms : List Term := Proof.Events881.exact225566RawTerms
def summary : Bound := (.finite 25559040)
def resultEvent : Nat := 225566
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225566.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32768
      (.finite ⟨26, by decide⟩)
      (.finite ⟨30, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge225564.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge225564.frameStart)
    (owner := owner) (leftOwner := SemanticResult225558.owner)
    (rightOwner := SemanticResult10730.owner)
    (leftResult := 225558) (rightResult := 10730)
    (leftActual := SemanticResult225558.actual selector witness)
    (rightActual := SemanticResult10730.actual selector witness)
    (leftRaw := SemanticResult225558.rawTerms)
    (rightRaw := SemanticResult10730.rawTerms)
    (working := LeftOperatorMerge225564.working)
    (leftBinding := 225559) (rightBinding := 225560)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26075⟩) (rightExpression := ⟨12966⟩)
    (coefficientTransfer := 225561) (summaryTransfer := 225563)
    (rightCoefficientProducer := 10729)
    (rightSummaryTransfer := 225562)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨30, by decide⟩)
    (rightRecordedMaximum := 30)
    (rightSummaryMaximum := ⟨30, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32768)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge225564.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority10729.actual selector witness)
    (summaryMagnitude := LeftBound225563.actual selector witness)
    (reconstruction := LeftOperatorMerge225564.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult225558.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10730.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10729.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority10729.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge225564.operationAgreement
  · exact LeftBound225563.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge225564.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult225566

namespace SemanticResult225571
def owner : Owner := ⟨.program ⟨257⟩, ⟨12967⟩⟩
def rawTerms : List Term := Proof.Events881.exact225571RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 225571
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225571.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge225570.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge225570.frameStart)
    (transferEvent := 225569) (owner := owner)
    (leftResult := 10730) (rightResult := 222153)
    (working := LeftOperatorMerge225570.working)
    (reconstruction := LeftOperatorMerge225570.reconstruction)
    (leftReference := .predecessor 0 225567 .coefficient) (rightReference := .predecessor 1 225568 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult10730.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult222153.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge225570.operationAgreement
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
end SemanticResult225571

namespace SemanticResult225576
def owner : Owner := ⟨.program ⟨257⟩, ⟨8487⟩⟩
def rawTerms : List Term := Proof.Events881.exact225576RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 225576
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225576.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge225575.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge225575.frameStart)
    (transferEvent := 225574) (owner := owner)
    (leftResult := 222023) (rightResult := 20628)
    (working := LeftOperatorMerge225575.working)
    (reconstruction := LeftOperatorMerge225575.reconstruction)
    (leftReference := .predecessor 0 225572 .coefficient) (rightReference := .predecessor 1 225573 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult222023.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20628.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge225575.operationAgreement
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
end SemanticResult225576

namespace SemanticResult225580
def owner : Owner := ⟨.program ⟨257⟩, ⟨12968⟩⟩
def rawTerms : List Term := Proof.Events881.exact225580RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 225580
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225580.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 225577) (rightBinding := 225578)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨8487⟩) (rightExpression := ⟨12967⟩)
    (transferEvent := 225579)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult225576.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult225571.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult225580

namespace SemanticResult225586
def owner : Owner := ⟨.program ⟨257⟩, ⟨12969⟩⟩
def rawTerms : List Term := Proof.Events881.exact225586RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 225586
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225586.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 225583) (survivorTransfer := 225584)
    (survivorEvent := 225585) (resultEvent := resultEvent)
    (rightCoefficientProducer := 20619)
    (owner := owner) (leftOwner := SemanticResult225580.owner)
    (rightOwner := SemanticResult20620.owner)
    (leftResult := 225580) (rightResult := 20620)
    (leftBinding := 225581) (rightBinding := 225582)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12968⟩) (rightExpression := ⟨121⟩)
    (leftActual := SemanticResult225580.actual selector witness)
    (rightActual := SemanticResult20620.actual selector witness)
    (leftRaw := SemanticResult225580.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨121⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound20619.actual selector witness)
    (survivorMagnitude := LeftBound225584.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult225580.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20620.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20619.derived selector witness)
  · exact LeftBound225584.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult225586

namespace SemanticResult225596
def owner : Owner := ⟨.program ⟨257⟩, ⟨12970⟩⟩
def rawTerms : List Term := Proof.Events881.exact225596RawTerms
def summary : Bound := (.finite 279172874240)
def resultEvent : Nat := 225596
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225596.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1310720
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge225592.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge225592.frameStart)
    (owner := owner) (leftOwner := SemanticResult225586.owner)
    (rightOwner := SemanticResult20617.owner)
    (leftResult := 225586) (rightResult := 20617)
    (leftActual := SemanticResult225586.actual selector witness)
    (rightActual := SemanticResult20617.actual selector witness)
    (leftRaw := SemanticResult225586.rawTerms)
    (rightRaw := SemanticResult20617.rawTerms)
    (working := LeftOperatorMerge225592.working)
    (leftBinding := 225587) (rightBinding := 225588)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12969⟩) (rightExpression := ⟨9545⟩)
    (coefficientTransfer := 225589) (summaryTransfer := 225591)
    (rightCoefficientProducer := 20616)
    (rightSummaryTransfer := 225590)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1310720)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge225592.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound20616.actual selector witness)
    (summaryMagnitude := LeftBound225591.actual selector witness)
    (reconstruction := LeftOperatorMerge225592.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult225586.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20617.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20616.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound20616.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge225592.operationAgreement
  · exact LeftBound225591.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge225592.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 225593 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge225592.working
    [{ coefficient := (-1), key := LeftRelationMerge225593.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge225593.frameStart
      LeftRelationMerge225593.owner (.relation 225593) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge225593.deltas
    rows := LeftRelationMerge225593.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge225592.working LeftRelationMerge225593.source
        (relationContext LeftRelationMerge225593.source
          LeftRelationMerge225593.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge225592.working, LeftRelationMerge225593.deltas,
    LeftRelationMerge225593.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 225593)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨12970⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12966⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge225592.working) (working := relationWorking0)
    (reconstruction := relationReconstruction0)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt0 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact mergeClaim selector selectorLower selectorUpper witness
  · exact relationAgreement0
  · decide +kernel
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (relationClaim0 selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult225596

namespace SemanticResult225602
def owner : Owner := ⟨.program ⟨257⟩, ⟨26077⟩⟩
def rawTerms : List Term := Proof.Events881.exact225602RawTerms
def summary : Bound := (.finite 279198433280)
def resultEvent : Nat := 225602
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult225602.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge225600.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult225596.owner)
    (rightOwner := SemanticResult225566.owner)
    (leftResult := 225596) (rightResult := 225566)
    (leftActual := SemanticResult225596.actual selector witness)
    (rightActual := SemanticResult225566.actual selector witness)
    (leftRaw := SemanticResult225596.rawTerms)
    (rightRaw := SemanticResult225566.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 279172874240)
    (rightMaximum := 25559040) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 225597) (rightBinding := 225598)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12970⟩) (rightExpression := ⟨26076⟩)
    (coefficientTransfer := 225599) (summaryTransfer := 225601)
    (base := LeftOperatorMerge225600.base)
    (reconstruction := LeftOperatorMerge225600.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult225596.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult225566.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge225600.operationAgreement
  · rfl
  · decide
end SemanticResult225602

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
