import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard988
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard050
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard180
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard181
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard954
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard986
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard987

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult138732
def owner : Owner := ⟨.program ⟨257⟩, ⟨69627⟩⟩
def rawTerms : List Term := Proof.Events541.exact138732RawTerms
def summary : Bound := (.finite 32191361068277642793642192273408)
def resultEvent : Nat := 138732
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138732.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge138729.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult138725.owner)
    (rightOwner := SemanticResult138547.owner)
    (leftResult := 138725) (rightResult := 138547)
    (leftActual := SemanticResult138725.actual selector witness)
    (rightActual := SemanticResult138547.actual selector witness)
    (leftRaw := SemanticResult138725.rawTerms)
    (rightRaw := SemanticResult138547.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32191361068277440720800338411520) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 138726) (rightBinding := 138727)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67940⟩) (rightExpression := ⟨69626⟩)
    (coefficientTransfer := 138728) (summaryTransfer := 138731)
    (base := LeftOperatorMerge138729.base)
    (reconstruction := LeftOperatorMerge138729.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult138725.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult138547.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge138729.operationAgreement
  · rfl
  · decide
end SemanticResult138732

namespace SemanticResult138739
def owner : Owner := ⟨.program ⟨257⟩, ⟨64018⟩⟩
def rawTerms : List Term := Proof.Events541.exact138739RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 138739
def producerEvent : Nat := 138738
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138739.actual selector witness
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
end SemanticResult138739

namespace SemanticResult138742
def owner : Owner := ⟨.program ⟨257⟩, ⟨64655⟩⟩
def rawTerms : List Term := Proof.Events541.exact138742RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 138742
def producerEvent : Nat := 138741
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138742.actual selector witness
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
end SemanticResult138742

namespace SemanticResult138749
def owner : Owner := ⟨.program ⟨257⟩, ⟨63887⟩⟩
def rawTerms : List Term := Proof.Events541.exact138749RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 138749
def producerEvent : Nat := 138748
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138749.actual selector witness
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
end SemanticResult138749

namespace SemanticResult138752
def owner : Owner := ⟨.program ⟨257⟩, ⟨64362⟩⟩
def rawTerms : List Term := Proof.Events542.exact138752RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 138752
def producerEvent : Nat := 138751
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138752.actual selector witness
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
end SemanticResult138752

namespace SemanticResult138757
def owner : Owner := ⟨.program ⟨257⟩, ⟨25407⟩⟩
def rawTerms : List Term := Proof.Events542.exact138757RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 138757
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138757.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge138756.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge138756.frameStart)
    (transferEvent := 138755) (owner := owner)
    (leftResult := 6285) (rightResult := 134403)
    (working := LeftOperatorMerge138756.working)
    (reconstruction := LeftOperatorMerge138756.reconstruction)
    (leftReference := .predecessor 0 138753 .coefficient) (rightReference := .predecessor 1 138754 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult6285.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult134403.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge138756.operationAgreement
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
end SemanticResult138757

namespace SemanticResult138762
def owner : Owner := ⟨.program ⟨257⟩, ⟨7783⟩⟩
def rawTerms : List Term := Proof.Events542.exact138762RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 138762
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138762.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge138761.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge138761.frameStart)
    (transferEvent := 138760) (owner := owner)
    (leftResult := 134273) (rightResult := 21589)
    (working := LeftOperatorMerge138761.working)
    (reconstruction := LeftOperatorMerge138761.reconstruction)
    (leftReference := .predecessor 0 138758 .coefficient) (rightReference := .predecessor 1 138759 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult134273.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21589.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge138761.operationAgreement
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
end SemanticResult138762

namespace SemanticResult138766
def owner : Owner := ⟨.program ⟨257⟩, ⟨25408⟩⟩
def rawTerms : List Term := Proof.Events542.exact138766RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 138766
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138766.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 138763) (rightBinding := 138764)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7783⟩) (rightExpression := ⟨25407⟩)
    (transferEvent := 138765)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult138762.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult138757.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult138766

namespace SemanticResult138772
def owner : Owner := ⟨.program ⟨257⟩, ⟨25409⟩⟩
def rawTerms : List Term := Proof.Events542.exact138772RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 138772
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138772.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 138769) (survivorTransfer := 138770)
    (survivorEvent := 138771) (resultEvent := resultEvent)
    (rightCoefficientProducer := 21580)
    (owner := owner) (leftOwner := SemanticResult138766.owner)
    (rightOwner := SemanticResult21581.owner)
    (leftResult := 138766) (rightResult := 21581)
    (leftBinding := 138767) (rightBinding := 138768)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨25408⟩) (rightExpression := ⟨101⟩)
    (leftActual := SemanticResult138766.actual selector witness)
    (rightActual := SemanticResult21581.actual selector witness)
    (leftRaw := SemanticResult138766.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨101⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound21580.actual selector witness)
    (survivorMagnitude := LeftBound138770.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult138766.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21581.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21580.derived selector witness)
  · exact LeftBound138770.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult138772

namespace SemanticResult138780
def owner : Owner := ⟨.program ⟨257⟩, ⟨62279⟩⟩
def rawTerms : List Term := Proof.Events542.exact138780RawTerms
def summary : Bound := (.finite 18743296)
def resultEvent : Nat := 138780
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138780.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32768
      (.finite ⟨26, by decide⟩)
      (.finite ⟨22, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge138778.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge138778.frameStart)
    (owner := owner) (leftOwner := SemanticResult138772.owner)
    (rightOwner := SemanticResult6288.owner)
    (leftResult := 138772) (rightResult := 6288)
    (leftActual := SemanticResult138772.actual selector witness)
    (rightActual := SemanticResult6288.actual selector witness)
    (leftRaw := SemanticResult138772.rawTerms)
    (rightRaw := SemanticResult6288.rawTerms)
    (working := LeftOperatorMerge138778.working)
    (leftBinding := 138773) (rightBinding := 138774)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨25409⟩) (rightExpression := ⟨62276⟩)
    (coefficientTransfer := 138775) (summaryTransfer := 138777)
    (rightCoefficientProducer := 6287)
    (rightSummaryTransfer := 138776)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨22, by decide⟩)
    (rightRecordedMaximum := 22)
    (rightSummaryMaximum := ⟨22, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32768)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge138778.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority6287.actual selector witness)
    (summaryMagnitude := LeftBound138777.actual selector witness)
    (reconstruction := LeftOperatorMerge138778.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult138772.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6288.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6287.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority6287.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge138778.operationAgreement
  · exact LeftBound138777.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge138778.working summary) := by
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
end SemanticResult138780

namespace SemanticResult138785
def owner : Owner := ⟨.program ⟨257⟩, ⟨62280⟩⟩
def rawTerms : List Term := Proof.Events542.exact138785RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 138785
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138785.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge138784.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge138784.frameStart)
    (transferEvent := 138783) (owner := owner)
    (leftResult := 6288) (rightResult := 134403)
    (working := LeftOperatorMerge138784.working)
    (reconstruction := LeftOperatorMerge138784.reconstruction)
    (leftReference := .predecessor 0 138781 .coefficient) (rightReference := .predecessor 1 138782 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult6288.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult134403.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge138784.operationAgreement
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
end SemanticResult138785

namespace SemanticResult138790
def owner : Owner := ⟨.program ⟨257⟩, ⟨7801⟩⟩
def rawTerms : List Term := Proof.Events542.exact138790RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 138790
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138790.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge138789.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge138789.frameStart)
    (transferEvent := 138788) (owner := owner)
    (leftResult := 134273) (rightResult := 21630)
    (working := LeftOperatorMerge138789.working)
    (reconstruction := LeftOperatorMerge138789.reconstruction)
    (leftReference := .predecessor 0 138786 .coefficient) (rightReference := .predecessor 1 138787 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult134273.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21630.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge138789.operationAgreement
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
end SemanticResult138790

namespace SemanticResult138794
def owner : Owner := ⟨.program ⟨257⟩, ⟨62281⟩⟩
def rawTerms : List Term := Proof.Events542.exact138794RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 138794
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138794.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 138791) (rightBinding := 138792)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7801⟩) (rightExpression := ⟨62280⟩)
    (transferEvent := 138793)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult138790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult138785.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult138794

namespace SemanticResult138800
def owner : Owner := ⟨.program ⟨257⟩, ⟨62282⟩⟩
def rawTerms : List Term := Proof.Events542.exact138800RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 138800
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138800.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 138797) (survivorTransfer := 138798)
    (survivorEvent := 138799) (resultEvent := resultEvent)
    (rightCoefficientProducer := 21621)
    (owner := owner) (leftOwner := SemanticResult138794.owner)
    (rightOwner := SemanticResult21622.owner)
    (leftResult := 138794) (rightResult := 21622)
    (leftBinding := 138795) (rightBinding := 138796)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62281⟩) (rightExpression := ⟨119⟩)
    (leftActual := SemanticResult138794.actual selector witness)
    (rightActual := SemanticResult21622.actual selector witness)
    (leftRaw := SemanticResult138794.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨119⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound21621.actual selector witness)
    (survivorMagnitude := LeftBound138798.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult138794.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21622.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21621.derived selector witness)
  · exact LeftBound138798.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult138800

namespace SemanticResult138810
def owner : Owner := ⟨.program ⟨257⟩, ⟨62283⟩⟩
def rawTerms : List Term := Proof.Events542.exact138810RawTerms
def summary : Bound := (.finite 279172874240)
def resultEvent : Nat := 138810
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138810.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1310720
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge138806.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge138806.frameStart)
    (owner := owner) (leftOwner := SemanticResult138800.owner)
    (rightOwner := SemanticResult21619.owner)
    (leftResult := 138800) (rightResult := 21619)
    (leftActual := SemanticResult138800.actual selector witness)
    (rightActual := SemanticResult21619.actual selector witness)
    (leftRaw := SemanticResult138800.rawTerms)
    (rightRaw := SemanticResult21619.rawTerms)
    (working := LeftOperatorMerge138806.working)
    (leftBinding := 138801) (rightBinding := 138802)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62282⟩) (rightExpression := ⟨9539⟩)
    (coefficientTransfer := 138803) (summaryTransfer := 138805)
    (rightCoefficientProducer := 21618)
    (rightSummaryTransfer := 138804)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1310720)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge138806.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound21618.actual selector witness)
    (summaryMagnitude := LeftBound138805.actual selector witness)
    (reconstruction := LeftOperatorMerge138806.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult138800.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21619.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21618.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound21618.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge138806.operationAgreement
  · exact LeftBound138805.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge138806.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 138807 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge138806.working
    [{ coefficient := (-1), key := LeftRelationMerge138807.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge138807.frameStart
      LeftRelationMerge138807.owner (.relation 138807) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge138807.deltas
    rows := LeftRelationMerge138807.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge138806.working LeftRelationMerge138807.source
        (relationContext LeftRelationMerge138807.source
          LeftRelationMerge138807.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge138806.working, LeftRelationMerge138807.deltas,
    LeftRelationMerge138807.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 138807)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨62283⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge138806.working) (working := relationWorking0)
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
end SemanticResult138810

namespace SemanticResult138816
def owner : Owner := ⟨.program ⟨257⟩, ⟨62284⟩⟩
def rawTerms : List Term := Proof.Events542.exact138816RawTerms
def summary : Bound := (.finite 279191617536)
def resultEvent : Nat := 138816
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult138816.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge138814.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult138810.owner)
    (rightOwner := SemanticResult138780.owner)
    (leftResult := 138810) (rightResult := 138780)
    (leftActual := SemanticResult138810.actual selector witness)
    (rightActual := SemanticResult138780.actual selector witness)
    (leftRaw := SemanticResult138810.rawTerms)
    (rightRaw := SemanticResult138780.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 279172874240)
    (rightMaximum := 18743296) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 138811) (rightBinding := 138812)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62283⟩) (rightExpression := ⟨62279⟩)
    (coefficientTransfer := 138813) (summaryTransfer := 138815)
    (base := LeftOperatorMerge138814.base)
    (reconstruction := LeftOperatorMerge138814.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult138810.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult138780.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge138814.operationAgreement
  · rfl
  · decide
end SemanticResult138816

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
