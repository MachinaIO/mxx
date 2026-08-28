import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1405
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard074
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard196
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard197
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1356
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1357
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1403
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1404

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult199160
def owner : Owner := ⟨.program ⟨257⟩, ⟨55997⟩⟩
def rawTerms : List Term := Proof.Events777.exact199160RawTerms
def summary : Bound := (.finite 32189789464712143775715074244608)
def resultEvent : Nat := 199160
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199160.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge199157.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult199153.owner)
    (rightOwner := SemanticResult198975.owner)
    (leftResult := 199153) (rightResult := 198975)
    (leftActual := SemanticResult199153.actual selector witness)
    (rightActual := SemanticResult198975.actual selector witness)
    (leftRaw := SemanticResult199153.rawTerms)
    (rightRaw := SemanticResult198975.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32189789464711941702873220382720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 199154) (rightBinding := 199155)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54779⟩) (rightExpression := ⟨55996⟩)
    (coefficientTransfer := 199156) (summaryTransfer := 199159)
    (base := LeftOperatorMerge199157.base)
    (reconstruction := LeftOperatorMerge199157.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult199153.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult198975.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge199157.operationAgreement
  · rfl
  · decide
end SemanticResult199160

namespace SemanticResult199167
def owner : Owner := ⟨.program ⟨257⟩, ⟨52179⟩⟩
def rawTerms : List Term := Proof.Events777.exact199167RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 199167
def producerEvent : Nat := 199166
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199167.actual selector witness
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
end SemanticResult199167

namespace SemanticResult199170
def owner : Owner := ⟨.program ⟨257⟩, ⟨53014⟩⟩
def rawTerms : List Term := Proof.Events778.exact199170RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 199170
def producerEvent : Nat := 199169
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199170.actual selector witness
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
end SemanticResult199170

namespace SemanticResult199177
def owner : Owner := ⟨.program ⟨257⟩, ⟨52021⟩⟩
def rawTerms : List Term := Proof.Events778.exact199177RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 199177
def producerEvent : Nat := 199176
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199177.actual selector witness
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
end SemanticResult199177

namespace SemanticResult199180
def owner : Owner := ⟨.program ⟨257⟩, ⟨52541⟩⟩
def rawTerms : List Term := Proof.Events778.exact199180RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 199180
def producerEvent : Nat := 199179
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199180.actual selector witness
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
end SemanticResult199180

namespace SemanticResult199185
def owner : Owner := ⟨.program ⟨257⟩, ⟨24555⟩⟩
def rawTerms : List Term := Proof.Events778.exact199185RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 199185
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199185.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge199184.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge199184.frameStart)
    (transferEvent := 199183) (owner := owner)
    (leftResult := 9369) (rightResult := 192903)
    (working := LeftOperatorMerge199184.working)
    (reconstruction := LeftOperatorMerge199184.reconstruction)
    (leftReference := .predecessor 0 199181 .coefficient) (rightReference := .predecessor 1 199182 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult9369.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult192903.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge199184.operationAgreement
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
end SemanticResult199185

namespace SemanticResult199190
def owner : Owner := ⟨.program ⟨257⟩, ⟨8842⟩⟩
def rawTerms : List Term := Proof.Events778.exact199190RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 199190
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199190.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge199189.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge199189.frameStart)
    (transferEvent := 199188) (owner := owner)
    (leftResult := 192773) (rightResult := 23593)
    (working := LeftOperatorMerge199189.working)
    (reconstruction := LeftOperatorMerge199189.reconstruction)
    (leftReference := .predecessor 0 199186 .coefficient) (rightReference := .predecessor 1 199187 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult192773.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23593.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge199189.operationAgreement
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
end SemanticResult199190

namespace SemanticResult199194
def owner : Owner := ⟨.program ⟨257⟩, ⟨24556⟩⟩
def rawTerms : List Term := Proof.Events778.exact199194RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 199194
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199194.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 199191) (rightBinding := 199192)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨8842⟩) (rightExpression := ⟨24555⟩)
    (transferEvent := 199193)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult199190.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult199185.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult199194

namespace SemanticResult199200
def owner : Owner := ⟨.program ⟨257⟩, ⟨24557⟩⟩
def rawTerms : List Term := Proof.Events778.exact199200RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 199200
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199200.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 199197) (survivorTransfer := 199198)
    (survivorEvent := 199199) (resultEvent := resultEvent)
    (rightCoefficientProducer := 23584)
    (owner := owner) (leftOwner := SemanticResult199194.owner)
    (rightOwner := SemanticResult23585.owner)
    (leftResult := 199194) (rightResult := 23585)
    (leftBinding := 199195) (rightBinding := 199196)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨24556⟩) (rightExpression := ⟨134⟩)
    (leftActual := SemanticResult199194.actual selector witness)
    (rightActual := SemanticResult23585.actual selector witness)
    (leftRaw := SemanticResult199194.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨134⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound23584.actual selector witness)
    (survivorMagnitude := LeftBound199198.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult199194.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23585.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23584.derived selector witness)
  · exact LeftBound199198.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult199200

namespace SemanticResult199208
def owner : Owner := ⟨.program ⟨257⟩, ⟨50602⟩⟩
def rawTerms : List Term := Proof.Events778.exact199208RawTerms
def summary : Bound := (.finite 8519680)
def resultEvent : Nat := 199208
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199208.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32768
      (.finite ⟨26, by decide⟩)
      (.finite ⟨10, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge199206.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge199206.frameStart)
    (owner := owner) (leftOwner := SemanticResult199200.owner)
    (rightOwner := SemanticResult9372.owner)
    (leftResult := 199200) (rightResult := 9372)
    (leftActual := SemanticResult199200.actual selector witness)
    (rightActual := SemanticResult9372.actual selector witness)
    (leftRaw := SemanticResult199200.rawTerms)
    (rightRaw := SemanticResult9372.rawTerms)
    (working := LeftOperatorMerge199206.working)
    (leftBinding := 199201) (rightBinding := 199202)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨24557⟩) (rightExpression := ⟨50599⟩)
    (coefficientTransfer := 199203) (summaryTransfer := 199205)
    (rightCoefficientProducer := 9371)
    (rightSummaryTransfer := 199204)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨10, by decide⟩)
    (rightRecordedMaximum := 10)
    (rightSummaryMaximum := ⟨10, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32768)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge199206.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority9371.actual selector witness)
    (summaryMagnitude := LeftBound199205.actual selector witness)
    (reconstruction := LeftOperatorMerge199206.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult199200.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9372.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9371.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority9371.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge199206.operationAgreement
  · exact LeftBound199205.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge199206.working summary) := by
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
end SemanticResult199208

namespace SemanticResult199213
def owner : Owner := ⟨.program ⟨257⟩, ⟨50603⟩⟩
def rawTerms : List Term := Proof.Events778.exact199213RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 199213
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199213.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge199212.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge199212.frameStart)
    (transferEvent := 199211) (owner := owner)
    (leftResult := 9372) (rightResult := 192903)
    (working := LeftOperatorMerge199212.working)
    (reconstruction := LeftOperatorMerge199212.reconstruction)
    (leftReference := .predecessor 0 199209 .coefficient) (rightReference := .predecessor 1 199210 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult9372.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult192903.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge199212.operationAgreement
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
end SemanticResult199213

namespace SemanticResult199218
def owner : Owner := ⟨.program ⟨257⟩, ⟨8822⟩⟩
def rawTerms : List Term := Proof.Events778.exact199218RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 199218
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199218.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge199217.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge199217.frameStart)
    (transferEvent := 199216) (owner := owner)
    (leftResult := 192773) (rightResult := 23634)
    (working := LeftOperatorMerge199217.working)
    (reconstruction := LeftOperatorMerge199217.reconstruction)
    (leftReference := .predecessor 0 199214 .coefficient) (rightReference := .predecessor 1 199215 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult192773.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23634.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge199217.operationAgreement
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
end SemanticResult199218

namespace SemanticResult199222
def owner : Owner := ⟨.program ⟨257⟩, ⟨50604⟩⟩
def rawTerms : List Term := Proof.Events778.exact199222RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 199222
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199222.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 199219) (rightBinding := 199220)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨8822⟩) (rightExpression := ⟨50603⟩)
    (transferEvent := 199221)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult199218.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult199213.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult199222

namespace SemanticResult199228
def owner : Owner := ⟨.program ⟨257⟩, ⟨50605⟩⟩
def rawTerms : List Term := Proof.Events778.exact199228RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 199228
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199228.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 199225) (survivorTransfer := 199226)
    (survivorEvent := 199227) (resultEvent := resultEvent)
    (rightCoefficientProducer := 23625)
    (owner := owner) (leftOwner := SemanticResult199222.owner)
    (rightOwner := SemanticResult23626.owner)
    (leftResult := 199222) (rightResult := 23626)
    (leftBinding := 199223) (rightBinding := 199224)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨50604⟩) (rightExpression := ⟨114⟩)
    (leftActual := SemanticResult199222.actual selector witness)
    (rightActual := SemanticResult23626.actual selector witness)
    (leftRaw := SemanticResult199222.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨114⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound23625.actual selector witness)
    (survivorMagnitude := LeftBound199226.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult199222.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23626.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23625.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23625.derived selector witness)
  · exact LeftBound199226.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult199228

namespace SemanticResult199238
def owner : Owner := ⟨.program ⟨257⟩, ⟨50606⟩⟩
def rawTerms : List Term := Proof.Events778.exact199238RawTerms
def summary : Bound := (.finite 279172874240)
def resultEvent : Nat := 199238
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199238.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1310720
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge199234.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge199234.frameStart)
    (owner := owner) (leftOwner := SemanticResult199228.owner)
    (rightOwner := SemanticResult23623.owner)
    (leftResult := 199228) (rightResult := 23623)
    (leftActual := SemanticResult199228.actual selector witness)
    (rightActual := SemanticResult23623.actual selector witness)
    (leftRaw := SemanticResult199228.rawTerms)
    (rightRaw := SemanticResult23623.rawTerms)
    (working := LeftOperatorMerge199234.working)
    (leftBinding := 199229) (rightBinding := 199230)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨50605⟩) (rightExpression := ⟨9581⟩)
    (coefficientTransfer := 199231) (summaryTransfer := 199233)
    (rightCoefficientProducer := 23622)
    (rightSummaryTransfer := 199232)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1310720)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge199234.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound23622.actual selector witness)
    (summaryMagnitude := LeftBound199233.actual selector witness)
    (reconstruction := LeftOperatorMerge199234.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult199228.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23623.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23622.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound23622.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge199234.operationAgreement
  · exact LeftBound199233.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge199234.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 199235 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨50599⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge199234.working
    [{ coefficient := (-1), key := LeftRelationMerge199235.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge199235.frameStart
      LeftRelationMerge199235.owner (.relation 199235) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge199235.deltas
    rows := LeftRelationMerge199235.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge199234.working LeftRelationMerge199235.source
        (relationContext LeftRelationMerge199235.source
          LeftRelationMerge199235.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge199234.working, LeftRelationMerge199235.deltas,
    LeftRelationMerge199235.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 199235)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨50606⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨50599⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge199234.working) (working := relationWorking0)
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
end SemanticResult199238

namespace SemanticResult199244
def owner : Owner := ⟨.program ⟨257⟩, ⟨50607⟩⟩
def rawTerms : List Term := Proof.Events778.exact199244RawTerms
def summary : Bound := (.finite 279181393920)
def resultEvent : Nat := 199244
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult199244.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge199242.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult199238.owner)
    (rightOwner := SemanticResult199208.owner)
    (leftResult := 199238) (rightResult := 199208)
    (leftActual := SemanticResult199238.actual selector witness)
    (rightActual := SemanticResult199208.actual selector witness)
    (leftRaw := SemanticResult199238.rawTerms)
    (rightRaw := SemanticResult199208.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 279172874240)
    (rightMaximum := 8519680) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 199239) (rightBinding := 199240)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨50606⟩) (rightExpression := ⟨50602⟩)
    (coefficientTransfer := 199241) (summaryTransfer := 199243)
    (base := LeftOperatorMerge199242.base)
    (reconstruction := LeftOperatorMerge199242.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult199238.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult199208.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge199242.operationAgreement
  · rfl
  · decide
end SemanticResult199244

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
