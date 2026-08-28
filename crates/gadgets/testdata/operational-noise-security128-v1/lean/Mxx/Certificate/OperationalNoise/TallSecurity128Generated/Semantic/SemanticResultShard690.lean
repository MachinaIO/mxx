import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard690
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard032
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard184
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard185
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard652
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard653
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard688
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard689

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult95339
def owner : Owner := ⟨.program ⟨257⟩, ⟨65030⟩⟩
def rawTerms : List Term := Proof.Events372.exact95339RawTerms
def summary : Bound := (.finite 32190771716940580661919523012608)
def resultEvent : Nat := 95339
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95339.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge95336.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult95332.owner)
    (rightOwner := SemanticResult95154.owner)
    (leftResult := 95332) (rightResult := 95154)
    (leftActual := SemanticResult95332.actual selector witness)
    (rightActual := SemanticResult95154.actual selector witness)
    (leftRaw := SemanticResult95332.rawTerms)
    (rightRaw := SemanticResult95154.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32190771716940378589077669150720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 95333) (rightBinding := 95334)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63779⟩) (rightExpression := ⟨65029⟩)
    (coefficientTransfer := 95335) (summaryTransfer := 95338)
    (base := LeftOperatorMerge95336.base)
    (reconstruction := LeftOperatorMerge95336.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95332.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult95154.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge95336.operationAgreement
  · rfl
  · decide
end SemanticResult95339

namespace SemanticResult95346
def owner : Owner := ⟨.program ⟨257⟩, ⟨61146⟩⟩
def rawTerms : List Term := Proof.Events372.exact95346RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95346
def producerEvent : Nat := 95345
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95346.actual selector witness
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
end SemanticResult95346

namespace SemanticResult95349
def owner : Owner := ⟨.program ⟨257⟩, ⟨62047⟩⟩
def rawTerms : List Term := Proof.Events372.exact95349RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95349
def producerEvent : Nat := 95348
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95349.actual selector witness
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
end SemanticResult95349

namespace SemanticResult95356
def owner : Owner := ⟨.program ⟨257⟩, ⟨60979⟩⟩
def rawTerms : List Term := Proof.Events372.exact95356RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95356
def producerEvent : Nat := 95355
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95356.actual selector witness
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
end SemanticResult95356

namespace SemanticResult95359
def owner : Owner := ⟨.program ⟨257⟩, ⟨61514⟩⟩
def rawTerms : List Term := Proof.Events372.exact95359RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95359
def producerEvent : Nat := 95358
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95359.actual selector witness
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
end SemanticResult95359

namespace SemanticResult95364
def owner : Owner := ⟨.program ⟨257⟩, ⟨25311⟩⟩
def rawTerms : List Term := Proof.Events372.exact95364RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95364
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95364.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95363.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge95363.frameStart)
    (transferEvent := 95362) (owner := owner)
    (leftResult := 4064) (rightResult := 90528)
    (working := LeftOperatorMerge95363.working)
    (reconstruction := LeftOperatorMerge95363.reconstruction)
    (leftReference := .predecessor 0 95360 .coefficient) (rightReference := .predecessor 1 95361 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4064.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult90528.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge95363.operationAgreement
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
end SemanticResult95364

namespace SemanticResult95369
def owner : Owner := ⟨.program ⟨257⟩, ⟨9908⟩⟩
def rawTerms : List Term := Proof.Events372.exact95369RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95369
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95369.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95368.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge95368.frameStart)
    (transferEvent := 95367) (owner := owner)
    (leftResult := 90398) (rightResult := 22090)
    (working := LeftOperatorMerge95368.working)
    (reconstruction := LeftOperatorMerge95368.reconstruction)
    (leftReference := .predecessor 0 95365 .coefficient) (rightReference := .predecessor 1 95366 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult90398.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult22090.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge95368.operationAgreement
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
end SemanticResult95369

namespace SemanticResult95373
def owner : Owner := ⟨.program ⟨257⟩, ⟨25312⟩⟩
def rawTerms : List Term := Proof.Events372.exact95373RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95373
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95373.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 95370) (rightBinding := 95371)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9908⟩) (rightExpression := ⟨25311⟩)
    (transferEvent := 95372)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95369.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult95364.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult95373

namespace SemanticResult95379
def owner : Owner := ⟨.program ⟨257⟩, ⟨25313⟩⟩
def rawTerms : List Term := Proof.Events372.exact95379RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 95379
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95379.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 95376) (survivorTransfer := 95377)
    (survivorEvent := 95378) (resultEvent := resultEvent)
    (rightCoefficientProducer := 22081)
    (owner := owner) (leftOwner := SemanticResult95373.owner)
    (rightOwner := SemanticResult22082.owner)
    (leftResult := 95373) (rightResult := 22082)
    (leftBinding := 95374) (rightBinding := 95375)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨25312⟩) (rightExpression := ⟨100⟩)
    (leftActual := SemanticResult95373.actual selector witness)
    (rightActual := SemanticResult22082.actual selector witness)
    (leftRaw := SemanticResult95373.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨100⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound22081.actual selector witness)
    (survivorMagnitude := LeftBound95377.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95373.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult22082.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22081.derived selector witness)
  · exact LeftBound95377.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult95379

namespace SemanticResult95387
def owner : Owner := ⟨.program ⟨257⟩, ⟨59623⟩⟩
def rawTerms : List Term := Proof.Events372.exact95387RawTerms
def summary : Bound := (.finite 15335424)
def resultEvent : Nat := 95387
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95387.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32768
      (.finite ⟨26, by decide⟩)
      (.finite ⟨18, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95385.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge95385.frameStart)
    (owner := owner) (leftOwner := SemanticResult95379.owner)
    (rightOwner := SemanticResult4067.owner)
    (leftResult := 95379) (rightResult := 4067)
    (leftActual := SemanticResult95379.actual selector witness)
    (rightActual := SemanticResult4067.actual selector witness)
    (leftRaw := SemanticResult95379.rawTerms)
    (rightRaw := SemanticResult4067.rawTerms)
    (working := LeftOperatorMerge95385.working)
    (leftBinding := 95380) (rightBinding := 95381)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨25313⟩) (rightExpression := ⟨59620⟩)
    (coefficientTransfer := 95382) (summaryTransfer := 95384)
    (rightCoefficientProducer := 4066)
    (rightSummaryTransfer := 95383)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨18, by decide⟩)
    (rightRecordedMaximum := 18)
    (rightSummaryMaximum := ⟨18, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32768)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge95385.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority4066.actual selector witness)
    (summaryMagnitude := LeftBound95384.actual selector witness)
    (reconstruction := LeftOperatorMerge95385.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95379.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4067.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4066.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority4066.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge95385.operationAgreement
  · exact LeftBound95384.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95385.working summary) := by
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
end SemanticResult95387

namespace SemanticResult95392
def owner : Owner := ⟨.program ⟨257⟩, ⟨59624⟩⟩
def rawTerms : List Term := Proof.Events372.exact95392RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95392
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95392.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95391.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge95391.frameStart)
    (transferEvent := 95390) (owner := owner)
    (leftResult := 4067) (rightResult := 90528)
    (working := LeftOperatorMerge95391.working)
    (reconstruction := LeftOperatorMerge95391.reconstruction)
    (leftReference := .predecessor 0 95388 .coefficient) (rightReference := .predecessor 1 95389 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4067.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult90528.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge95391.operationAgreement
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
end SemanticResult95392

namespace SemanticResult95397
def owner : Owner := ⟨.program ⟨257⟩, ⟨9925⟩⟩
def rawTerms : List Term := Proof.Events372.exact95397RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95397
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95397.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95396.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge95396.frameStart)
    (transferEvent := 95395) (owner := owner)
    (leftResult := 90398) (rightResult := 22131)
    (working := LeftOperatorMerge95396.working)
    (reconstruction := LeftOperatorMerge95396.reconstruction)
    (leftReference := .predecessor 0 95393 .coefficient) (rightReference := .predecessor 1 95394 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult90398.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult22131.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge95396.operationAgreement
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
end SemanticResult95397

namespace SemanticResult95401
def owner : Owner := ⟨.program ⟨257⟩, ⟨59625⟩⟩
def rawTerms : List Term := Proof.Events372.exact95401RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95401
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95401.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 95398) (rightBinding := 95399)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9925⟩) (rightExpression := ⟨59624⟩)
    (transferEvent := 95400)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95397.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult95392.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult95401

namespace SemanticResult95407
def owner : Owner := ⟨.program ⟨257⟩, ⟨59626⟩⟩
def rawTerms : List Term := Proof.Events372.exact95407RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 95407
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95407.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 95404) (survivorTransfer := 95405)
    (survivorEvent := 95406) (resultEvent := resultEvent)
    (rightCoefficientProducer := 22122)
    (owner := owner) (leftOwner := SemanticResult95401.owner)
    (rightOwner := SemanticResult22123.owner)
    (leftResult := 95401) (rightResult := 22123)
    (leftBinding := 95402) (rightBinding := 95403)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59625⟩) (rightExpression := ⟨117⟩)
    (leftActual := SemanticResult95401.actual selector witness)
    (rightActual := SemanticResult22123.actual selector witness)
    (leftRaw := SemanticResult95401.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨117⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound22122.actual selector witness)
    (survivorMagnitude := LeftBound95405.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95401.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult22123.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22122.derived selector witness)
  · exact LeftBound95405.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult95407

namespace SemanticResult95417
def owner : Owner := ⟨.program ⟨257⟩, ⟨59627⟩⟩
def rawTerms : List Term := Proof.Events372.exact95417RawTerms
def summary : Bound := (.finite 279172874240)
def resultEvent : Nat := 95417
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95417.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1310720
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95413.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge95413.frameStart)
    (owner := owner) (leftOwner := SemanticResult95407.owner)
    (rightOwner := SemanticResult22120.owner)
    (leftResult := 95407) (rightResult := 22120)
    (leftActual := SemanticResult95407.actual selector witness)
    (rightActual := SemanticResult22120.actual selector witness)
    (leftRaw := SemanticResult95407.rawTerms)
    (rightRaw := SemanticResult22120.rawTerms)
    (working := LeftOperatorMerge95413.working)
    (leftBinding := 95408) (rightBinding := 95409)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59626⟩) (rightExpression := ⟨9536⟩)
    (coefficientTransfer := 95410) (summaryTransfer := 95412)
    (rightCoefficientProducer := 22119)
    (rightSummaryTransfer := 95411)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1310720)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge95413.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound22119.actual selector witness)
    (summaryMagnitude := LeftBound95412.actual selector witness)
    (reconstruction := LeftOperatorMerge95413.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95407.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult22120.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22119.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound22119.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge95413.operationAgreement
  · exact LeftBound95412.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95413.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 95414 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge95413.working
    [{ coefficient := (-1), key := LeftRelationMerge95414.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge95414.frameStart
      LeftRelationMerge95414.owner (.relation 95414) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge95414.deltas
    rows := LeftRelationMerge95414.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge95413.working LeftRelationMerge95414.source
        (relationContext LeftRelationMerge95414.source
          LeftRelationMerge95414.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge95413.working, LeftRelationMerge95414.deltas,
    LeftRelationMerge95414.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 95414)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨59627⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge95413.working) (working := relationWorking0)
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
end SemanticResult95417

namespace SemanticResult95423
def owner : Owner := ⟨.program ⟨257⟩, ⟨59628⟩⟩
def rawTerms : List Term := Proof.Events372.exact95423RawTerms
def summary : Bound := (.finite 279188209664)
def resultEvent : Nat := 95423
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult95423.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge95421.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult95417.owner)
    (rightOwner := SemanticResult95387.owner)
    (leftResult := 95417) (rightResult := 95387)
    (leftActual := SemanticResult95417.actual selector witness)
    (rightActual := SemanticResult95387.actual selector witness)
    (leftRaw := SemanticResult95417.rawTerms)
    (rightRaw := SemanticResult95387.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 279172874240)
    (rightMaximum := 15335424) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 95418) (rightBinding := 95419)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59627⟩) (rightExpression := ⟨59623⟩)
    (coefficientTransfer := 95420) (summaryTransfer := 95422)
    (base := LeftOperatorMerge95421.base)
    (reconstruction := LeftOperatorMerge95421.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95417.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult95387.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge95421.operationAgreement
  · rfl
  · decide
end SemanticResult95423

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
