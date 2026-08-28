import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard518
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard453
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard485
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard488
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard492
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard496
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard499
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard503
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard507
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard511
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard514
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard517

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult69913
def owner : Owner := ⟨.program ⟨257⟩, ⟨16147⟩⟩
def rawTerms : List Term := Proof.Events273.exact69913RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69913
def producerEvent : Nat := 69912
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69913.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 69828, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult69913

namespace SemanticResult69918
def owner : Owner := ⟨.program ⟨257⟩, ⟨16148⟩⟩
def rawTerms : List Term := Proof.Events273.exact69918RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69918
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69918.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge69917.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge69917.frameStart)
    (transferEvent := 69916) (owner := owner)
    (leftResult := 69890) (rightResult := 69913)
    (working := LeftOperatorMerge69917.working)
    (reconstruction := LeftOperatorMerge69917.reconstruction)
    (leftReference := .predecessor 0 69914 .coefficient) (rightReference := .predecessor 1 69915 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult69890.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult69913.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge69917.operationAgreement
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
end SemanticResult69918

namespace SemanticResult69921
def owner : Owner := ⟨.program ⟨257⟩, ⟨7198⟩⟩
def rawTerms : List Term := Proof.Events273.exact69921RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69921
def producerEvent : Nat := 69920
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69921.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 69828, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult69921

namespace SemanticResult69925
def owner : Owner := ⟨.program ⟨257⟩, ⟨16149⟩⟩
def rawTerms : List Term := Proof.Events273.exact69925RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69925
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69925.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 69922) (rightBinding := 69923)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7198⟩) (rightExpression := ⟨16148⟩)
    (transferEvent := 69924)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69921.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult69918.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult69925

namespace SemanticResult69929
def owner : Owner := ⟨.program ⟨257⟩, ⟨17961⟩⟩
def rawTerms : List Term := Proof.Events273.exact69929RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69929
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69929.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 69926) (rightBinding := 69927)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16149⟩) (rightExpression := ⟨17958⟩)
    (transferEvent := 69928)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69925.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult69910.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult69929

namespace SemanticResult69938
def owner : Owner := ⟨.program ⟨257⟩, ⟨16739⟩⟩
def rawTerms : List Term := Proof.Events273.exact69938RawTerms
def summary : Bound := (.finite 202072841853861888)
def resultEvent : Nat := 69938
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69938.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1376256
      (.finite ⟨26, by decide⟩)
      (.finite ⟨5647228698, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge69773.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge69773.frameStart)
    (owner := owner) (leftOwner := SemanticResult61370.owner)
    (rightOwner := SemanticResult69767.owner)
    (leftResult := 61370) (rightResult := 69767)
    (leftActual := SemanticResult61370.actual selector witness)
    (rightActual := SemanticResult69767.actual selector witness)
    (leftRaw := SemanticResult61370.rawTerms)
    (rightRaw := SemanticResult69767.rawTerms)
    (working := LeftOperatorMerge69773.working)
    (leftBinding := 69768) (rightBinding := 69769)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10792⟩) (rightExpression := ⟨16738⟩)
    (coefficientTransfer := 69770) (summaryTransfer := 69772)
    (rightCoefficientProducer := 69766)
    (rightSummaryTransfer := 69771)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨5647228698, by decide⟩)
    (rightRecordedMaximum := 5647228698)
    (rightSummaryMaximum := ⟨5647228698, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1376256)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge69773.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound69766.actual selector witness)
    (summaryMagnitude := LeftBound69772.actual selector witness)
    (reconstruction := LeftOperatorMerge69773.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult61370.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult69767.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69766.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound69766.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge69773.operationAgreement
  · exact LeftBound69772.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge69773.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 69933 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17064⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17064⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16147⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge69773.working
    [{ coefficient := (1), key := LeftRelationMerge69933.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge69933.frameStart
      LeftRelationMerge69933.owner (.relation 69933) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge69933.deltas
    rows := LeftRelationMerge69933.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge69773.working LeftRelationMerge69933.source
        (relationContext LeftRelationMerge69933.source
          LeftRelationMerge69933.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge69773.working, LeftRelationMerge69933.deltas,
    LeftRelationMerge69933.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 69933)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨16739⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16736⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16736⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge69773.working) (working := relationWorking0)
    (reconstruction := relationReconstruction0)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt0 selector selectorLower selectorUpper
  · rfl
  · rfl
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
end SemanticResult69938

namespace SemanticResult69945
def owner : Owner := ⟨.program ⟨257⟩, ⟨17960⟩⟩
def rawTerms : List Term := Proof.Events273.exact69945RawTerms
def summary : Bound := (.finite 32188807212483706889510625476608)
def resultEvent : Nat := 69945
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69945.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge69942.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult69938.owner)
    (rightOwner := SemanticResult69760.owner)
    (leftResult := 69938) (rightResult := 69760)
    (leftActual := SemanticResult69938.actual selector witness)
    (rightActual := SemanticResult69760.actual selector witness)
    (leftRaw := SemanticResult69938.rawTerms)
    (rightRaw := SemanticResult69760.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32188807212483504816668771614720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 69939) (rightBinding := 69940)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16739⟩) (rightExpression := ⟨17959⟩)
    (coefficientTransfer := 69941) (summaryTransfer := 69944)
    (base := LeftOperatorMerge69942.base)
    (reconstruction := LeftOperatorMerge69942.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69938.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult69760.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge69942.operationAgreement
  · rfl
  · decide
end SemanticResult69945

namespace SemanticResult69950
def owner : Owner := ⟨.program ⟨257⟩, ⟨20873⟩⟩
def rawTerms : List Term := Proof.Events273.exact69950RawTerms
def summary : Bound := (.finite 64377712650190257467641695830016)
def resultEvent : Nat := 69950
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69950.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult69945.owner)
    (rightOwner := SemanticResult69463.owner)
    (leftResult := 69945) (rightResult := 69463)
    (leftActual := SemanticResult69945.actual selector witness)
    (rightActual := SemanticResult69463.actual selector witness)
    (leftRaw := SemanticResult69945.rawTerms)
    (rightRaw := SemanticResult69463.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 32188807212483706889510625476608)
    (rightMaximum := 32188905437706550578131070353408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 69946) (rightBinding := 69947)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17960⟩) (rightExpression := ⟨20872⟩)
    (transferEvent := 69948) (summaryTransferEvent := 69949)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69945.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult69463.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult69950

namespace SemanticResult69955
def owner : Owner := ⟨.program ⟨257⟩, ⟨24093⟩⟩
def rawTerms : List Term := Proof.Events273.exact69955RawTerms
def summary : Bound := (.finite 96566716313119651734393211060224)
def resultEvent : Nat := 69955
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69955.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult69950.owner)
    (rightOwner := SemanticResult68981.owner)
    (leftResult := 69950) (rightResult := 68981)
    (leftActual := SemanticResult69950.actual selector witness)
    (rightActual := SemanticResult68981.actual selector witness)
    (leftRaw := SemanticResult69950.rawTerms)
    (rightRaw := SemanticResult68981.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 64377712650190257467641695830016)
    (rightMaximum := 32189003662929394266751515230208) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 69951) (rightBinding := 69952)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20873⟩) (rightExpression := ⟨24092⟩)
    (transferEvent := 69953) (summaryTransferEvent := 69954)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69950.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult68981.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult69955

namespace SemanticResult69960
def owner : Owner := ⟨.program ⟨257⟩, ⟨34113⟩⟩
def rawTerms : List Term := Proof.Events273.exact69960RawTerms
def summary : Bound := (.finite 128755916426494733378385616044032)
def resultEvent : Nat := 69960
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69960.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult69955.owner)
    (rightOwner := SemanticResult68499.owner)
    (leftResult := 69955) (rightResult := 68499)
    (leftActual := SemanticResult69955.actual selector witness)
    (rightActual := SemanticResult68499.actual selector witness)
    (leftRaw := SemanticResult69955.rawTerms)
    (rightRaw := SemanticResult68499.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 96566716313119651734393211060224)
    (rightMaximum := 32189200113375081643992404983808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 69956) (rightBinding := 69957)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨24093⟩) (rightExpression := ⟨34112⟩)
    (transferEvent := 69958) (summaryTransferEvent := 69959)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69955.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult68499.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult69960

namespace SemanticResult69965
def owner : Owner := ⟨.program ⟨257⟩, ⟨53173⟩⟩
def rawTerms : List Term := Proof.Events273.exact69965RawTerms
def summary : Bound := (.finite 160945509440761189776859800535040)
def resultEvent : Nat := 69965
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69965.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult69960.owner)
    (rightOwner := SemanticResult68017.owner)
    (leftResult := 69960) (rightResult := 68017)
    (leftActual := SemanticResult69960.actual selector witness)
    (rightActual := SemanticResult68017.actual selector witness)
    (leftRaw := SemanticResult69960.rawTerms)
    (rightRaw := SemanticResult68017.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 128755916426494733378385616044032)
    (rightMaximum := 32189593014266456398474184491008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 69961) (rightBinding := 69962)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨34113⟩) (rightExpression := ⟨53172⟩)
    (transferEvent := 69963) (summaryTransferEvent := 69964)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69960.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult68017.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult69965

namespace SemanticResult69970
def owner : Owner := ⟨.program ⟨257⟩, ⟨56153⟩⟩
def rawTerms : List Term := Proof.Events273.exact69970RawTerms
def summary : Bound := (.finite 193135298905473333552574874779648)
def resultEvent : Nat := 69970
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69970.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult69965.owner)
    (rightOwner := SemanticResult67535.owner)
    (leftResult := 69965) (rightResult := 67535)
    (leftActual := SemanticResult69965.actual selector witness)
    (rightActual := SemanticResult67535.actual selector witness)
    (leftRaw := SemanticResult69965.rawTerms)
    (rightRaw := SemanticResult67535.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 160945509440761189776859800535040)
    (rightMaximum := 32189789464712143775715074244608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 69966) (rightBinding := 69967)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53173⟩) (rightExpression := ⟨56152⟩)
    (transferEvent := 69968) (summaryTransferEvent := 69969)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69965.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult67535.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult69970

namespace SemanticResult69975
def owner : Owner := ⟨.program ⟨257⟩, ⟨59133⟩⟩
def rawTerms : List Term := Proof.Events273.exact69975RawTerms
def summary : Bound := (.finite 225325481271076852082771728531456)
def resultEvent : Nat := 69975
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69975.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult69970.owner)
    (rightOwner := SemanticResult67053.owner)
    (leftResult := 69970) (rightResult := 67053)
    (leftActual := SemanticResult69970.actual selector witness)
    (rightActual := SemanticResult67053.actual selector witness)
    (leftRaw := SemanticResult69970.rawTerms)
    (rightRaw := SemanticResult67053.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 193135298905473333552574874779648)
    (rightMaximum := 32190182365603518530196853751808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 69971) (rightBinding := 69972)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56153⟩) (rightExpression := ⟨59132⟩)
    (transferEvent := 69973) (summaryTransferEvent := 69974)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69970.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult67053.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult69975

namespace SemanticResult69980
def owner : Owner := ⟨.program ⟨257⟩, ⟨62113⟩⟩
def rawTerms : List Term := Proof.Events273.exact69980RawTerms
def summary : Bound := (.finite 257515860087126057990209472036864)
def resultEvent : Nat := 69980
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69980.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult69975.owner)
    (rightOwner := SemanticResult66571.owner)
    (leftResult := 69975) (rightResult := 66571)
    (leftActual := SemanticResult69975.actual selector witness)
    (rightActual := SemanticResult66571.actual selector witness)
    (leftRaw := SemanticResult69975.rawTerms)
    (rightRaw := SemanticResult66571.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 225325481271076852082771728531456)
    (rightMaximum := 32190378816049205907437743505408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 69976) (rightBinding := 69977)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59133⟩) (rightExpression := ⟨62112⟩)
    (transferEvent := 69978) (summaryTransferEvent := 69979)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69975.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult66571.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult69980

namespace SemanticResult69985
def owner : Owner := ⟨.program ⟨257⟩, ⟨65093⟩⟩
def rawTerms : List Term := Proof.Events273.exact69985RawTerms
def summary : Bound := (.finite 289706631804066638652128995049472)
def resultEvent : Nat := 69985
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69985.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult69980.owner)
    (rightOwner := SemanticResult66089.owner)
    (leftResult := 69980) (rightResult := 66089)
    (leftActual := SemanticResult69980.actual selector witness)
    (rightActual := SemanticResult66089.actual selector witness)
    (leftRaw := SemanticResult69980.rawTerms)
    (rightRaw := SemanticResult66089.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 257515860087126057990209472036864)
    (rightMaximum := 32190771716940580661919523012608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 69981) (rightBinding := 69982)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62113⟩) (rightExpression := ⟨65092⟩)
    (transferEvent := 69983) (summaryTransferEvent := 69984)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69980.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult66089.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult69985

namespace SemanticResult69990
def owner : Owner := ⟨.program ⟨257⟩, ⟨70734⟩⟩
def rawTerms : List Term := Proof.Events273.exact69990RawTerms
def summary : Bound := (.finite 321897992872344281445771187322880)
def resultEvent : Nat := 69990
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult69990.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult69985.owner)
    (rightOwner := SemanticResult65607.owner)
    (leftResult := 69985) (rightResult := 65607)
    (leftActual := SemanticResult69985.actual selector witness)
    (rightActual := SemanticResult65607.actual selector witness)
    (leftRaw := SemanticResult69985.rawTerms)
    (rightRaw := SemanticResult65607.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 289706631804066638652128995049472)
    (rightMaximum := 32191361068277642793642192273408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 69986) (rightBinding := 69987)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65093⟩) (rightExpression := ⟨70733⟩)
    (transferEvent := 69988) (summaryTransferEvent := 69989)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69985.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65607.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult69990

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
