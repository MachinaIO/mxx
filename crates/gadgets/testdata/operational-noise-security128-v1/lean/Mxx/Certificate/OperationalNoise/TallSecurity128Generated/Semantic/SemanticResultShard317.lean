import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard317
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard252
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard280
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard284
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard287
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard291
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard295
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard298
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard302
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard306
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard309
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard313
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard316

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult40671
def owner : Owner := ⟨.program ⟨257⟩, ⟨7198⟩⟩
def rawTerms : List Term := Proof.Events158.exact40671RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40671
def producerEvent : Nat := 40670
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40671.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 40578, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult40671

namespace SemanticResult40675
def owner : Owner := ⟨.program ⟨257⟩, ⟨16181⟩⟩
def rawTerms : List Term := Proof.Events158.exact40675RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40675
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40675.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 40672) (rightBinding := 40673)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7198⟩) (rightExpression := ⟨16180⟩)
    (transferEvent := 40674)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40671.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40668.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40675

namespace SemanticResult40679
def owner : Owner := ⟨.program ⟨257⟩, ⟨18017⟩⟩
def rawTerms : List Term := Proof.Events158.exact40679RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40679
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40679.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 40676) (rightBinding := 40677)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16181⟩) (rightExpression := ⟨18014⟩)
    (transferEvent := 40678)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40675.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40660.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40679

namespace SemanticResult40688
def owner : Owner := ⟨.program ⟨257⟩, ⟨16779⟩⟩
def rawTerms : List Term := Proof.Events158.exact40688RawTerms
def summary : Bound := (.finite 202072841853861888)
def resultEvent : Nat := 40688
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40688.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1376256
      (.finite ⟨26, by decide⟩)
      (.finite ⟨5647228698, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40523.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge40523.frameStart)
    (owner := owner) (leftOwner := SemanticResult32120.owner)
    (rightOwner := SemanticResult40517.owner)
    (leftResult := 32120) (rightResult := 40517)
    (leftActual := SemanticResult32120.actual selector witness)
    (rightActual := SemanticResult40517.actual selector witness)
    (leftRaw := SemanticResult32120.rawTerms)
    (rightRaw := SemanticResult40517.rawTerms)
    (working := LeftOperatorMerge40523.working)
    (leftBinding := 40518) (rightBinding := 40519)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11643⟩) (rightExpression := ⟨16778⟩)
    (coefficientTransfer := 40520) (summaryTransfer := 40522)
    (rightCoefficientProducer := 40516)
    (rightSummaryTransfer := 40521)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨5647228698, by decide⟩)
    (rightRecordedMaximum := 5647228698)
    (rightSummaryMaximum := ⟨5647228698, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1376256)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge40523.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound40516.actual selector witness)
    (summaryMagnitude := LeftBound40522.actual selector witness)
    (reconstruction := LeftOperatorMerge40523.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult32120.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40517.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40516.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound40516.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge40523.operationAgreement
  · exact LeftBound40522.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40523.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 40683 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨18013⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17082⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨16179⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨18013⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17082⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16179⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge40523.working
    [{ coefficient := (1), key := LeftRelationMerge40683.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge40683.frameStart
      LeftRelationMerge40683.owner (.relation 40683) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge40683.deltas
    rows := LeftRelationMerge40683.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge40523.working LeftRelationMerge40683.source
        (relationContext LeftRelationMerge40683.source
          LeftRelationMerge40683.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge40523.working, LeftRelationMerge40683.deltas,
    LeftRelationMerge40683.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 40683)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨16779⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16776⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16776⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge40523.working) (working := relationWorking0)
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
end SemanticResult40688

namespace SemanticResult40695
def owner : Owner := ⟨.program ⟨257⟩, ⟨18016⟩⟩
def rawTerms : List Term := Proof.Events158.exact40695RawTerms
def summary : Bound := (.finite 32188807212483706889510625476608)
def resultEvent : Nat := 40695
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40695.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge40692.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult40688.owner)
    (rightOwner := SemanticResult40510.owner)
    (leftResult := 40688) (rightResult := 40510)
    (leftActual := SemanticResult40688.actual selector witness)
    (rightActual := SemanticResult40510.actual selector witness)
    (leftRaw := SemanticResult40688.rawTerms)
    (rightRaw := SemanticResult40510.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32188807212483504816668771614720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 40689) (rightBinding := 40690)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16779⟩) (rightExpression := ⟨18015⟩)
    (coefficientTransfer := 40691) (summaryTransfer := 40694)
    (base := LeftOperatorMerge40692.base)
    (reconstruction := LeftOperatorMerge40692.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40688.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40510.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge40692.operationAgreement
  · rfl
  · decide
end SemanticResult40695

namespace SemanticResult40700
def owner : Owner := ⟨.program ⟨257⟩, ⟨20935⟩⟩
def rawTerms : List Term := Proof.Events158.exact40700RawTerms
def summary : Bound := (.finite 64377712650190257467641695830016)
def resultEvent : Nat := 40700
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40700.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult40695.owner)
    (rightOwner := SemanticResult40213.owner)
    (leftResult := 40695) (rightResult := 40213)
    (leftActual := SemanticResult40695.actual selector witness)
    (rightActual := SemanticResult40213.actual selector witness)
    (leftRaw := SemanticResult40695.rawTerms)
    (rightRaw := SemanticResult40213.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 32188807212483706889510625476608)
    (rightMaximum := 32188905437706550578131070353408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 40696) (rightBinding := 40697)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18016⟩) (rightExpression := ⟨20934⟩)
    (transferEvent := 40698) (summaryTransferEvent := 40699)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40695.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40213.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40700

namespace SemanticResult40705
def owner : Owner := ⟨.program ⟨257⟩, ⟨24155⟩⟩
def rawTerms : List Term := Proof.Events159.exact40705RawTerms
def summary : Bound := (.finite 96566716313119651734393211060224)
def resultEvent : Nat := 40705
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40705.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult40700.owner)
    (rightOwner := SemanticResult39731.owner)
    (leftResult := 40700) (rightResult := 39731)
    (leftActual := SemanticResult40700.actual selector witness)
    (rightActual := SemanticResult39731.actual selector witness)
    (leftRaw := SemanticResult40700.rawTerms)
    (rightRaw := SemanticResult39731.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 64377712650190257467641695830016)
    (rightMaximum := 32189003662929394266751515230208) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 40701) (rightBinding := 40702)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20935⟩) (rightExpression := ⟨24154⟩)
    (transferEvent := 40703) (summaryTransferEvent := 40704)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40700.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39731.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40705

namespace SemanticResult40710
def owner : Owner := ⟨.program ⟨257⟩, ⟨34175⟩⟩
def rawTerms : List Term := Proof.Events159.exact40710RawTerms
def summary : Bound := (.finite 128755916426494733378385616044032)
def resultEvent : Nat := 40710
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40710.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult40705.owner)
    (rightOwner := SemanticResult39249.owner)
    (leftResult := 40705) (rightResult := 39249)
    (leftActual := SemanticResult40705.actual selector witness)
    (rightActual := SemanticResult39249.actual selector witness)
    (leftRaw := SemanticResult40705.rawTerms)
    (rightRaw := SemanticResult39249.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 96566716313119651734393211060224)
    (rightMaximum := 32189200113375081643992404983808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 40706) (rightBinding := 40707)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨24155⟩) (rightExpression := ⟨34174⟩)
    (transferEvent := 40708) (summaryTransferEvent := 40709)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40705.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39249.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40710

namespace SemanticResult40715
def owner : Owner := ⟨.program ⟨257⟩, ⟨53235⟩⟩
def rawTerms : List Term := Proof.Events159.exact40715RawTerms
def summary : Bound := (.finite 160945509440761189776859800535040)
def resultEvent : Nat := 40715
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40715.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult40710.owner)
    (rightOwner := SemanticResult38767.owner)
    (leftResult := 40710) (rightResult := 38767)
    (leftActual := SemanticResult40710.actual selector witness)
    (rightActual := SemanticResult38767.actual selector witness)
    (leftRaw := SemanticResult40710.rawTerms)
    (rightRaw := SemanticResult38767.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 128755916426494733378385616044032)
    (rightMaximum := 32189593014266456398474184491008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 40711) (rightBinding := 40712)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨34175⟩) (rightExpression := ⟨53234⟩)
    (transferEvent := 40713) (summaryTransferEvent := 40714)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40710.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult38767.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40715

namespace SemanticResult40720
def owner : Owner := ⟨.program ⟨257⟩, ⟨56215⟩⟩
def rawTerms : List Term := Proof.Events159.exact40720RawTerms
def summary : Bound := (.finite 193135298905473333552574874779648)
def resultEvent : Nat := 40720
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40720.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult40715.owner)
    (rightOwner := SemanticResult38285.owner)
    (leftResult := 40715) (rightResult := 38285)
    (leftActual := SemanticResult40715.actual selector witness)
    (rightActual := SemanticResult38285.actual selector witness)
    (leftRaw := SemanticResult40715.rawTerms)
    (rightRaw := SemanticResult38285.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 160945509440761189776859800535040)
    (rightMaximum := 32189789464712143775715074244608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 40716) (rightBinding := 40717)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53235⟩) (rightExpression := ⟨56214⟩)
    (transferEvent := 40718) (summaryTransferEvent := 40719)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40715.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult38285.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40720

namespace SemanticResult40725
def owner : Owner := ⟨.program ⟨257⟩, ⟨59195⟩⟩
def rawTerms : List Term := Proof.Events159.exact40725RawTerms
def summary : Bound := (.finite 225325481271076852082771728531456)
def resultEvent : Nat := 40725
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40725.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult40720.owner)
    (rightOwner := SemanticResult37803.owner)
    (leftResult := 40720) (rightResult := 37803)
    (leftActual := SemanticResult40720.actual selector witness)
    (rightActual := SemanticResult37803.actual selector witness)
    (leftRaw := SemanticResult40720.rawTerms)
    (rightRaw := SemanticResult37803.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 193135298905473333552574874779648)
    (rightMaximum := 32190182365603518530196853751808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 40721) (rightBinding := 40722)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56215⟩) (rightExpression := ⟨59194⟩)
    (transferEvent := 40723) (summaryTransferEvent := 40724)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40720.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult37803.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40725

namespace SemanticResult40730
def owner : Owner := ⟨.program ⟨257⟩, ⟨62175⟩⟩
def rawTerms : List Term := Proof.Events159.exact40730RawTerms
def summary : Bound := (.finite 257515860087126057990209472036864)
def resultEvent : Nat := 40730
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40730.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult40725.owner)
    (rightOwner := SemanticResult37321.owner)
    (leftResult := 40725) (rightResult := 37321)
    (leftActual := SemanticResult40725.actual selector witness)
    (rightActual := SemanticResult37321.actual selector witness)
    (leftRaw := SemanticResult40725.rawTerms)
    (rightRaw := SemanticResult37321.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 225325481271076852082771728531456)
    (rightMaximum := 32190378816049205907437743505408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 40726) (rightBinding := 40727)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59195⟩) (rightExpression := ⟨62174⟩)
    (transferEvent := 40728) (summaryTransferEvent := 40729)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40725.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult37321.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40730

namespace SemanticResult40735
def owner : Owner := ⟨.program ⟨257⟩, ⟨65155⟩⟩
def rawTerms : List Term := Proof.Events159.exact40735RawTerms
def summary : Bound := (.finite 289706631804066638652128995049472)
def resultEvent : Nat := 40735
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40735.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult40730.owner)
    (rightOwner := SemanticResult36839.owner)
    (leftResult := 40730) (rightResult := 36839)
    (leftActual := SemanticResult40730.actual selector witness)
    (rightActual := SemanticResult36839.actual selector witness)
    (leftRaw := SemanticResult40730.rawTerms)
    (rightRaw := SemanticResult36839.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 257515860087126057990209472036864)
    (rightMaximum := 32190771716940580661919523012608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 40731) (rightBinding := 40732)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62175⟩) (rightExpression := ⟨65154⟩)
    (transferEvent := 40733) (summaryTransferEvent := 40734)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40730.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36839.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40735

namespace SemanticResult40740
def owner : Owner := ⟨.program ⟨257⟩, ⟨70892⟩⟩
def rawTerms : List Term := Proof.Events159.exact40740RawTerms
def summary : Bound := (.finite 321897992872344281445771187322880)
def resultEvent : Nat := 40740
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40740.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult40735.owner)
    (rightOwner := SemanticResult36357.owner)
    (leftResult := 40735) (rightResult := 36357)
    (leftActual := SemanticResult40735.actual selector witness)
    (rightActual := SemanticResult36357.actual selector witness)
    (leftRaw := SemanticResult40735.rawTerms)
    (rightRaw := SemanticResult36357.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 289706631804066638652128995049472)
    (rightMaximum := 32191361068277642793642192273408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 40736) (rightBinding := 40737)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65155⟩) (rightExpression := ⟨70891⟩)
    (transferEvent := 40738) (summaryTransferEvent := 40739)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40735.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36357.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40740

namespace SemanticResult40745
def owner : Owner := ⟨.program ⟨257⟩, ⟨70893⟩⟩
def rawTerms : List Term := Proof.Events159.exact40745RawTerms
def summary : Bound := (.finite 354089550391067611616654269349888)
def resultEvent : Nat := 40745
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40745.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult40740.owner)
    (rightOwner := SemanticResult35875.owner)
    (leftResult := 40740) (rightResult := 35875)
    (leftActual := SemanticResult40740.actual selector witness)
    (rightActual := SemanticResult35875.actual selector witness)
    (leftRaw := SemanticResult40740.rawTerms)
    (rightRaw := SemanticResult35875.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 321897992872344281445771187322880)
    (rightMaximum := 32191557518723330170883082027008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 40741) (rightBinding := 40742)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70892⟩) (rightExpression := ⟨28517⟩)
    (transferEvent := 40743) (summaryTransferEvent := 40744)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40740.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35875.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40745

namespace SemanticResult40750
def owner : Owner := ⟨.program ⟨257⟩, ⟨70894⟩⟩
def rawTerms : List Term := Proof.Events159.exact40750RawTerms
def summary : Bound := (.finite 386281697261128003919260020637696)
def resultEvent : Nat := 40750
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult40750.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult40745.owner)
    (rightOwner := SemanticResult35393.owner)
    (leftResult := 40745) (rightResult := 35393)
    (leftActual := SemanticResult40745.actual selector witness)
    (rightActual := SemanticResult35393.actual selector witness)
    (leftRaw := SemanticResult40745.rawTerms)
    (rightRaw := SemanticResult35393.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 354089550391067611616654269349888)
    (rightMaximum := 32192146870060392302605751287808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 40746) (rightBinding := 40747)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70893⟩) (rightExpression := ⟨31197⟩)
    (transferEvent := 40748) (summaryTransferEvent := 40749)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40745.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35393.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40750

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
