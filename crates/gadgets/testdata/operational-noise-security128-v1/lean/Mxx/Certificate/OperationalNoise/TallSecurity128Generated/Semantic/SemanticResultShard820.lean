import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard820
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard754
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard768
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard772
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard775
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard779
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard783
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard786
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard790
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard794
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard797
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard801
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard805
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard809
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard812
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard816
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard818
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard819

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult113813
def owner : Owner := ⟨.program ⟨257⟩, ⟨16619⟩⟩
def rawTerms : List Term := Proof.Events444.exact113813RawTerms
def summary : Bound := (.finite 202072841853861888)
def resultEvent : Nat := 113813
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113813.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1376256
      (.finite ⟨26, by decide⟩)
      (.finite ⟨5647228698, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge113648.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge113648.frameStart)
    (owner := owner) (leftOwner := SemanticResult105245.owner)
    (rightOwner := SemanticResult113642.owner)
    (leftResult := 105245) (rightResult := 113642)
    (leftActual := SemanticResult105245.actual selector witness)
    (rightActual := SemanticResult113642.actual selector witness)
    (leftRaw := SemanticResult105245.rawTerms)
    (rightRaw := SemanticResult113642.rawTerms)
    (working := LeftOperatorMerge113648.working)
    (leftBinding := 113643) (rightBinding := 113644)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5770⟩) (rightExpression := ⟨16618⟩)
    (coefficientTransfer := 113645) (summaryTransfer := 113647)
    (rightCoefficientProducer := 113641)
    (rightSummaryTransfer := 113646)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨5647228698, by decide⟩)
    (rightRecordedMaximum := 5647228698)
    (rightSummaryMaximum := ⟨5647228698, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1376256)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge113648.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound113641.actual selector witness)
    (summaryMagnitude := LeftBound113647.actual selector witness)
    (reconstruction := LeftOperatorMerge113648.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult105245.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult113642.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113641.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound113641.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge113648.operationAgreement
  · exact LeftBound113647.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge113648.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 113808 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17789⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨15796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17010⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨16051⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17789⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17010⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16051⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge113648.working
    [{ coefficient := (1), key := LeftRelationMerge113808.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge113808.frameStart
      LeftRelationMerge113808.owner (.relation 113808) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge113808.deltas
    rows := LeftRelationMerge113808.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge113648.working LeftRelationMerge113808.source
        (relationContext LeftRelationMerge113808.source
          LeftRelationMerge113808.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge113648.working, LeftRelationMerge113808.deltas,
    LeftRelationMerge113808.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 113808)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨16619⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16616⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16616⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge113648.working) (working := relationWorking0)
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
end SemanticResult113813

namespace SemanticResult113820
def owner : Owner := ⟨.program ⟨257⟩, ⟨17792⟩⟩
def rawTerms : List Term := Proof.Events444.exact113820RawTerms
def summary : Bound := (.finite 32188807212483706889510625476608)
def resultEvent : Nat := 113820
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113820.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge113817.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult113813.owner)
    (rightOwner := SemanticResult113635.owner)
    (leftResult := 113813) (rightResult := 113635)
    (leftActual := SemanticResult113813.actual selector witness)
    (rightActual := SemanticResult113635.actual selector witness)
    (leftRaw := SemanticResult113813.rawTerms)
    (rightRaw := SemanticResult113635.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32188807212483504816668771614720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 113814) (rightBinding := 113815)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16619⟩) (rightExpression := ⟨17791⟩)
    (coefficientTransfer := 113816) (summaryTransfer := 113819)
    (base := LeftOperatorMerge113817.base)
    (reconstruction := LeftOperatorMerge113817.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult113813.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult113635.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge113817.operationAgreement
  · rfl
  · decide
end SemanticResult113820

namespace SemanticResult113825
def owner : Owner := ⟨.program ⟨257⟩, ⟨20687⟩⟩
def rawTerms : List Term := Proof.Events444.exact113825RawTerms
def summary : Bound := (.finite 64377712650190257467641695830016)
def resultEvent : Nat := 113825
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113825.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult113820.owner)
    (rightOwner := SemanticResult113338.owner)
    (leftResult := 113820) (rightResult := 113338)
    (leftActual := SemanticResult113820.actual selector witness)
    (rightActual := SemanticResult113338.actual selector witness)
    (leftRaw := SemanticResult113820.rawTerms)
    (rightRaw := SemanticResult113338.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 32188807212483706889510625476608)
    (rightMaximum := 32188905437706550578131070353408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 113821) (rightBinding := 113822)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17792⟩) (rightExpression := ⟨20686⟩)
    (transferEvent := 113823) (summaryTransferEvent := 113824)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult113820.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult113338.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult113825

namespace SemanticResult113830
def owner : Owner := ⟨.program ⟨257⟩, ⟨23907⟩⟩
def rawTerms : List Term := Proof.Events444.exact113830RawTerms
def summary : Bound := (.finite 96566716313119651734393211060224)
def resultEvent : Nat := 113830
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113830.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult113825.owner)
    (rightOwner := SemanticResult112856.owner)
    (leftResult := 113825) (rightResult := 112856)
    (leftActual := SemanticResult113825.actual selector witness)
    (rightActual := SemanticResult112856.actual selector witness)
    (leftRaw := SemanticResult113825.rawTerms)
    (rightRaw := SemanticResult112856.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 64377712650190257467641695830016)
    (rightMaximum := 32189003662929394266751515230208) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 113826) (rightBinding := 113827)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20687⟩) (rightExpression := ⟨23906⟩)
    (transferEvent := 113828) (summaryTransferEvent := 113829)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult113825.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult112856.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult113830

namespace SemanticResult113835
def owner : Owner := ⟨.program ⟨257⟩, ⟨33927⟩⟩
def rawTerms : List Term := Proof.Events444.exact113835RawTerms
def summary : Bound := (.finite 128755916426494733378385616044032)
def resultEvent : Nat := 113835
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113835.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult113830.owner)
    (rightOwner := SemanticResult112374.owner)
    (leftResult := 113830) (rightResult := 112374)
    (leftActual := SemanticResult113830.actual selector witness)
    (rightActual := SemanticResult112374.actual selector witness)
    (leftRaw := SemanticResult113830.rawTerms)
    (rightRaw := SemanticResult112374.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 96566716313119651734393211060224)
    (rightMaximum := 32189200113375081643992404983808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 113831) (rightBinding := 113832)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23907⟩) (rightExpression := ⟨33926⟩)
    (transferEvent := 113833) (summaryTransferEvent := 113834)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult113830.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult112374.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult113835

namespace SemanticResult113840
def owner : Owner := ⟨.program ⟨257⟩, ⟨52987⟩⟩
def rawTerms : List Term := Proof.Events444.exact113840RawTerms
def summary : Bound := (.finite 160945509440761189776859800535040)
def resultEvent : Nat := 113840
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113840.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult113835.owner)
    (rightOwner := SemanticResult111892.owner)
    (leftResult := 113835) (rightResult := 111892)
    (leftActual := SemanticResult113835.actual selector witness)
    (rightActual := SemanticResult111892.actual selector witness)
    (leftRaw := SemanticResult113835.rawTerms)
    (rightRaw := SemanticResult111892.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 128755916426494733378385616044032)
    (rightMaximum := 32189593014266456398474184491008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 113836) (rightBinding := 113837)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33927⟩) (rightExpression := ⟨52986⟩)
    (transferEvent := 113838) (summaryTransferEvent := 113839)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult113835.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult111892.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult113840

namespace SemanticResult113845
def owner : Owner := ⟨.program ⟨257⟩, ⟨55967⟩⟩
def rawTerms : List Term := Proof.Events444.exact113845RawTerms
def summary : Bound := (.finite 193135298905473333552574874779648)
def resultEvent : Nat := 113845
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113845.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult113840.owner)
    (rightOwner := SemanticResult111410.owner)
    (leftResult := 113840) (rightResult := 111410)
    (leftActual := SemanticResult113840.actual selector witness)
    (rightActual := SemanticResult111410.actual selector witness)
    (leftRaw := SemanticResult113840.rawTerms)
    (rightRaw := SemanticResult111410.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 160945509440761189776859800535040)
    (rightMaximum := 32189789464712143775715074244608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 113841) (rightBinding := 113842)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨52987⟩) (rightExpression := ⟨55966⟩)
    (transferEvent := 113843) (summaryTransferEvent := 113844)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult113840.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult111410.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult113845

namespace SemanticResult113850
def owner : Owner := ⟨.program ⟨257⟩, ⟨58947⟩⟩
def rawTerms : List Term := Proof.Events444.exact113850RawTerms
def summary : Bound := (.finite 225325481271076852082771728531456)
def resultEvent : Nat := 113850
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113850.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult113845.owner)
    (rightOwner := SemanticResult110928.owner)
    (leftResult := 113845) (rightResult := 110928)
    (leftActual := SemanticResult113845.actual selector witness)
    (rightActual := SemanticResult110928.actual selector witness)
    (leftRaw := SemanticResult113845.rawTerms)
    (rightRaw := SemanticResult110928.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 193135298905473333552574874779648)
    (rightMaximum := 32190182365603518530196853751808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 113846) (rightBinding := 113847)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55967⟩) (rightExpression := ⟨58946⟩)
    (transferEvent := 113848) (summaryTransferEvent := 113849)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult113845.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult110928.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult113850

namespace SemanticResult113855
def owner : Owner := ⟨.program ⟨257⟩, ⟨61927⟩⟩
def rawTerms : List Term := Proof.Events444.exact113855RawTerms
def summary : Bound := (.finite 257515860087126057990209472036864)
def resultEvent : Nat := 113855
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113855.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult113850.owner)
    (rightOwner := SemanticResult110446.owner)
    (leftResult := 113850) (rightResult := 110446)
    (leftActual := SemanticResult113850.actual selector witness)
    (rightActual := SemanticResult110446.actual selector witness)
    (leftRaw := SemanticResult113850.rawTerms)
    (rightRaw := SemanticResult110446.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 225325481271076852082771728531456)
    (rightMaximum := 32190378816049205907437743505408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 113851) (rightBinding := 113852)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58947⟩) (rightExpression := ⟨61926⟩)
    (transferEvent := 113853) (summaryTransferEvent := 113854)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult113850.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult110446.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult113855

namespace SemanticResult113860
def owner : Owner := ⟨.program ⟨257⟩, ⟨64907⟩⟩
def rawTerms : List Term := Proof.Events444.exact113860RawTerms
def summary : Bound := (.finite 289706631804066638652128995049472)
def resultEvent : Nat := 113860
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113860.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult113855.owner)
    (rightOwner := SemanticResult109964.owner)
    (leftResult := 113855) (rightResult := 109964)
    (leftActual := SemanticResult113855.actual selector witness)
    (rightActual := SemanticResult109964.actual selector witness)
    (leftRaw := SemanticResult113855.rawTerms)
    (rightRaw := SemanticResult109964.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 257515860087126057990209472036864)
    (rightMaximum := 32190771716940580661919523012608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 113856) (rightBinding := 113857)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61927⟩) (rightExpression := ⟨64906⟩)
    (transferEvent := 113858) (summaryTransferEvent := 113859)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult113855.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult109964.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult113860

namespace SemanticResult113865
def owner : Owner := ⟨.program ⟨257⟩, ⟨70260⟩⟩
def rawTerms : List Term := Proof.Events444.exact113865RawTerms
def summary : Bound := (.finite 321897992872344281445771187322880)
def resultEvent : Nat := 113865
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113865.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult113860.owner)
    (rightOwner := SemanticResult109482.owner)
    (leftResult := 113860) (rightResult := 109482)
    (leftActual := SemanticResult113860.actual selector witness)
    (rightActual := SemanticResult109482.actual selector witness)
    (leftRaw := SemanticResult113860.rawTerms)
    (rightRaw := SemanticResult109482.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 289706631804066638652128995049472)
    (rightMaximum := 32191361068277642793642192273408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 113861) (rightBinding := 113862)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64907⟩) (rightExpression := ⟨70259⟩)
    (transferEvent := 113863) (summaryTransferEvent := 113864)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult113860.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult109482.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult113865

namespace SemanticResult113870
def owner : Owner := ⟨.program ⟨257⟩, ⟨70261⟩⟩
def rawTerms : List Term := Proof.Events444.exact113870RawTerms
def summary : Bound := (.finite 354089550391067611616654269349888)
def resultEvent : Nat := 113870
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113870.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult113865.owner)
    (rightOwner := SemanticResult109000.owner)
    (leftResult := 113865) (rightResult := 109000)
    (leftActual := SemanticResult113865.actual selector witness)
    (rightActual := SemanticResult109000.actual selector witness)
    (leftRaw := SemanticResult113865.rawTerms)
    (rightRaw := SemanticResult109000.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 321897992872344281445771187322880)
    (rightMaximum := 32191557518723330170883082027008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 113866) (rightBinding := 113867)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70260⟩) (rightExpression := ⟨28317⟩)
    (transferEvent := 113868) (summaryTransferEvent := 113869)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult113865.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult109000.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult113870

namespace SemanticResult113875
def owner : Owner := ⟨.program ⟨257⟩, ⟨70262⟩⟩
def rawTerms : List Term := Proof.Events444.exact113875RawTerms
def summary : Bound := (.finite 386281697261128003919260020637696)
def resultEvent : Nat := 113875
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113875.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult113870.owner)
    (rightOwner := SemanticResult108518.owner)
    (leftResult := 113870) (rightResult := 108518)
    (leftActual := SemanticResult113870.actual selector witness)
    (rightActual := SemanticResult108518.actual selector witness)
    (leftRaw := SemanticResult113870.rawTerms)
    (rightRaw := SemanticResult108518.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 354089550391067611616654269349888)
    (rightMaximum := 32192146870060392302605751287808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 113871) (rightBinding := 113872)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70261⟩) (rightExpression := ⟨30997⟩)
    (transferEvent := 113873) (summaryTransferEvent := 113874)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult113870.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult108518.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult113875

namespace SemanticResult113880
def owner : Owner := ⟨.program ⟨257⟩, ⟨70263⟩⟩
def rawTerms : List Term := Proof.Events444.exact113880RawTerms
def summary : Bound := (.finite 418474237032079770976347551432704)
def resultEvent : Nat := 113880
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113880.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult113875.owner)
    (rightOwner := SemanticResult108036.owner)
    (leftResult := 113875) (rightResult := 108036)
    (leftActual := SemanticResult113875.actual selector witness)
    (rightActual := SemanticResult108036.actual selector witness)
    (leftRaw := SemanticResult113875.rawTerms)
    (rightRaw := SemanticResult108036.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 386281697261128003919260020637696)
    (rightMaximum := 32192539770951767057087530795008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 113876) (rightBinding := 113877)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70262⟩) (rightExpression := ⟨36657⟩)
    (transferEvent := 113878) (summaryTransferEvent := 113879)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult113875.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult108036.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult113880

namespace SemanticResult113885
def owner : Owner := ⟨.program ⟨257⟩, ⟨70264⟩⟩
def rawTerms : List Term := Proof.Events444.exact113885RawTerms
def summary : Bound := (.finite 450666973253477225410675971981312)
def resultEvent : Nat := 113885
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113885.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult113880.owner)
    (rightOwner := SemanticResult107554.owner)
    (leftResult := 113880) (rightResult := 107554)
    (leftActual := SemanticResult113880.actual selector witness)
    (rightActual := SemanticResult107554.actual selector witness)
    (leftRaw := SemanticResult113880.rawTerms)
    (rightRaw := SemanticResult107554.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 418474237032079770976347551432704)
    (rightMaximum := 32192736221397454434328420548608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 113881) (rightBinding := 113882)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70263⟩) (rightExpression := ⟨39337⟩)
    (transferEvent := 113883) (summaryTransferEvent := 113884)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult113880.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult107554.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult113885

namespace SemanticResult113890
def owner : Owner := ⟨.program ⟨257⟩, ⟨70265⟩⟩
def rawTerms : List Term := Proof.Events444.exact113890RawTerms
def summary : Bound := (.finite 482860102375766054599486172037120)
def resultEvent : Nat := 113890
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult113890.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult113885.owner)
    (rightOwner := SemanticResult107072.owner)
    (leftResult := 113885) (rightResult := 107072)
    (leftActual := SemanticResult113885.actual selector witness)
    (rightActual := SemanticResult107072.actual selector witness)
    (leftRaw := SemanticResult113885.rawTerms)
    (rightRaw := SemanticResult107072.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 450666973253477225410675971981312)
    (rightMaximum := 32193129122288829188810200055808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 113886) (rightBinding := 113887)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70264⟩) (rightExpression := ⟨42017⟩)
    (transferEvent := 113888) (summaryTransferEvent := 113889)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult113885.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult107072.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult113890

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
