import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1725
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1659
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1677
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1680
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1684
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1688
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1691
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1695
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1699
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1703
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1706
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1710
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1714
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1717
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1721
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1724

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult245429
def owner : Owner := ⟨.program ⟨257⟩, ⟨17709⟩⟩
def rawTerms : List Term := Proof.Events958.exact245429RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 245429
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245429.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 245426) (rightBinding := 245427)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16005⟩) (rightExpression := ⟨17706⟩)
    (transferEvent := 245428)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult245425.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult245410.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult245429

namespace SemanticResult245438
def owner : Owner := ⟨.program ⟨257⟩, ⟨16559⟩⟩
def rawTerms : List Term := Proof.Events958.exact245438RawTerms
def summary : Bound := (.finite 202072841853861888)
def resultEvent : Nat := 245438
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245438.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1376256
      (.finite ⟨26, by decide⟩)
      (.finite ⟨5647228698, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge245273.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge245273.frameStart)
    (owner := owner) (leftOwner := SemanticResult236870.owner)
    (rightOwner := SemanticResult245267.owner)
    (leftResult := 236870) (rightResult := 245267)
    (leftActual := SemanticResult236870.actual selector witness)
    (rightActual := SemanticResult245267.actual selector witness)
    (leftRaw := SemanticResult236870.rawTerms)
    (rightRaw := SemanticResult245267.rawTerms)
    (working := LeftOperatorMerge245273.working)
    (leftBinding := 245268) (rightBinding := 245269)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5563⟩) (rightExpression := ⟨16558⟩)
    (coefficientTransfer := 245270) (summaryTransfer := 245272)
    (rightCoefficientProducer := 245266)
    (rightSummaryTransfer := 245271)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨5647228698, by decide⟩)
    (rightRecordedMaximum := 5647228698)
    (rightSummaryMaximum := ⟨5647228698, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1376256)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge245273.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound245266.actual selector witness)
    (summaryMagnitude := LeftBound245272.actual selector witness)
    (reconstruction := LeftOperatorMerge245273.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236870.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult245267.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245266.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound245266.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge245273.operationAgreement
  · exact LeftBound245272.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge245273.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 245433 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17705⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨15772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16983⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨16003⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17705⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16983⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16003⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge245273.working
    [{ coefficient := (1), key := LeftRelationMerge245433.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge245433.frameStart
      LeftRelationMerge245433.owner (.relation 245433) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge245433.deltas
    rows := LeftRelationMerge245433.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge245273.working LeftRelationMerge245433.source
        (relationContext LeftRelationMerge245433.source
          LeftRelationMerge245433.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge245273.working, LeftRelationMerge245433.deltas,
    LeftRelationMerge245433.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 245433)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨16559⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16556⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16556⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge245273.working) (working := relationWorking0)
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
end SemanticResult245438

namespace SemanticResult245445
def owner : Owner := ⟨.program ⟨257⟩, ⟨17708⟩⟩
def rawTerms : List Term := Proof.Events958.exact245445RawTerms
def summary : Bound := (.finite 32188807212483706889510625476608)
def resultEvent : Nat := 245445
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245445.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge245442.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult245438.owner)
    (rightOwner := SemanticResult245260.owner)
    (leftResult := 245438) (rightResult := 245260)
    (leftActual := SemanticResult245438.actual selector witness)
    (rightActual := SemanticResult245260.actual selector witness)
    (leftRaw := SemanticResult245438.rawTerms)
    (rightRaw := SemanticResult245260.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32188807212483504816668771614720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 245439) (rightBinding := 245440)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16559⟩) (rightExpression := ⟨17707⟩)
    (coefficientTransfer := 245441) (summaryTransfer := 245444)
    (base := LeftOperatorMerge245442.base)
    (reconstruction := LeftOperatorMerge245442.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult245438.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult245260.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge245442.operationAgreement
  · rfl
  · decide
end SemanticResult245445

namespace SemanticResult245450
def owner : Owner := ⟨.program ⟨257⟩, ⟨20594⟩⟩
def rawTerms : List Term := Proof.Events958.exact245450RawTerms
def summary : Bound := (.finite 64377712650190257467641695830016)
def resultEvent : Nat := 245450
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245450.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult245445.owner)
    (rightOwner := SemanticResult244963.owner)
    (leftResult := 245445) (rightResult := 244963)
    (leftActual := SemanticResult245445.actual selector witness)
    (rightActual := SemanticResult244963.actual selector witness)
    (leftRaw := SemanticResult245445.rawTerms)
    (rightRaw := SemanticResult244963.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 32188807212483706889510625476608)
    (rightMaximum := 32188905437706550578131070353408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 245446) (rightBinding := 245447)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17708⟩) (rightExpression := ⟨20593⟩)
    (transferEvent := 245448) (summaryTransferEvent := 245449)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult245445.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult244963.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult245450

namespace SemanticResult245455
def owner : Owner := ⟨.program ⟨257⟩, ⟨23814⟩⟩
def rawTerms : List Term := Proof.Events958.exact245455RawTerms
def summary : Bound := (.finite 96566716313119651734393211060224)
def resultEvent : Nat := 245455
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245455.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult245450.owner)
    (rightOwner := SemanticResult244481.owner)
    (leftResult := 245450) (rightResult := 244481)
    (leftActual := SemanticResult245450.actual selector witness)
    (rightActual := SemanticResult244481.actual selector witness)
    (leftRaw := SemanticResult245450.rawTerms)
    (rightRaw := SemanticResult244481.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 64377712650190257467641695830016)
    (rightMaximum := 32189003662929394266751515230208) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 245451) (rightBinding := 245452)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20594⟩) (rightExpression := ⟨23813⟩)
    (transferEvent := 245453) (summaryTransferEvent := 245454)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult245450.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult244481.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult245455

namespace SemanticResult245460
def owner : Owner := ⟨.program ⟨257⟩, ⟨33834⟩⟩
def rawTerms : List Term := Proof.Events958.exact245460RawTerms
def summary : Bound := (.finite 128755916426494733378385616044032)
def resultEvent : Nat := 245460
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245460.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult245455.owner)
    (rightOwner := SemanticResult243999.owner)
    (leftResult := 245455) (rightResult := 243999)
    (leftActual := SemanticResult245455.actual selector witness)
    (rightActual := SemanticResult243999.actual selector witness)
    (leftRaw := SemanticResult245455.rawTerms)
    (rightRaw := SemanticResult243999.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 96566716313119651734393211060224)
    (rightMaximum := 32189200113375081643992404983808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 245456) (rightBinding := 245457)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23814⟩) (rightExpression := ⟨33833⟩)
    (transferEvent := 245458) (summaryTransferEvent := 245459)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult245455.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult243999.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult245460

namespace SemanticResult245465
def owner : Owner := ⟨.program ⟨257⟩, ⟨52894⟩⟩
def rawTerms : List Term := Proof.Events958.exact245465RawTerms
def summary : Bound := (.finite 160945509440761189776859800535040)
def resultEvent : Nat := 245465
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245465.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult245460.owner)
    (rightOwner := SemanticResult243517.owner)
    (leftResult := 245460) (rightResult := 243517)
    (leftActual := SemanticResult245460.actual selector witness)
    (rightActual := SemanticResult243517.actual selector witness)
    (leftRaw := SemanticResult245460.rawTerms)
    (rightRaw := SemanticResult243517.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 128755916426494733378385616044032)
    (rightMaximum := 32189593014266456398474184491008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 245461) (rightBinding := 245462)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33834⟩) (rightExpression := ⟨52893⟩)
    (transferEvent := 245463) (summaryTransferEvent := 245464)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult245460.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult243517.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult245465

namespace SemanticResult245470
def owner : Owner := ⟨.program ⟨257⟩, ⟨55874⟩⟩
def rawTerms : List Term := Proof.Events958.exact245470RawTerms
def summary : Bound := (.finite 193135298905473333552574874779648)
def resultEvent : Nat := 245470
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245470.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult245465.owner)
    (rightOwner := SemanticResult243035.owner)
    (leftResult := 245465) (rightResult := 243035)
    (leftActual := SemanticResult245465.actual selector witness)
    (rightActual := SemanticResult243035.actual selector witness)
    (leftRaw := SemanticResult245465.rawTerms)
    (rightRaw := SemanticResult243035.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 160945509440761189776859800535040)
    (rightMaximum := 32189789464712143775715074244608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 245466) (rightBinding := 245467)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨52894⟩) (rightExpression := ⟨55873⟩)
    (transferEvent := 245468) (summaryTransferEvent := 245469)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult245465.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult243035.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult245470

namespace SemanticResult245475
def owner : Owner := ⟨.program ⟨257⟩, ⟨58854⟩⟩
def rawTerms : List Term := Proof.Events958.exact245475RawTerms
def summary : Bound := (.finite 225325481271076852082771728531456)
def resultEvent : Nat := 245475
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245475.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult245470.owner)
    (rightOwner := SemanticResult242553.owner)
    (leftResult := 245470) (rightResult := 242553)
    (leftActual := SemanticResult245470.actual selector witness)
    (rightActual := SemanticResult242553.actual selector witness)
    (leftRaw := SemanticResult245470.rawTerms)
    (rightRaw := SemanticResult242553.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 193135298905473333552574874779648)
    (rightMaximum := 32190182365603518530196853751808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 245471) (rightBinding := 245472)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55874⟩) (rightExpression := ⟨58853⟩)
    (transferEvent := 245473) (summaryTransferEvent := 245474)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult245470.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult242553.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult245475

namespace SemanticResult245480
def owner : Owner := ⟨.program ⟨257⟩, ⟨61834⟩⟩
def rawTerms : List Term := Proof.Events958.exact245480RawTerms
def summary : Bound := (.finite 257515860087126057990209472036864)
def resultEvent : Nat := 245480
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245480.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult245475.owner)
    (rightOwner := SemanticResult242071.owner)
    (leftResult := 245475) (rightResult := 242071)
    (leftActual := SemanticResult245475.actual selector witness)
    (rightActual := SemanticResult242071.actual selector witness)
    (leftRaw := SemanticResult245475.rawTerms)
    (rightRaw := SemanticResult242071.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 225325481271076852082771728531456)
    (rightMaximum := 32190378816049205907437743505408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 245476) (rightBinding := 245477)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58854⟩) (rightExpression := ⟨61833⟩)
    (transferEvent := 245478) (summaryTransferEvent := 245479)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult245475.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult242071.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult245480

namespace SemanticResult245485
def owner : Owner := ⟨.program ⟨257⟩, ⟨64814⟩⟩
def rawTerms : List Term := Proof.Events958.exact245485RawTerms
def summary : Bound := (.finite 289706631804066638652128995049472)
def resultEvent : Nat := 245485
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245485.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult245480.owner)
    (rightOwner := SemanticResult241589.owner)
    (leftResult := 245480) (rightResult := 241589)
    (leftActual := SemanticResult245480.actual selector witness)
    (rightActual := SemanticResult241589.actual selector witness)
    (leftRaw := SemanticResult245480.rawTerms)
    (rightRaw := SemanticResult241589.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 257515860087126057990209472036864)
    (rightMaximum := 32190771716940580661919523012608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 245481) (rightBinding := 245482)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61834⟩) (rightExpression := ⟨64813⟩)
    (transferEvent := 245483) (summaryTransferEvent := 245484)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult245480.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult241589.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult245485

namespace SemanticResult245490
def owner : Owner := ⟨.program ⟨257⟩, ⟨70023⟩⟩
def rawTerms : List Term := Proof.Events958.exact245490RawTerms
def summary : Bound := (.finite 321897992872344281445771187322880)
def resultEvent : Nat := 245490
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245490.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult245485.owner)
    (rightOwner := SemanticResult241107.owner)
    (leftResult := 245485) (rightResult := 241107)
    (leftActual := SemanticResult245485.actual selector witness)
    (rightActual := SemanticResult241107.actual selector witness)
    (leftRaw := SemanticResult245485.rawTerms)
    (rightRaw := SemanticResult241107.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 289706631804066638652128995049472)
    (rightMaximum := 32191361068277642793642192273408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 245486) (rightBinding := 245487)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64814⟩) (rightExpression := ⟨70022⟩)
    (transferEvent := 245488) (summaryTransferEvent := 245489)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult245485.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult241107.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult245490

namespace SemanticResult245495
def owner : Owner := ⟨.program ⟨257⟩, ⟨70024⟩⟩
def rawTerms : List Term := Proof.Events958.exact245495RawTerms
def summary : Bound := (.finite 354089550391067611616654269349888)
def resultEvent : Nat := 245495
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245495.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult245490.owner)
    (rightOwner := SemanticResult240625.owner)
    (leftResult := 245490) (rightResult := 240625)
    (leftActual := SemanticResult245490.actual selector witness)
    (rightActual := SemanticResult240625.actual selector witness)
    (leftRaw := SemanticResult245490.rawTerms)
    (rightRaw := SemanticResult240625.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 321897992872344281445771187322880)
    (rightMaximum := 32191557518723330170883082027008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 245491) (rightBinding := 245492)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70023⟩) (rightExpression := ⟨28242⟩)
    (transferEvent := 245493) (summaryTransferEvent := 245494)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult245490.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult240625.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult245495

namespace SemanticResult245500
def owner : Owner := ⟨.program ⟨257⟩, ⟨70025⟩⟩
def rawTerms : List Term := Proof.Events958.exact245500RawTerms
def summary : Bound := (.finite 386281697261128003919260020637696)
def resultEvent : Nat := 245500
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245500.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult245495.owner)
    (rightOwner := SemanticResult240143.owner)
    (leftResult := 245495) (rightResult := 240143)
    (leftActual := SemanticResult245495.actual selector witness)
    (rightActual := SemanticResult240143.actual selector witness)
    (leftRaw := SemanticResult245495.rawTerms)
    (rightRaw := SemanticResult240143.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 354089550391067611616654269349888)
    (rightMaximum := 32192146870060392302605751287808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 245496) (rightBinding := 245497)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70024⟩) (rightExpression := ⟨30922⟩)
    (transferEvent := 245498) (summaryTransferEvent := 245499)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult245495.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult240143.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult245500

namespace SemanticResult245505
def owner : Owner := ⟨.program ⟨257⟩, ⟨70026⟩⟩
def rawTerms : List Term := Proof.Events959.exact245505RawTerms
def summary : Bound := (.finite 418474237032079770976347551432704)
def resultEvent : Nat := 245505
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245505.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult245500.owner)
    (rightOwner := SemanticResult239661.owner)
    (leftResult := 245500) (rightResult := 239661)
    (leftActual := SemanticResult245500.actual selector witness)
    (rightActual := SemanticResult239661.actual selector witness)
    (leftRaw := SemanticResult245500.rawTerms)
    (rightRaw := SemanticResult239661.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 386281697261128003919260020637696)
    (rightMaximum := 32192539770951767057087530795008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 245501) (rightBinding := 245502)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70025⟩) (rightExpression := ⟨36582⟩)
    (transferEvent := 245503) (summaryTransferEvent := 245504)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult245500.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult239661.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult245505

namespace SemanticResult245510
def owner : Owner := ⟨.program ⟨257⟩, ⟨70027⟩⟩
def rawTerms : List Term := Proof.Events959.exact245510RawTerms
def summary : Bound := (.finite 450666973253477225410675971981312)
def resultEvent : Nat := 245510
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult245510.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult245505.owner)
    (rightOwner := SemanticResult239179.owner)
    (leftResult := 245505) (rightResult := 239179)
    (leftActual := SemanticResult245505.actual selector witness)
    (rightActual := SemanticResult239179.actual selector witness)
    (leftRaw := SemanticResult245505.rawTerms)
    (rightRaw := SemanticResult239179.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 418474237032079770976347551432704)
    (rightMaximum := 32192736221397454434328420548608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 245506) (rightBinding := 245507)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70026⟩) (rightExpression := ⟨39262⟩)
    (transferEvent := 245508) (summaryTransferEvent := 245509)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult245505.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult239179.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult245510

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
