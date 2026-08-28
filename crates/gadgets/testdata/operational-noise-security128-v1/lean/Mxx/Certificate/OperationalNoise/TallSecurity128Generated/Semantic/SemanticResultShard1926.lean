import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1926
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1861
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1885
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1889
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1893
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1896
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1900
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1904
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1907
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1911
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1915
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1918
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1922
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1925

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult274671
def owner : Owner := ⟨.program ⟨257⟩, ⟨7198⟩⟩
def rawTerms : List Term := Proof.Events1072.exact274671RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 274671
def producerEvent : Nat := 274670
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274671.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 274578, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult274671

namespace SemanticResult274675
def owner : Owner := ⟨.program ⟨257⟩, ⟨15905⟩⟩
def rawTerms : List Term := Proof.Events1072.exact274675RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 274675
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274675.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 274672) (rightBinding := 274673)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7198⟩) (rightExpression := ⟨15904⟩)
    (transferEvent := 274674)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult274671.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult274668.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult274675

namespace SemanticResult274679
def owner : Owner := ⟨.program ⟨257⟩, ⟨17533⟩⟩
def rawTerms : List Term := Proof.Events1072.exact274679RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 274679
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274679.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 274676) (rightBinding := 274677)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15905⟩) (rightExpression := ⟨17530⟩)
    (transferEvent := 274678)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult274675.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult274660.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult274679

namespace SemanticResult274688
def owner : Owner := ⟨.program ⟨257⟩, ⟨16433⟩⟩
def rawTerms : List Term := Proof.Events1073.exact274688RawTerms
def summary : Bound := (.finite 202072841853861888)
def resultEvent : Nat := 274688
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274688.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1376256
      (.finite ⟨26, by decide⟩)
      (.finite ⟨5647228698, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge274523.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge274523.frameStart)
    (owner := owner) (leftOwner := SemanticResult266120.owner)
    (rightOwner := SemanticResult274517.owner)
    (leftResult := 266120) (rightResult := 274517)
    (leftActual := SemanticResult266120.actual selector witness)
    (rightActual := SemanticResult274517.actual selector witness)
    (leftRaw := SemanticResult266120.rawTerms)
    (rightRaw := SemanticResult274517.rawTerms)
    (working := LeftOperatorMerge274523.working)
    (leftBinding := 274518) (rightBinding := 274519)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5449⟩) (rightExpression := ⟨16432⟩)
    (coefficientTransfer := 274520) (summaryTransfer := 274522)
    (rightCoefficientProducer := 274516)
    (rightSummaryTransfer := 274521)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨5647228698, by decide⟩)
    (rightRecordedMaximum := 5647228698)
    (rightSummaryMaximum := ⟨5647228698, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1376256)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge274523.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound274516.actual selector witness)
    (summaryMagnitude := LeftBound274522.actual selector witness)
    (reconstruction := LeftOperatorMerge274523.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult266120.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult274517.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274516.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound274516.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge274523.operationAgreement
  · exact LeftBound274522.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge274523.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 274683 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17529⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16926⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17529⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16926⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15903⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge274523.working
    [{ coefficient := (1), key := LeftRelationMerge274683.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge274683.frameStart
      LeftRelationMerge274683.owner (.relation 274683) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge274683.deltas
    rows := LeftRelationMerge274683.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge274523.working LeftRelationMerge274683.source
        (relationContext LeftRelationMerge274683.source
          LeftRelationMerge274683.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge274523.working, LeftRelationMerge274683.deltas,
    LeftRelationMerge274683.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 274683)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨16433⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16430⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16430⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge274523.working) (working := relationWorking0)
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
end SemanticResult274688

namespace SemanticResult274695
def owner : Owner := ⟨.program ⟨257⟩, ⟨17532⟩⟩
def rawTerms : List Term := Proof.Events1073.exact274695RawTerms
def summary : Bound := (.finite 32188807212483706889510625476608)
def resultEvent : Nat := 274695
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274695.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge274692.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult274688.owner)
    (rightOwner := SemanticResult274510.owner)
    (leftResult := 274688) (rightResult := 274510)
    (leftActual := SemanticResult274688.actual selector witness)
    (rightActual := SemanticResult274510.actual selector witness)
    (leftRaw := SemanticResult274688.rawTerms)
    (rightRaw := SemanticResult274510.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32188807212483504816668771614720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 274689) (rightBinding := 274690)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16433⟩) (rightExpression := ⟨17531⟩)
    (coefficientTransfer := 274691) (summaryTransfer := 274694)
    (base := LeftOperatorMerge274692.base)
    (reconstruction := LeftOperatorMerge274692.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult274688.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult274510.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge274692.operationAgreement
  · rfl
  · decide
end SemanticResult274695

namespace SemanticResult274700
def owner : Owner := ⟨.program ⟨257⟩, ⟨20399⟩⟩
def rawTerms : List Term := Proof.Events1073.exact274700RawTerms
def summary : Bound := (.finite 64377712650190257467641695830016)
def resultEvent : Nat := 274700
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274700.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult274695.owner)
    (rightOwner := SemanticResult274213.owner)
    (leftResult := 274695) (rightResult := 274213)
    (leftActual := SemanticResult274695.actual selector witness)
    (rightActual := SemanticResult274213.actual selector witness)
    (leftRaw := SemanticResult274695.rawTerms)
    (rightRaw := SemanticResult274213.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 32188807212483706889510625476608)
    (rightMaximum := 32188905437706550578131070353408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 274696) (rightBinding := 274697)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17532⟩) (rightExpression := ⟨20398⟩)
    (transferEvent := 274698) (summaryTransferEvent := 274699)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult274695.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult274213.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult274700

namespace SemanticResult274705
def owner : Owner := ⟨.program ⟨257⟩, ⟨23619⟩⟩
def rawTerms : List Term := Proof.Events1073.exact274705RawTerms
def summary : Bound := (.finite 96566716313119651734393211060224)
def resultEvent : Nat := 274705
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274705.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult274700.owner)
    (rightOwner := SemanticResult273731.owner)
    (leftResult := 274700) (rightResult := 273731)
    (leftActual := SemanticResult274700.actual selector witness)
    (rightActual := SemanticResult273731.actual selector witness)
    (leftRaw := SemanticResult274700.rawTerms)
    (rightRaw := SemanticResult273731.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 64377712650190257467641695830016)
    (rightMaximum := 32189003662929394266751515230208) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 274701) (rightBinding := 274702)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20399⟩) (rightExpression := ⟨23618⟩)
    (transferEvent := 274703) (summaryTransferEvent := 274704)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult274700.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult273731.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult274705

namespace SemanticResult274710
def owner : Owner := ⟨.program ⟨257⟩, ⟨33639⟩⟩
def rawTerms : List Term := Proof.Events1073.exact274710RawTerms
def summary : Bound := (.finite 128755916426494733378385616044032)
def resultEvent : Nat := 274710
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274710.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult274705.owner)
    (rightOwner := SemanticResult273249.owner)
    (leftResult := 274705) (rightResult := 273249)
    (leftActual := SemanticResult274705.actual selector witness)
    (rightActual := SemanticResult273249.actual selector witness)
    (leftRaw := SemanticResult274705.rawTerms)
    (rightRaw := SemanticResult273249.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 96566716313119651734393211060224)
    (rightMaximum := 32189200113375081643992404983808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 274706) (rightBinding := 274707)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23619⟩) (rightExpression := ⟨33638⟩)
    (transferEvent := 274708) (summaryTransferEvent := 274709)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult274705.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult273249.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult274710

namespace SemanticResult274715
def owner : Owner := ⟨.program ⟨257⟩, ⟨52699⟩⟩
def rawTerms : List Term := Proof.Events1073.exact274715RawTerms
def summary : Bound := (.finite 160945509440761189776859800535040)
def resultEvent : Nat := 274715
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274715.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult274710.owner)
    (rightOwner := SemanticResult272767.owner)
    (leftResult := 274710) (rightResult := 272767)
    (leftActual := SemanticResult274710.actual selector witness)
    (rightActual := SemanticResult272767.actual selector witness)
    (leftRaw := SemanticResult274710.rawTerms)
    (rightRaw := SemanticResult272767.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 128755916426494733378385616044032)
    (rightMaximum := 32189593014266456398474184491008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 274711) (rightBinding := 274712)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33639⟩) (rightExpression := ⟨52698⟩)
    (transferEvent := 274713) (summaryTransferEvent := 274714)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult274710.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult272767.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult274715

namespace SemanticResult274720
def owner : Owner := ⟨.program ⟨257⟩, ⟨55679⟩⟩
def rawTerms : List Term := Proof.Events1073.exact274720RawTerms
def summary : Bound := (.finite 193135298905473333552574874779648)
def resultEvent : Nat := 274720
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274720.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult274715.owner)
    (rightOwner := SemanticResult272285.owner)
    (leftResult := 274715) (rightResult := 272285)
    (leftActual := SemanticResult274715.actual selector witness)
    (rightActual := SemanticResult272285.actual selector witness)
    (leftRaw := SemanticResult274715.rawTerms)
    (rightRaw := SemanticResult272285.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 160945509440761189776859800535040)
    (rightMaximum := 32189789464712143775715074244608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 274716) (rightBinding := 274717)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨52699⟩) (rightExpression := ⟨55678⟩)
    (transferEvent := 274718) (summaryTransferEvent := 274719)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult274715.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult272285.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult274720

namespace SemanticResult274725
def owner : Owner := ⟨.program ⟨257⟩, ⟨58659⟩⟩
def rawTerms : List Term := Proof.Events1073.exact274725RawTerms
def summary : Bound := (.finite 225325481271076852082771728531456)
def resultEvent : Nat := 274725
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274725.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult274720.owner)
    (rightOwner := SemanticResult271803.owner)
    (leftResult := 274720) (rightResult := 271803)
    (leftActual := SemanticResult274720.actual selector witness)
    (rightActual := SemanticResult271803.actual selector witness)
    (leftRaw := SemanticResult274720.rawTerms)
    (rightRaw := SemanticResult271803.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 193135298905473333552574874779648)
    (rightMaximum := 32190182365603518530196853751808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 274721) (rightBinding := 274722)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55679⟩) (rightExpression := ⟨58658⟩)
    (transferEvent := 274723) (summaryTransferEvent := 274724)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult274720.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult271803.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult274725

namespace SemanticResult274730
def owner : Owner := ⟨.program ⟨257⟩, ⟨61639⟩⟩
def rawTerms : List Term := Proof.Events1073.exact274730RawTerms
def summary : Bound := (.finite 257515860087126057990209472036864)
def resultEvent : Nat := 274730
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274730.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult274725.owner)
    (rightOwner := SemanticResult271321.owner)
    (leftResult := 274725) (rightResult := 271321)
    (leftActual := SemanticResult274725.actual selector witness)
    (rightActual := SemanticResult271321.actual selector witness)
    (leftRaw := SemanticResult274725.rawTerms)
    (rightRaw := SemanticResult271321.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 225325481271076852082771728531456)
    (rightMaximum := 32190378816049205907437743505408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 274726) (rightBinding := 274727)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58659⟩) (rightExpression := ⟨61638⟩)
    (transferEvent := 274728) (summaryTransferEvent := 274729)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult274725.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult271321.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult274730

namespace SemanticResult274735
def owner : Owner := ⟨.program ⟨257⟩, ⟨64619⟩⟩
def rawTerms : List Term := Proof.Events1073.exact274735RawTerms
def summary : Bound := (.finite 289706631804066638652128995049472)
def resultEvent : Nat := 274735
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274735.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult274730.owner)
    (rightOwner := SemanticResult270839.owner)
    (leftResult := 274730) (rightResult := 270839)
    (leftActual := SemanticResult274730.actual selector witness)
    (rightActual := SemanticResult270839.actual selector witness)
    (leftRaw := SemanticResult274730.rawTerms)
    (rightRaw := SemanticResult270839.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 257515860087126057990209472036864)
    (rightMaximum := 32190771716940580661919523012608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 274731) (rightBinding := 274732)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61639⟩) (rightExpression := ⟨64618⟩)
    (transferEvent := 274733) (summaryTransferEvent := 274734)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult274730.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult270839.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult274735

namespace SemanticResult274740
def owner : Owner := ⟨.program ⟨257⟩, ⟨69524⟩⟩
def rawTerms : List Term := Proof.Events1073.exact274740RawTerms
def summary : Bound := (.finite 321897992872344281445771187322880)
def resultEvent : Nat := 274740
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274740.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult274735.owner)
    (rightOwner := SemanticResult270357.owner)
    (leftResult := 274735) (rightResult := 270357)
    (leftActual := SemanticResult274735.actual selector witness)
    (rightActual := SemanticResult270357.actual selector witness)
    (leftRaw := SemanticResult274735.rawTerms)
    (rightRaw := SemanticResult270357.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 289706631804066638652128995049472)
    (rightMaximum := 32191361068277642793642192273408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 274736) (rightBinding := 274737)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64619⟩) (rightExpression := ⟨69523⟩)
    (transferEvent := 274738) (summaryTransferEvent := 274739)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult274735.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult270357.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult274740

namespace SemanticResult274745
def owner : Owner := ⟨.program ⟨257⟩, ⟨69525⟩⟩
def rawTerms : List Term := Proof.Events1073.exact274745RawTerms
def summary : Bound := (.finite 354089550391067611616654269349888)
def resultEvent : Nat := 274745
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274745.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult274740.owner)
    (rightOwner := SemanticResult269875.owner)
    (leftResult := 274740) (rightResult := 269875)
    (leftActual := SemanticResult274740.actual selector witness)
    (rightActual := SemanticResult269875.actual selector witness)
    (leftRaw := SemanticResult274740.rawTerms)
    (rightRaw := SemanticResult269875.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 321897992872344281445771187322880)
    (rightMaximum := 32191557518723330170883082027008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 274741) (rightBinding := 274742)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69524⟩) (rightExpression := ⟨28085⟩)
    (transferEvent := 274743) (summaryTransferEvent := 274744)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult274740.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult269875.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult274745

namespace SemanticResult274750
def owner : Owner := ⟨.program ⟨257⟩, ⟨69526⟩⟩
def rawTerms : List Term := Proof.Events1073.exact274750RawTerms
def summary : Bound := (.finite 386281697261128003919260020637696)
def resultEvent : Nat := 274750
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult274750.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult274745.owner)
    (rightOwner := SemanticResult269393.owner)
    (leftResult := 274745) (rightResult := 269393)
    (leftActual := SemanticResult274745.actual selector witness)
    (rightActual := SemanticResult269393.actual selector witness)
    (leftRaw := SemanticResult274745.rawTerms)
    (rightRaw := SemanticResult269393.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 354089550391067611616654269349888)
    (rightMaximum := 32192146870060392302605751287808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 274746) (rightBinding := 274747)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69525⟩) (rightExpression := ⟨30765⟩)
    (transferEvent := 274748) (summaryTransferEvent := 274749)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult274745.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult269393.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult274750

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
