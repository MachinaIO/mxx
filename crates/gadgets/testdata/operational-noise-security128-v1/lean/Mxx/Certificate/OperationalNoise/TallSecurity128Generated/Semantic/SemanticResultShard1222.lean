import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1222
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1157
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1185
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1189
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1192
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1196
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1200
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1203
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1207
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1211
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1214
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1218
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1221

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult172293
def owner : Owner := ⟨.program ⟨257⟩, ⟨16100⟩⟩
def rawTerms : List Term := Proof.Events673.exact172293RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 172293
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172293.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge172292.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge172292.frameStart)
    (transferEvent := 172291) (owner := owner)
    (leftResult := 172265) (rightResult := 172288)
    (working := LeftOperatorMerge172292.working)
    (reconstruction := LeftOperatorMerge172292.reconstruction)
    (leftReference := .predecessor 0 172289 .coefficient) (rightReference := .predecessor 1 172290 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult172265.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult172288.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge172292.operationAgreement
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
end SemanticResult172293

namespace SemanticResult172296
def owner : Owner := ⟨.program ⟨257⟩, ⟨7198⟩⟩
def rawTerms : List Term := Proof.Events673.exact172296RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 172296
def producerEvent : Nat := 172295
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172296.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 172203, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult172296

namespace SemanticResult172300
def owner : Owner := ⟨.program ⟨257⟩, ⟨16101⟩⟩
def rawTerms : List Term := Proof.Events673.exact172300RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 172300
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172300.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 172297) (rightBinding := 172298)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7198⟩) (rightExpression := ⟨16100⟩)
    (transferEvent := 172299)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult172296.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult172293.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult172300

namespace SemanticResult172304
def owner : Owner := ⟨.program ⟨257⟩, ⟨17877⟩⟩
def rawTerms : List Term := Proof.Events673.exact172304RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 172304
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172304.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 172301) (rightBinding := 172302)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16101⟩) (rightExpression := ⟨17874⟩)
    (transferEvent := 172303)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult172300.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult172285.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult172304

namespace SemanticResult172313
def owner : Owner := ⟨.program ⟨257⟩, ⟨16679⟩⟩
def rawTerms : List Term := Proof.Events673.exact172313RawTerms
def summary : Bound := (.finite 202072841853861888)
def resultEvent : Nat := 172313
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172313.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1376256
      (.finite ⟨26, by decide⟩)
      (.finite ⟨5647228698, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge172148.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge172148.frameStart)
    (owner := owner) (leftOwner := SemanticResult163745.owner)
    (rightOwner := SemanticResult172142.owner)
    (leftResult := 163745) (rightResult := 172142)
    (leftActual := SemanticResult163745.actual selector witness)
    (rightActual := SemanticResult172142.actual selector witness)
    (leftRaw := SemanticResult163745.rawTerms)
    (rightRaw := SemanticResult172142.rawTerms)
    (working := LeftOperatorMerge172148.working)
    (leftBinding := 172143) (rightBinding := 172144)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6466⟩) (rightExpression := ⟨16678⟩)
    (coefficientTransfer := 172145) (summaryTransfer := 172147)
    (rightCoefficientProducer := 172141)
    (rightSummaryTransfer := 172146)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨5647228698, by decide⟩)
    (rightRecordedMaximum := 5647228698)
    (rightSummaryMaximum := ⟨5647228698, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1376256)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge172148.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound172141.actual selector witness)
    (summaryMagnitude := LeftBound172147.actual selector witness)
    (reconstruction := LeftOperatorMerge172148.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult163745.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult172142.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172141.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound172141.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge172148.operationAgreement
  · exact LeftBound172147.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge172148.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 172308 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17037⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17037⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16099⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge172148.working
    [{ coefficient := (1), key := LeftRelationMerge172308.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge172308.frameStart
      LeftRelationMerge172308.owner (.relation 172308) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge172308.deltas
    rows := LeftRelationMerge172308.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge172148.working LeftRelationMerge172308.source
        (relationContext LeftRelationMerge172308.source
          LeftRelationMerge172308.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge172148.working, LeftRelationMerge172308.deltas,
    LeftRelationMerge172308.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 172308)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨16679⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16676⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16676⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge172148.working) (working := relationWorking0)
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
end SemanticResult172313

namespace SemanticResult172320
def owner : Owner := ⟨.program ⟨257⟩, ⟨17876⟩⟩
def rawTerms : List Term := Proof.Events673.exact172320RawTerms
def summary : Bound := (.finite 32188807212483706889510625476608)
def resultEvent : Nat := 172320
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172320.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge172317.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult172313.owner)
    (rightOwner := SemanticResult172135.owner)
    (leftResult := 172313) (rightResult := 172135)
    (leftActual := SemanticResult172313.actual selector witness)
    (rightActual := SemanticResult172135.actual selector witness)
    (leftRaw := SemanticResult172313.rawTerms)
    (rightRaw := SemanticResult172135.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32188807212483504816668771614720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 172314) (rightBinding := 172315)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16679⟩) (rightExpression := ⟨17875⟩)
    (coefficientTransfer := 172316) (summaryTransfer := 172319)
    (base := LeftOperatorMerge172317.base)
    (reconstruction := LeftOperatorMerge172317.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult172313.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult172135.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge172317.operationAgreement
  · rfl
  · decide
end SemanticResult172320

namespace SemanticResult172325
def owner : Owner := ⟨.program ⟨257⟩, ⟨20780⟩⟩
def rawTerms : List Term := Proof.Events673.exact172325RawTerms
def summary : Bound := (.finite 64377712650190257467641695830016)
def resultEvent : Nat := 172325
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172325.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult172320.owner)
    (rightOwner := SemanticResult171838.owner)
    (leftResult := 172320) (rightResult := 171838)
    (leftActual := SemanticResult172320.actual selector witness)
    (rightActual := SemanticResult171838.actual selector witness)
    (leftRaw := SemanticResult172320.rawTerms)
    (rightRaw := SemanticResult171838.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 32188807212483706889510625476608)
    (rightMaximum := 32188905437706550578131070353408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 172321) (rightBinding := 172322)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17876⟩) (rightExpression := ⟨20779⟩)
    (transferEvent := 172323) (summaryTransferEvent := 172324)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult172320.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult171838.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult172325

namespace SemanticResult172330
def owner : Owner := ⟨.program ⟨257⟩, ⟨24000⟩⟩
def rawTerms : List Term := Proof.Events673.exact172330RawTerms
def summary : Bound := (.finite 96566716313119651734393211060224)
def resultEvent : Nat := 172330
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172330.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult172325.owner)
    (rightOwner := SemanticResult171356.owner)
    (leftResult := 172325) (rightResult := 171356)
    (leftActual := SemanticResult172325.actual selector witness)
    (rightActual := SemanticResult171356.actual selector witness)
    (leftRaw := SemanticResult172325.rawTerms)
    (rightRaw := SemanticResult171356.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 64377712650190257467641695830016)
    (rightMaximum := 32189003662929394266751515230208) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 172326) (rightBinding := 172327)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20780⟩) (rightExpression := ⟨23999⟩)
    (transferEvent := 172328) (summaryTransferEvent := 172329)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult172325.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult171356.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult172330

namespace SemanticResult172335
def owner : Owner := ⟨.program ⟨257⟩, ⟨34020⟩⟩
def rawTerms : List Term := Proof.Events673.exact172335RawTerms
def summary : Bound := (.finite 128755916426494733378385616044032)
def resultEvent : Nat := 172335
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172335.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult172330.owner)
    (rightOwner := SemanticResult170874.owner)
    (leftResult := 172330) (rightResult := 170874)
    (leftActual := SemanticResult172330.actual selector witness)
    (rightActual := SemanticResult170874.actual selector witness)
    (leftRaw := SemanticResult172330.rawTerms)
    (rightRaw := SemanticResult170874.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 96566716313119651734393211060224)
    (rightMaximum := 32189200113375081643992404983808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 172331) (rightBinding := 172332)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨24000⟩) (rightExpression := ⟨34019⟩)
    (transferEvent := 172333) (summaryTransferEvent := 172334)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult172330.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult170874.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult172335

namespace SemanticResult172340
def owner : Owner := ⟨.program ⟨257⟩, ⟨53080⟩⟩
def rawTerms : List Term := Proof.Events673.exact172340RawTerms
def summary : Bound := (.finite 160945509440761189776859800535040)
def resultEvent : Nat := 172340
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172340.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult172335.owner)
    (rightOwner := SemanticResult170392.owner)
    (leftResult := 172335) (rightResult := 170392)
    (leftActual := SemanticResult172335.actual selector witness)
    (rightActual := SemanticResult170392.actual selector witness)
    (leftRaw := SemanticResult172335.rawTerms)
    (rightRaw := SemanticResult170392.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 128755916426494733378385616044032)
    (rightMaximum := 32189593014266456398474184491008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 172336) (rightBinding := 172337)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨34020⟩) (rightExpression := ⟨53079⟩)
    (transferEvent := 172338) (summaryTransferEvent := 172339)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult172335.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult170392.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult172340

namespace SemanticResult172345
def owner : Owner := ⟨.program ⟨257⟩, ⟨56060⟩⟩
def rawTerms : List Term := Proof.Events673.exact172345RawTerms
def summary : Bound := (.finite 193135298905473333552574874779648)
def resultEvent : Nat := 172345
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172345.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult172340.owner)
    (rightOwner := SemanticResult169910.owner)
    (leftResult := 172340) (rightResult := 169910)
    (leftActual := SemanticResult172340.actual selector witness)
    (rightActual := SemanticResult169910.actual selector witness)
    (leftRaw := SemanticResult172340.rawTerms)
    (rightRaw := SemanticResult169910.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 160945509440761189776859800535040)
    (rightMaximum := 32189789464712143775715074244608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 172341) (rightBinding := 172342)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53080⟩) (rightExpression := ⟨56059⟩)
    (transferEvent := 172343) (summaryTransferEvent := 172344)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult172340.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult169910.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult172345

namespace SemanticResult172350
def owner : Owner := ⟨.program ⟨257⟩, ⟨59040⟩⟩
def rawTerms : List Term := Proof.Events673.exact172350RawTerms
def summary : Bound := (.finite 225325481271076852082771728531456)
def resultEvent : Nat := 172350
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172350.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult172345.owner)
    (rightOwner := SemanticResult169428.owner)
    (leftResult := 172345) (rightResult := 169428)
    (leftActual := SemanticResult172345.actual selector witness)
    (rightActual := SemanticResult169428.actual selector witness)
    (leftRaw := SemanticResult172345.rawTerms)
    (rightRaw := SemanticResult169428.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 193135298905473333552574874779648)
    (rightMaximum := 32190182365603518530196853751808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 172346) (rightBinding := 172347)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56060⟩) (rightExpression := ⟨59039⟩)
    (transferEvent := 172348) (summaryTransferEvent := 172349)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult172345.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult169428.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult172350

namespace SemanticResult172355
def owner : Owner := ⟨.program ⟨257⟩, ⟨62020⟩⟩
def rawTerms : List Term := Proof.Events673.exact172355RawTerms
def summary : Bound := (.finite 257515860087126057990209472036864)
def resultEvent : Nat := 172355
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172355.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult172350.owner)
    (rightOwner := SemanticResult168946.owner)
    (leftResult := 172350) (rightResult := 168946)
    (leftActual := SemanticResult172350.actual selector witness)
    (rightActual := SemanticResult168946.actual selector witness)
    (leftRaw := SemanticResult172350.rawTerms)
    (rightRaw := SemanticResult168946.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 225325481271076852082771728531456)
    (rightMaximum := 32190378816049205907437743505408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 172351) (rightBinding := 172352)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59040⟩) (rightExpression := ⟨62019⟩)
    (transferEvent := 172353) (summaryTransferEvent := 172354)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult172350.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult168946.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult172355

namespace SemanticResult172360
def owner : Owner := ⟨.program ⟨257⟩, ⟨65000⟩⟩
def rawTerms : List Term := Proof.Events673.exact172360RawTerms
def summary : Bound := (.finite 289706631804066638652128995049472)
def resultEvent : Nat := 172360
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172360.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult172355.owner)
    (rightOwner := SemanticResult168464.owner)
    (leftResult := 172355) (rightResult := 168464)
    (leftActual := SemanticResult172355.actual selector witness)
    (rightActual := SemanticResult168464.actual selector witness)
    (leftRaw := SemanticResult172355.rawTerms)
    (rightRaw := SemanticResult168464.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 257515860087126057990209472036864)
    (rightMaximum := 32190771716940580661919523012608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 172356) (rightBinding := 172357)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62020⟩) (rightExpression := ⟨64999⟩)
    (transferEvent := 172358) (summaryTransferEvent := 172359)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult172355.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult168464.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult172360

namespace SemanticResult172365
def owner : Owner := ⟨.program ⟨257⟩, ⟨70497⟩⟩
def rawTerms : List Term := Proof.Events673.exact172365RawTerms
def summary : Bound := (.finite 321897992872344281445771187322880)
def resultEvent : Nat := 172365
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172365.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult172360.owner)
    (rightOwner := SemanticResult167982.owner)
    (leftResult := 172360) (rightResult := 167982)
    (leftActual := SemanticResult172360.actual selector witness)
    (rightActual := SemanticResult167982.actual selector witness)
    (leftRaw := SemanticResult172360.rawTerms)
    (rightRaw := SemanticResult167982.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 289706631804066638652128995049472)
    (rightMaximum := 32191361068277642793642192273408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 172361) (rightBinding := 172362)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65000⟩) (rightExpression := ⟨70496⟩)
    (transferEvent := 172363) (summaryTransferEvent := 172364)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult172360.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult167982.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult172365

namespace SemanticResult172370
def owner : Owner := ⟨.program ⟨257⟩, ⟨70498⟩⟩
def rawTerms : List Term := Proof.Events673.exact172370RawTerms
def summary : Bound := (.finite 354089550391067611616654269349888)
def resultEvent : Nat := 172370
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult172370.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult172365.owner)
    (rightOwner := SemanticResult167500.owner)
    (leftResult := 172365) (rightResult := 167500)
    (leftActual := SemanticResult172365.actual selector witness)
    (rightActual := SemanticResult167500.actual selector witness)
    (leftRaw := SemanticResult172365.rawTerms)
    (rightRaw := SemanticResult167500.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 321897992872344281445771187322880)
    (rightMaximum := 32191557518723330170883082027008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 172366) (rightBinding := 172367)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70497⟩) (rightExpression := ⟨28392⟩)
    (transferEvent := 172368) (summaryTransferEvent := 172369)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult172365.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult167500.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult172370

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
