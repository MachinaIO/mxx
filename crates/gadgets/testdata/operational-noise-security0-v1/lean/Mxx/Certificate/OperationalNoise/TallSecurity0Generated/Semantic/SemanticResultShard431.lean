import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard431
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard401
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard405
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard409
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard413
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard416
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard420
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard424
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard427
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard430

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult59302
def owner : Owner := ⟨.program ⟨214⟩, ⟨26371⟩⟩
def rawTerms : List Term := Proof.Events231.exact59302RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 59302
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59302.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge59298.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge59298.frameStart)
    (transferEvent := 59297) (owner := owner)
    (leftResult := 59294) (rightResult := 59271)
    (working := LeftOperatorMerge59298.working)
    (reconstruction := LeftOperatorMerge59298.reconstruction)
    (leftReference := .predecessor 0 59295 .coefficient) (rightReference := .predecessor 1 59296 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult59294.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult59271.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge59298.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 59300 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26370⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14796⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23724⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23724⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge59298.working
    [{ coefficient := (-1), key := LeftRelationMerge59300.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge59300.frameStart
      LeftRelationMerge59300.owner (.relation 59300) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge59300.deltas
    rows := LeftRelationMerge59300.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge59298.working LeftRelationMerge59300.source
        (relationContext LeftRelationMerge59300.source
          LeftRelationMerge59300.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge59298.working, LeftRelationMerge59300.deltas,
    LeftRelationMerge59300.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 59300)
    (frameStart := 59220) (owner := ⟨.program ⟨214⟩, ⟨26371⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨14796⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26370⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26370⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge59298.working) (working := relationWorking0)
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
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (relationClaim0 selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult59302

namespace SemanticResult59305
def owner : Owner := ⟨.program ⟨214⟩, ⟨15268⟩⟩
def rawTerms : List Term := Proof.Events231.exact59305RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 59305
def producerEvent : Nat := 59304
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59305.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 59220, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult59305

namespace SemanticResult59310
def owner : Owner := ⟨.program ⟨214⟩, ⟨15269⟩⟩
def rawTerms : List Term := Proof.Events231.exact59310RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 59310
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59310.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge59309.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge59309.frameStart)
    (transferEvent := 59308) (owner := owner)
    (leftResult := 59282) (rightResult := 59305)
    (working := LeftOperatorMerge59309.working)
    (reconstruction := LeftOperatorMerge59309.reconstruction)
    (leftReference := .predecessor 0 59306 .coefficient) (rightReference := .predecessor 1 59307 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult59282.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult59305.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge59309.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult59310

namespace SemanticResult59313
def owner : Owner := ⟨.program ⟨214⟩, ⟨6709⟩⟩
def rawTerms : List Term := Proof.Events231.exact59313RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 59313
def producerEvent : Nat := 59312
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59313.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 59220, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult59313

namespace SemanticResult59317
def owner : Owner := ⟨.program ⟨214⟩, ⟨15270⟩⟩
def rawTerms : List Term := Proof.Events231.exact59317RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 59317
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59317.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 59314) (rightBinding := 59315)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6709⟩) (rightExpression := ⟨15269⟩)
    (transferEvent := 59316)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult59313.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult59310.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult59317

namespace SemanticResult59321
def owner : Owner := ⟨.program ⟨214⟩, ⟨26374⟩⟩
def rawTerms : List Term := Proof.Events231.exact59321RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 59321
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59321.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 59318) (rightBinding := 59319)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15270⟩) (rightExpression := ⟨26371⟩)
    (transferEvent := 59320)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult59317.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult59302.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult59321

namespace SemanticResult59330
def owner : Owner := ⟨.program ⟨214⟩, ⟨20399⟩⟩
def rawTerms : List Term := Proof.Events231.exact59330RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 59330
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59330.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge59165.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge59165.frameStart)
    (owner := owner) (leftOwner := SemanticResult50762.owner)
    (rightOwner := SemanticResult59159.owner)
    (leftResult := 50762) (rightResult := 59159)
    (leftActual := SemanticResult50762.actual selector witness)
    (rightActual := SemanticResult59159.actual selector witness)
    (leftRaw := SemanticResult50762.rawTerms)
    (rightRaw := SemanticResult59159.rawTerms)
    (working := LeftOperatorMerge59165.working)
    (leftBinding := 59160) (rightBinding := 59161)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5547⟩) (rightExpression := ⟨20398⟩)
    (coefficientTransfer := 59162) (summaryTransfer := 59164)
    (rightCoefficientProducer := 59158)
    (rightSummaryTransfer := 59163)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge59165.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound59158.actual selector witness)
    (summaryMagnitude := LeftBound59164.actual selector witness)
    (reconstruction := LeftOperatorMerge59165.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50762.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult59159.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59158.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound59158.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge59165.operationAgreement
  · exact LeftBound59164.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge59165.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 59325 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26370⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14796⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23724⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15268⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26370⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14796⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23724⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15268⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge59165.working
    [{ coefficient := (1), key := LeftRelationMerge59325.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge59325.frameStart
      LeftRelationMerge59325.owner (.relation 59325) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge59325.deltas
    rows := LeftRelationMerge59325.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge59165.working LeftRelationMerge59325.source
        (relationContext LeftRelationMerge59325.source
          LeftRelationMerge59325.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge59165.working, LeftRelationMerge59325.deltas,
    LeftRelationMerge59325.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 59325)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨20399⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20396⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20396⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge59165.working) (working := relationWorking0)
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
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (relationClaim0 selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult59330

namespace SemanticResult59337
def owner : Owner := ⟨.program ⟨214⟩, ⟨26373⟩⟩
def rawTerms : List Term := Proof.Events231.exact59337RawTerms
def summary : Bound := (.finite 1291889174379421642752)
def resultEvent : Nat := 59337
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59337.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge59334.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult59330.owner)
    (rightOwner := SemanticResult59152.owner)
    (leftResult := 59330) (rightResult := 59152)
    (leftActual := SemanticResult59330.actual selector witness)
    (rightActual := SemanticResult59152.actual selector witness)
    (leftRaw := SemanticResult59330.rawTerms)
    (rightRaw := SemanticResult59152.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1291889172568118132736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 59331) (rightBinding := 59332)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20399⟩) (rightExpression := ⟨26372⟩)
    (coefficientTransfer := 59333) (summaryTransfer := 59336)
    (base := LeftOperatorMerge59334.base)
    (reconstruction := LeftOperatorMerge59334.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult59330.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult59152.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge59334.operationAgreement
  · rfl
  · decide
end SemanticResult59337

namespace SemanticResult59342
def owner : Owner := ⟨.program ⟨214⟩, ⟨26581⟩⟩
def rawTerms : List Term := Proof.Events231.exact59342RawTerms
def summary : Bound := (.finite 2583789554981353578496)
def resultEvent : Nat := 59342
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59342.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult59337.owner)
    (rightOwner := SemanticResult58855.owner)
    (leftResult := 59337) (rightResult := 58855)
    (leftActual := SemanticResult59337.actual selector witness)
    (rightActual := SemanticResult58855.actual selector witness)
    (leftRaw := SemanticResult59337.rawTerms)
    (rightRaw := SemanticResult58855.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1291889174379421642752)
    (rightMaximum := 1291900380601931935744) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 59338) (rightBinding := 59339)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26373⟩) (rightExpression := ⟨26580⟩)
    (transferEvent := 59340) (summaryTransferEvent := 59341)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult59337.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult58855.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult59342

namespace SemanticResult59347
def owner : Owner := ⟨.program ⟨214⟩, ⟨26798⟩⟩
def rawTerms : List Term := Proof.Events231.exact59347RawTerms
def summary : Bound := (.finite 3875701141805795807232)
def resultEvent : Nat := 59347
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59347.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult59342.owner)
    (rightOwner := SemanticResult58373.owner)
    (leftResult := 59342) (rightResult := 58373)
    (leftActual := SemanticResult59342.actual selector witness)
    (rightActual := SemanticResult58373.actual selector witness)
    (leftRaw := SemanticResult59342.rawTerms)
    (rightRaw := SemanticResult58373.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2583789554981353578496)
    (rightMaximum := 1291911586824442228736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 59343) (rightBinding := 59344)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26581⟩) (rightExpression := ⟨26797⟩)
    (transferEvent := 59345) (summaryTransferEvent := 59346)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult59342.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult58373.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult59347

namespace SemanticResult59352
def owner : Owner := ⟨.program ⟨214⟩, ⟨27015⟩⟩
def rawTerms : List Term := Proof.Events231.exact59352RawTerms
def summary : Bound := (.finite 5167635141075258621952)
def resultEvent : Nat := 59352
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59352.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult59347.owner)
    (rightOwner := SemanticResult57891.owner)
    (leftResult := 59347) (rightResult := 57891)
    (leftActual := SemanticResult59347.actual selector witness)
    (rightActual := SemanticResult57891.actual selector witness)
    (leftRaw := SemanticResult59347.rawTerms)
    (rightRaw := SemanticResult57891.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3875701141805795807232)
    (rightMaximum := 1291933999269462814720) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 59348) (rightBinding := 59349)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26798⟩) (rightExpression := ⟨27014⟩)
    (transferEvent := 59350) (summaryTransferEvent := 59351)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult59347.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult57891.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult59352

namespace SemanticResult59357
def owner : Owner := ⟨.program ⟨214⟩, ⟨27232⟩⟩
def rawTerms : List Term := Proof.Events231.exact59357RawTerms
def summary : Bound := (.finite 6459613965234762608640)
def resultEvent : Nat := 59357
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59357.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult59352.owner)
    (rightOwner := SemanticResult57409.owner)
    (leftResult := 59352) (rightResult := 57409)
    (leftActual := SemanticResult59352.actual selector witness)
    (rightActual := SemanticResult57409.actual selector witness)
    (leftRaw := SemanticResult59352.rawTerms)
    (rightRaw := SemanticResult57409.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5167635141075258621952)
    (rightMaximum := 1291978824159503986688) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 59353) (rightBinding := 59354)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27015⟩) (rightExpression := ⟨27231⟩)
    (transferEvent := 59355) (summaryTransferEvent := 59356)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult59352.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult57409.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult59357

namespace SemanticResult59362
def owner : Owner := ⟨.program ⟨214⟩, ⟨27449⟩⟩
def rawTerms : List Term := Proof.Events231.exact59362RawTerms
def summary : Bound := (.finite 7751615201839287181312)
def resultEvent : Nat := 59362
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59362.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult59357.owner)
    (rightOwner := SemanticResult56927.owner)
    (leftResult := 59357) (rightResult := 56927)
    (leftActual := SemanticResult59357.actual selector witness)
    (rightActual := SemanticResult56927.actual selector witness)
    (leftRaw := SemanticResult59357.rawTerms)
    (rightRaw := SemanticResult56927.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 6459613965234762608640)
    (rightMaximum := 1292001236604524572672) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 59358) (rightBinding := 59359)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27232⟩) (rightExpression := ⟨27448⟩)
    (transferEvent := 59360) (summaryTransferEvent := 59361)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult59357.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56927.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult59362

namespace SemanticResult59367
def owner : Owner := ⟨.program ⟨214⟩, ⟨27666⟩⟩
def rawTerms : List Term := Proof.Events231.exact59367RawTerms
def summary : Bound := (.finite 9043661263333852925952)
def resultEvent : Nat := 59367
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59367.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult59362.owner)
    (rightOwner := SemanticResult56445.owner)
    (leftResult := 59362) (rightResult := 56445)
    (leftActual := SemanticResult59362.actual selector witness)
    (rightActual := SemanticResult56445.actual selector witness)
    (leftRaw := SemanticResult59362.rawTerms)
    (rightRaw := SemanticResult56445.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 7751615201839287181312)
    (rightMaximum := 1292046061494565744640) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 59363) (rightBinding := 59364)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27449⟩) (rightExpression := ⟨27665⟩)
    (transferEvent := 59365) (summaryTransferEvent := 59366)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult59362.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56445.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult59367

namespace SemanticResult59372
def owner : Owner := ⟨.program ⟨214⟩, ⟨27883⟩⟩
def rawTerms : List Term := Proof.Events231.exact59372RawTerms
def summary : Bound := (.finite 10335729737273439256576)
def resultEvent : Nat := 59372
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59372.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult59367.owner)
    (rightOwner := SemanticResult55963.owner)
    (leftResult := 59367) (rightResult := 55963)
    (leftActual := SemanticResult59367.actual selector witness)
    (rightActual := SemanticResult55963.actual selector witness)
    (leftRaw := SemanticResult59367.rawTerms)
    (rightRaw := SemanticResult55963.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 9043661263333852925952)
    (rightMaximum := 1292068473939586330624) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 59368) (rightBinding := 59369)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27666⟩) (rightExpression := ⟨27882⟩)
    (transferEvent := 59370) (summaryTransferEvent := 59371)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult59367.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult55963.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult59372

namespace SemanticResult59377
def owner : Owner := ⟨.program ⟨214⟩, ⟨28100⟩⟩
def rawTerms : List Term := Proof.Events231.exact59377RawTerms
def summary : Bound := (.finite 11627843036103066759168)
def resultEvent : Nat := 59377
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59377.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult59372.owner)
    (rightOwner := SemanticResult55481.owner)
    (leftResult := 59372) (rightResult := 55481)
    (leftActual := SemanticResult59372.actual selector witness)
    (rightActual := SemanticResult55481.actual selector witness)
    (leftRaw := SemanticResult59372.rawTerms)
    (rightRaw := SemanticResult55481.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 10335729737273439256576)
    (rightMaximum := 1292113298829627502592) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 59373) (rightBinding := 59374)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27883⟩) (rightExpression := ⟨28099⟩)
    (transferEvent := 59375) (summaryTransferEvent := 59376)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult59372.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult55481.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult59377

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
