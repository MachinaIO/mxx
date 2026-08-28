import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard102
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard001
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard055
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard101

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult12009
def owner : Owner := ⟨.program ⟨214⟩, ⟨7849⟩⟩
def rawTerms : List Term := Proof.Events046.exact12009RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12009
def producerEvent : Nat := 12008
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12009.actual selector witness
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
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult12009

namespace SemanticResult12013
def owner : Owner := ⟨.program ⟨214⟩, ⟨7850⟩⟩
def rawTerms : List Term := Proof.Events046.exact12013RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12013
def producerEvent : Nat := 12012
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12013.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 12010 .coefficient) (.value (.predecessor 1 12011 .coefficient)), 0, .finite 8192, .scale (.predecessor 0 12010 .coefficient) (.value (.predecessor 1 12011 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult12013

namespace SemanticResult12016
def owner : Owner := ⟨.program ⟨214⟩, ⟨72⟩⟩
def rawTerms : List Term := Proof.Events046.exact12016RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12016
def producerEvent : Nat := 12015
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12016.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 12014 .coefficient), 0, .finite 26, .identity (.predecessor 0 12014 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult12016

namespace SemanticResult12021
def owner : Owner := ⟨.program ⟨214⟩, ⟨14030⟩⟩
def rawTerms : List Term := Proof.Events046.exact12021RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12021
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12021.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge12020.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge12020.frameStart)
    (transferEvent := 12019) (owner := owner)
    (leftResult := 307) (rightResult := 6449)
    (working := LeftOperatorMerge12020.working)
    (reconstruction := LeftOperatorMerge12020.reconstruction)
    (leftReference := .predecessor 0 12017 .coefficient) (rightReference := .predecessor 1 12018 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult307.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6449.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge12020.operationAgreement
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
end SemanticResult12021

namespace SemanticResult12024
def owner : Owner := ⟨.program ⟨214⟩, ⟨6758⟩⟩
def rawTerms : List Term := Proof.Events046.exact12024RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12024
def producerEvent : Nat := 12023
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12024.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 12022 .coefficient), 0, .large, .identity (.predecessor 0 12022 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult12024

namespace SemanticResult12029
def owner : Owner := ⟨.program ⟨214⟩, ⟨7366⟩⟩
def rawTerms : List Term := Proof.Events046.exact12029RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12029
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12029.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge12028.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge12028.frameStart)
    (transferEvent := 12027) (owner := owner)
    (leftResult := 6314) (rightResult := 12024)
    (working := LeftOperatorMerge12028.working)
    (reconstruction := LeftOperatorMerge12028.reconstruction)
    (leftReference := .predecessor 0 12025 .coefficient) (rightReference := .predecessor 1 12026 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult6314.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12024.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge12028.operationAgreement
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
end SemanticResult12029

namespace SemanticResult12033
def owner : Owner := ⟨.program ⟨214⟩, ⟨14031⟩⟩
def rawTerms : List Term := Proof.Events047.exact12033RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12033
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12033.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 12030) (rightBinding := 12031)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7366⟩) (rightExpression := ⟨14030⟩)
    (transferEvent := 12032)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12029.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12021.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult12033

namespace SemanticResult12039
def owner : Owner := ⟨.program ⟨214⟩, ⟨14032⟩⟩
def rawTerms : List Term := Proof.Events047.exact12039RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 12039
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12039.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 12036) (survivorTransfer := 12037)
    (survivorEvent := 12038) (resultEvent := resultEvent)
    (rightCoefficientProducer := 12015)
    (owner := owner) (leftOwner := SemanticResult12033.owner)
    (rightOwner := SemanticResult12016.owner)
    (leftResult := 12033) (rightResult := 12016)
    (leftBinding := 12034) (rightBinding := 12035)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14031⟩) (rightExpression := ⟨72⟩)
    (leftActual := SemanticResult12033.actual selector witness)
    (rightActual := SemanticResult12016.actual selector witness)
    (leftRaw := SemanticResult12033.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨72⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound12015.actual selector witness)
    (survivorMagnitude := LeftBound12037.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12033.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12016.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12015.derived selector witness)
  · exact LeftBound12037.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult12039

namespace SemanticResult12049
def owner : Owner := ⟨.program ⟨214⟩, ⟨14033⟩⟩
def rawTerms : List Term := Proof.Events047.exact12049RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 12049
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12049.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge12045.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge12045.frameStart)
    (owner := owner) (leftOwner := SemanticResult12039.owner)
    (rightOwner := SemanticResult12013.owner)
    (leftResult := 12039) (rightResult := 12013)
    (leftActual := SemanticResult12039.actual selector witness)
    (rightActual := SemanticResult12013.actual selector witness)
    (leftRaw := SemanticResult12039.rawTerms)
    (rightRaw := SemanticResult12013.rawTerms)
    (working := LeftOperatorMerge12045.working)
    (leftBinding := 12040) (rightBinding := 12041)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14032⟩) (rightExpression := ⟨7850⟩)
    (coefficientTransfer := 12042) (summaryTransfer := 12044)
    (rightCoefficientProducer := 12012)
    (rightSummaryTransfer := 12043)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge12045.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound12012.actual selector witness)
    (summaryMagnitude := LeftBound12044.actual selector witness)
    (reconstruction := LeftOperatorMerge12045.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12039.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12013.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12012.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound12012.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge12045.operationAgreement
  · exact LeftBound12044.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge12045.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 12046 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge12045.working
    [{ coefficient := (-1), key := LeftRelationMerge12046.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge12046.frameStart
      LeftRelationMerge12046.owner (.relation 12046) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge12046.deltas
    rows := LeftRelationMerge12046.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge12045.working LeftRelationMerge12046.source
        (relationContext LeftRelationMerge12046.source
          LeftRelationMerge12046.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge12045.working, LeftRelationMerge12046.deltas,
    LeftRelationMerge12046.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 12046)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨14033⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge12045.working) (working := relationWorking0)
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
end SemanticResult12049

namespace SemanticResult12055
def owner : Owner := ⟨.program ⟨214⟩, ⟨14034⟩⟩
def rawTerms : List Term := Proof.Events047.exact12055RawTerms
def summary : Bound := (.finite 95433728)
def resultEvent : Nat := 12055
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12055.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge12053.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult12049.owner)
    (rightOwner := SemanticResult12006.owner)
    (leftResult := 12049) (rightResult := 12006)
    (leftActual := SemanticResult12049.actual selector witness)
    (rightActual := SemanticResult12006.actual selector witness)
    (leftRaw := SemanticResult12049.rawTerms)
    (rightRaw := SemanticResult12006.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 13312) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 12050) (rightBinding := 12051)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14033⟩) (rightExpression := ⟨14029⟩)
    (coefficientTransfer := 12052) (summaryTransfer := 12054)
    (base := LeftOperatorMerge12053.base)
    (reconstruction := LeftOperatorMerge12053.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12049.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12006.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge12053.operationAgreement
  · rfl
  · decide
end SemanticResult12055

namespace SemanticResult12065
def owner : Owner := ⟨.program ⟨214⟩, ⟨26010⟩⟩
def rawTerms : List Term := Proof.Events047.exact12065RawTerms
def summary : Bound := (.finite 350243308699648)
def resultEvent : Nat := 12065
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12065.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95433728, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge12061.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge12061.frameStart)
    (owner := owner) (leftOwner := SemanticResult12055.owner)
    (rightOwner := SemanticResult11972.owner)
    (leftResult := 12055) (rightResult := 11972)
    (leftActual := SemanticResult12055.actual selector witness)
    (rightActual := SemanticResult11972.actual selector witness)
    (leftRaw := SemanticResult12055.rawTerms)
    (rightRaw := SemanticResult11972.rawTerms)
    (working := LeftOperatorMerge12061.working)
    (leftBinding := 12056) (rightBinding := 12057)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14034⟩) (rightExpression := ⟨26009⟩)
    (coefficientTransfer := 12058) (summaryTransfer := 12060)
    (rightCoefficientProducer := 11971)
    (rightSummaryTransfer := 12059)
    (leftMaximum := ⟨95433728, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge12061.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority11971.actual selector witness)
    (summaryMagnitude := LeftBound12060.actual selector witness)
    (reconstruction := LeftOperatorMerge12061.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult12055.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11972.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11971.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority11971.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge12061.operationAgreement
  · exact LeftBound12060.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge12061.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 12062 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨26009⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11401⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23550⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23550⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge12061.working
    [{ coefficient := (-1), key := LeftRelationMerge12062.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge12062.frameStart
      LeftRelationMerge12062.owner (.relation 12062) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge12062.deltas
    rows := LeftRelationMerge12062.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge12061.working LeftRelationMerge12062.source
        (relationContext LeftRelationMerge12062.source
          LeftRelationMerge12062.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge12061.working, LeftRelationMerge12062.deltas,
    LeftRelationMerge12062.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 12062)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨26010⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11401⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26009⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26009⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge12061.working) (working := relationWorking0)
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
end SemanticResult12065

namespace SemanticResult12068
def owner : Owner := ⟨.program ⟨214⟩, ⟨19472⟩⟩
def rawTerms : List Term := Proof.Events047.exact12068RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12068
def producerEvent : Nat := 12067
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12068.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨14⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨14⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult12068

namespace SemanticResult12072
def owner : Owner := ⟨.program ⟨214⟩, ⟨19474⟩⟩
def rawTerms : List Term := Proof.Events047.exact12072RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12072
def producerEvent : Nat := 12071
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12072.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 12069 .coefficient) (.value (.predecessor 1 12070 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 12069 .coefficient) (.value (.predecessor 1 12070 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult12072

namespace SemanticResult12150
def owner : Owner := ⟨.program ⟨214⟩, ⟨11401⟩⟩
def rawTerms : List Term := Proof.Events047.exact12150RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12150
def producerEvent : Nat := 12149
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12150.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 12127, .finite 16, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult12150

namespace SemanticResult12153
def owner : Owner := ⟨.program ⟨214⟩, ⟨14026⟩⟩
def rawTerms : List Term := Proof.Events047.exact12153RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12153
def producerEvent : Nat := 12152
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12153.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 12127, .finite 16, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult12153

namespace SemanticResult12158
def owner : Owner := ⟨.program ⟨214⟩, ⟨14027⟩⟩
def rawTerms : List Term := Proof.Events047.exact12158RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 12158
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult12158.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge12157.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge12157.frameStart)
    (transferEvent := 12156) (owner := owner)
    (leftResult := 12153) (rightResult := 12150)
    (working := LeftOperatorMerge12157.working)
    (reconstruction := LeftOperatorMerge12157.reconstruction)
    (leftReference := .predecessor 0 12154 .coefficient) (rightReference := .predecessor 1 12155 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult12153.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12150.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge12157.operationAgreement
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
end SemanticResult12158

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
