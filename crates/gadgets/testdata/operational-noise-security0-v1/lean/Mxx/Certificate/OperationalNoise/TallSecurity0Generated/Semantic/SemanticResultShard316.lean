import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard316
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard015
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard113
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard114
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard265
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard315

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult42768
def owner : Owner := ⟨.program ⟨214⟩, ⟨27246⟩⟩
def rawTerms : List Term := Proof.Events167.exact42768RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 42768
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42768.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 42765) (rightBinding := 42766)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15637⟩) (rightExpression := ⟨27242⟩)
    (transferEvent := 42767)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult42764.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult42749.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult42768

namespace SemanticResult42777
def owner : Owner := ⟨.program ⟨214⟩, ⟨20979⟩⟩
def rawTerms : List Term := Proof.Events167.exact42777RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 42777
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42777.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge42612.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge42612.frameStart)
    (owner := owner) (leftOwner := SemanticResult36137.owner)
    (rightOwner := SemanticResult42606.owner)
    (leftResult := 36137) (rightResult := 42606)
    (leftActual := SemanticResult36137.actual selector witness)
    (rightActual := SemanticResult42606.actual selector witness)
    (leftRaw := SemanticResult36137.rawTerms)
    (rightRaw := SemanticResult42606.rawTerms)
    (working := LeftOperatorMerge42612.working)
    (leftBinding := 42607) (rightBinding := 42608)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5553⟩) (rightExpression := ⟨20978⟩)
    (coefficientTransfer := 42609) (summaryTransfer := 42611)
    (rightCoefficientProducer := 42605)
    (rightSummaryTransfer := 42610)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge42612.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound42605.actual selector witness)
    (summaryMagnitude := LeftBound42611.actual selector witness)
    (reconstruction := LeftOperatorMerge42612.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36137.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult42606.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42605.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound42605.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge42612.operationAgreement
  · exact LeftBound42611.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge42612.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 42772 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27241⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27241⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23979⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge42612.working
    [{ coefficient := (1), key := LeftRelationMerge42772.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge42772.frameStart
      LeftRelationMerge42772.owner (.relation 42772) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge42772.deltas
    rows := LeftRelationMerge42772.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge42612.working LeftRelationMerge42772.source
        (relationContext LeftRelationMerge42772.source
          LeftRelationMerge42772.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge42612.working, LeftRelationMerge42772.deltas,
    LeftRelationMerge42772.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 42772)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨20979⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20976⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20976⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge42612.working) (working := relationWorking0)
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
end SemanticResult42777

namespace SemanticResult42784
def owner : Owner := ⟨.program ⟨214⟩, ⟨27244⟩⟩
def rawTerms : List Term := Proof.Events167.exact42784RawTerms
def summary : Bound := (.finite 1291978824159503986688)
def resultEvent : Nat := 42784
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42784.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge42781.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult42777.owner)
    (rightOwner := SemanticResult42599.owner)
    (leftResult := 42777) (rightResult := 42599)
    (leftActual := SemanticResult42777.actual selector witness)
    (rightActual := SemanticResult42599.actual selector witness)
    (leftRaw := SemanticResult42777.rawTerms)
    (rightRaw := SemanticResult42599.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1291978822348200476672) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 42778) (rightBinding := 42779)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20979⟩) (rightExpression := ⟨27243⟩)
    (coefficientTransfer := 42780) (summaryTransfer := 42783)
    (base := LeftOperatorMerge42781.base)
    (reconstruction := LeftOperatorMerge42781.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult42777.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult42599.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge42781.operationAgreement
  · rfl
  · decide
end SemanticResult42784

namespace SemanticResult42791
def owner : Owner := ⟨.program ⟨214⟩, ⟨23916⟩⟩
def rawTerms : List Term := Proof.Events167.exact42791RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 42791
def producerEvent : Nat := 42790
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42791.actual selector witness
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
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult42791

namespace SemanticResult42794
def owner : Owner := ⟨.program ⟨214⟩, ⟨27024⟩⟩
def rawTerms : List Term := Proof.Events167.exact42794RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 42794
def producerEvent : Nat := 42793
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42794.actual selector witness
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
end SemanticResult42794

namespace SemanticResult42801
def owner : Owner := ⟨.program ⟨214⟩, ⟨23168⟩⟩
def rawTerms : List Term := Proof.Events167.exact42801RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 42801
def producerEvent : Nat := 42800
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42801.actual selector witness
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
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult42801

namespace SemanticResult42804
def owner : Owner := ⟨.program ⟨214⟩, ⟨25306⟩⟩
def rawTerms : List Term := Proof.Events167.exact42804RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 42804
def producerEvent : Nat := 42803
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42804.actual selector witness
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
end SemanticResult42804

namespace SemanticResult42809
def owner : Owner := ⟨.program ⟨214⟩, ⟨11142⟩⟩
def rawTerms : List Term := Proof.Events167.exact42809RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 42809
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42809.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge42808.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge42808.frameStart)
    (transferEvent := 42807) (owner := owner)
    (leftResult := 1912) (rightResult := 36045)
    (working := LeftOperatorMerge42808.working)
    (reconstruction := LeftOperatorMerge42808.reconstruction)
    (leftReference := .predecessor 0 42805 .coefficient) (rightReference := .predecessor 1 42806 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1912.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge42808.operationAgreement
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
end SemanticResult42809

namespace SemanticResult42814
def owner : Owner := ⟨.program ⟨214⟩, ⟨7307⟩⟩
def rawTerms : List Term := Proof.Events167.exact42814RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 42814
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42814.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge42813.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge42813.frameStart)
    (transferEvent := 42812) (owner := owner)
    (leftResult := 35915) (rightResult := 13486)
    (working := LeftOperatorMerge42813.working)
    (reconstruction := LeftOperatorMerge42813.reconstruction)
    (leftReference := .predecessor 0 42810 .coefficient) (rightReference := .predecessor 1 42811 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13486.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge42813.operationAgreement
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
end SemanticResult42814

namespace SemanticResult42818
def owner : Owner := ⟨.program ⟨214⟩, ⟨11143⟩⟩
def rawTerms : List Term := Proof.Events167.exact42818RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 42818
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42818.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 42815) (rightBinding := 42816)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7307⟩) (rightExpression := ⟨11142⟩)
    (transferEvent := 42817)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult42814.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult42809.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult42818

namespace SemanticResult42824
def owner : Owner := ⟨.program ⟨214⟩, ⟨11144⟩⟩
def rawTerms : List Term := Proof.Events167.exact42824RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 42824
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42824.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 42821) (survivorTransfer := 42822)
    (survivorEvent := 42823) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13477)
    (owner := owner) (leftOwner := SemanticResult42818.owner)
    (rightOwner := SemanticResult13478.owner)
    (leftResult := 42818) (rightResult := 13478)
    (leftBinding := 42819) (rightBinding := 42820)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11143⟩) (rightExpression := ⟨89⟩)
    (leftActual := SemanticResult42818.actual selector witness)
    (rightActual := SemanticResult13478.actual selector witness)
    (leftRaw := SemanticResult42818.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨89⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13477.actual selector witness)
    (survivorMagnitude := LeftBound42822.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult42818.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13478.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13477.derived selector witness)
  · exact LeftBound42822.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult42824

namespace SemanticResult42832
def owner : Owner := ⟨.program ⟨214⟩, ⟨12184⟩⟩
def rawTerms : List Term := Proof.Events167.exact42832RawTerms
def summary : Bound := (.finite 4992)
def resultEvent : Nat := 42832
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42832.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨6, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge42830.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge42830.frameStart)
    (owner := owner) (leftOwner := SemanticResult42824.owner)
    (rightOwner := SemanticResult1915.owner)
    (leftResult := 42824) (rightResult := 1915)
    (leftActual := SemanticResult42824.actual selector witness)
    (rightActual := SemanticResult1915.actual selector witness)
    (leftRaw := SemanticResult42824.rawTerms)
    (rightRaw := SemanticResult1915.rawTerms)
    (working := LeftOperatorMerge42830.working)
    (leftBinding := 42825) (rightBinding := 42826)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11144⟩) (rightExpression := ⟨12181⟩)
    (coefficientTransfer := 42827) (summaryTransfer := 42829)
    (rightCoefficientProducer := 1914)
    (rightSummaryTransfer := 42828)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨6, by decide⟩)
    (rightRecordedMaximum := 6)
    (rightSummaryMaximum := ⟨6, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge42830.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1914.actual selector witness)
    (summaryMagnitude := LeftBound42829.actual selector witness)
    (reconstruction := LeftOperatorMerge42830.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult42824.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1915.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1914.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1914.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge42830.operationAgreement
  · exact LeftBound42829.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge42830.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult42832

namespace SemanticResult42837
def owner : Owner := ⟨.program ⟨214⟩, ⟨12185⟩⟩
def rawTerms : List Term := Proof.Events167.exact42837RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 42837
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42837.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge42836.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge42836.frameStart)
    (transferEvent := 42835) (owner := owner)
    (leftResult := 1915) (rightResult := 36045)
    (working := LeftOperatorMerge42836.working)
    (reconstruction := LeftOperatorMerge42836.reconstruction)
    (leftReference := .predecessor 0 42833 .coefficient) (rightReference := .predecessor 1 42834 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge42836.operationAgreement
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
end SemanticResult42837

namespace SemanticResult42842
def owner : Owner := ⟨.program ⟨214⟩, ⟨7324⟩⟩
def rawTerms : List Term := Proof.Events167.exact42842RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 42842
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42842.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge42841.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge42841.frameStart)
    (transferEvent := 42840) (owner := owner)
    (leftResult := 35915) (rightResult := 13527)
    (working := LeftOperatorMerge42841.working)
    (reconstruction := LeftOperatorMerge42841.reconstruction)
    (leftReference := .predecessor 0 42838 .coefficient) (rightReference := .predecessor 1 42839 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13527.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge42841.operationAgreement
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
end SemanticResult42842

namespace SemanticResult42846
def owner : Owner := ⟨.program ⟨214⟩, ⟨12186⟩⟩
def rawTerms : List Term := Proof.Events167.exact42846RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 42846
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42846.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 42843) (rightBinding := 42844)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7324⟩) (rightExpression := ⟨12185⟩)
    (transferEvent := 42845)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult42842.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult42837.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult42846

namespace SemanticResult42852
def owner : Owner := ⟨.program ⟨214⟩, ⟨12187⟩⟩
def rawTerms : List Term := Proof.Events167.exact42852RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 42852
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult42852.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 42849) (survivorTransfer := 42850)
    (survivorEvent := 42851) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13518)
    (owner := owner) (leftOwner := SemanticResult42846.owner)
    (rightOwner := SemanticResult13519.owner)
    (leftResult := 42846) (rightResult := 13519)
    (leftBinding := 42847) (rightBinding := 42848)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12186⟩) (rightExpression := ⟨106⟩)
    (leftActual := SemanticResult42846.actual selector witness)
    (rightActual := SemanticResult13519.actual selector witness)
    (leftRaw := SemanticResult42846.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨106⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13518.actual selector witness)
    (survivorMagnitude := LeftBound42850.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult42846.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13519.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13518.derived selector witness)
  · exact LeftBound42850.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult42852

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
