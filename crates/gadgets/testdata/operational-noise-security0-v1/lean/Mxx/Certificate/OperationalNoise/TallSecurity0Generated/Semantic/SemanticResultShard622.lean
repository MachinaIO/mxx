import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard622
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard033
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard117
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard118
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard566
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard621

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult87151
def owner : Owner := ⟨.program ⟨214⟩, ⟨10982⟩⟩
def rawTerms : List Term := Proof.Events340.exact87151RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 87151
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87151.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 87148) (survivorTransfer := 87149)
    (survivorEvent := 87150) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13978)
    (owner := owner) (leftOwner := SemanticResult87145.owner)
    (rightOwner := SemanticResult13979.owner)
    (leftResult := 87145) (rightResult := 13979)
    (leftBinding := 87146) (rightBinding := 87147)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10981⟩) (rightExpression := ⟨88⟩)
    (leftActual := SemanticResult87145.actual selector witness)
    (rightActual := SemanticResult13979.actual selector witness)
    (leftRaw := SemanticResult87145.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨88⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13978.actual selector witness)
    (survivorMagnitude := LeftBound87149.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult87145.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13979.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13978.derived selector witness)
  · exact LeftBound87149.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult87151

namespace SemanticResult87159
def owner : Owner := ⟨.program ⟨214⟩, ⟨10983⟩⟩
def rawTerms : List Term := Proof.Events340.exact87159RawTerms
def summary : Bound := (.finite 3328)
def resultEvent : Nat := 87159
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87159.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨4, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge87157.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge87157.frameStart)
    (owner := owner) (leftOwner := SemanticResult87151.owner)
    (rightOwner := SemanticResult4176.owner)
    (leftResult := 87151) (rightResult := 4176)
    (leftActual := SemanticResult87151.actual selector witness)
    (rightActual := SemanticResult4176.actual selector witness)
    (leftRaw := SemanticResult87151.rawTerms)
    (rightRaw := SemanticResult4176.rawTerms)
    (working := LeftOperatorMerge87157.working)
    (leftBinding := 87152) (rightBinding := 87153)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10982⟩) (rightExpression := ⟨10842⟩)
    (coefficientTransfer := 87154) (summaryTransfer := 87156)
    (rightCoefficientProducer := 4175)
    (rightSummaryTransfer := 87155)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨4, by decide⟩)
    (rightRecordedMaximum := 4)
    (rightSummaryMaximum := ⟨4, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge87157.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority4175.actual selector witness)
    (summaryMagnitude := LeftBound87156.actual selector witness)
    (reconstruction := LeftOperatorMerge87157.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult87151.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4176.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4175.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority4175.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge87157.operationAgreement
  · exact LeftBound87156.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge87157.working summary) := by
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
end SemanticResult87159

namespace SemanticResult87164
def owner : Owner := ⟨.program ⟨214⟩, ⟨10843⟩⟩
def rawTerms : List Term := Proof.Events340.exact87164RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 87164
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87164.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge87163.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge87163.frameStart)
    (transferEvent := 87162) (owner := owner)
    (leftResult := 4176) (rightResult := 79920)
    (working := LeftOperatorMerge87163.working)
    (reconstruction := LeftOperatorMerge87163.reconstruction)
    (leftReference := .predecessor 0 87160 .coefficient) (rightReference := .predecessor 1 87161 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4176.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79920.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge87163.operationAgreement
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
end SemanticResult87164

namespace SemanticResult87169
def owner : Owner := ⟨.program ⟨214⟩, ⟨7247⟩⟩
def rawTerms : List Term := Proof.Events340.exact87169RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 87169
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87169.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge87168.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge87168.frameStart)
    (transferEvent := 87167) (owner := owner)
    (leftResult := 79790) (rightResult := 14028)
    (working := LeftOperatorMerge87168.working)
    (reconstruction := LeftOperatorMerge87168.reconstruction)
    (leftReference := .predecessor 0 87165 .coefficient) (rightReference := .predecessor 1 87166 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14028.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge87168.operationAgreement
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
end SemanticResult87169

namespace SemanticResult87173
def owner : Owner := ⟨.program ⟨214⟩, ⟨10844⟩⟩
def rawTerms : List Term := Proof.Events340.exact87173RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 87173
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87173.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 87170) (rightBinding := 87171)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7247⟩) (rightExpression := ⟨10843⟩)
    (transferEvent := 87172)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult87169.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult87164.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult87173

namespace SemanticResult87179
def owner : Owner := ⟨.program ⟨214⟩, ⟨10845⟩⟩
def rawTerms : List Term := Proof.Events340.exact87179RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 87179
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87179.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 87176) (survivorTransfer := 87177)
    (survivorEvent := 87178) (resultEvent := resultEvent)
    (rightCoefficientProducer := 14019)
    (owner := owner) (leftOwner := SemanticResult87173.owner)
    (rightOwner := SemanticResult14020.owner)
    (leftResult := 87173) (rightResult := 14020)
    (leftBinding := 87174) (rightBinding := 87175)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10844⟩) (rightExpression := ⟨105⟩)
    (leftActual := SemanticResult87173.actual selector witness)
    (rightActual := SemanticResult14020.actual selector witness)
    (leftRaw := SemanticResult87173.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨105⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound14019.actual selector witness)
    (survivorMagnitude := LeftBound87177.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult87173.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14020.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14019.derived selector witness)
  · exact LeftBound87177.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult87179

namespace SemanticResult87189
def owner : Owner := ⟨.program ⟨214⟩, ⟨10846⟩⟩
def rawTerms : List Term := Proof.Events340.exact87189RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 87189
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87189.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge87185.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge87185.frameStart)
    (owner := owner) (leftOwner := SemanticResult87179.owner)
    (rightOwner := SemanticResult14017.owner)
    (leftResult := 87179) (rightResult := 14017)
    (leftActual := SemanticResult87179.actual selector witness)
    (rightActual := SemanticResult14017.actual selector witness)
    (leftRaw := SemanticResult87179.rawTerms)
    (rightRaw := SemanticResult14017.rawTerms)
    (working := LeftOperatorMerge87185.working)
    (leftBinding := 87180) (rightBinding := 87181)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10845⟩) (rightExpression := ⟨7838⟩)
    (coefficientTransfer := 87182) (summaryTransfer := 87184)
    (rightCoefficientProducer := 14016)
    (rightSummaryTransfer := 87183)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge87185.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound14016.actual selector witness)
    (summaryMagnitude := LeftBound87184.actual selector witness)
    (reconstruction := LeftOperatorMerge87185.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult87179.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14017.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14016.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound14016.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge87185.operationAgreement
  · exact LeftBound87184.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge87185.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 87186 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10842⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge87185.working
    [{ coefficient := (-1), key := LeftRelationMerge87186.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge87186.frameStart
      LeftRelationMerge87186.owner (.relation 87186) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge87186.deltas
    rows := LeftRelationMerge87186.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge87185.working LeftRelationMerge87186.source
        (relationContext LeftRelationMerge87186.source
          LeftRelationMerge87186.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge87185.working, LeftRelationMerge87186.deltas,
    LeftRelationMerge87186.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 87186)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10846⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10842⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge87185.working) (working := relationWorking0)
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
end SemanticResult87189

namespace SemanticResult87195
def owner : Owner := ⟨.program ⟨214⟩, ⟨10984⟩⟩
def rawTerms : List Term := Proof.Events340.exact87195RawTerms
def summary : Bound := (.finite 95423744)
def resultEvent : Nat := 87195
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87195.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge87193.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult87189.owner)
    (rightOwner := SemanticResult87159.owner)
    (leftResult := 87189) (rightResult := 87159)
    (leftActual := SemanticResult87189.actual selector witness)
    (rightActual := SemanticResult87159.actual selector witness)
    (leftRaw := SemanticResult87189.rawTerms)
    (rightRaw := SemanticResult87159.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 3328) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 87190) (rightBinding := 87191)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10846⟩) (rightExpression := ⟨10983⟩)
    (coefficientTransfer := 87192) (summaryTransfer := 87194)
    (base := LeftOperatorMerge87193.base)
    (reconstruction := LeftOperatorMerge87193.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult87189.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult87159.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge87193.operationAgreement
  · rfl
  · decide
end SemanticResult87195

namespace SemanticResult87205
def owner : Owner := ⟨.program ⟨214⟩, ⟨25066⟩⟩
def rawTerms : List Term := Proof.Events340.exact87205RawTerms
def summary : Bound := (.finite 350206667259904)
def resultEvent : Nat := 87205
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87205.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95423744, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge87201.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge87201.frameStart)
    (owner := owner) (leftOwner := SemanticResult87195.owner)
    (rightOwner := SemanticResult87131.owner)
    (leftResult := 87195) (rightResult := 87131)
    (leftActual := SemanticResult87195.actual selector witness)
    (rightActual := SemanticResult87131.actual selector witness)
    (leftRaw := SemanticResult87195.rawTerms)
    (rightRaw := SemanticResult87131.rawTerms)
    (working := LeftOperatorMerge87201.working)
    (leftBinding := 87196) (rightBinding := 87197)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10984⟩) (rightExpression := ⟨25065⟩)
    (coefficientTransfer := 87198) (summaryTransfer := 87200)
    (rightCoefficientProducer := 87130)
    (rightSummaryTransfer := 87199)
    (leftMaximum := ⟨95423744, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge87201.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority87130.actual selector witness)
    (summaryMagnitude := LeftBound87200.actual selector witness)
    (reconstruction := LeftOperatorMerge87201.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult87195.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult87131.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87130.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority87130.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge87201.operationAgreement
  · exact LeftBound87200.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge87201.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 87202 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25065⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23038⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23038⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge87201.working
    [{ coefficient := (-1), key := LeftRelationMerge87202.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge87202.frameStart
      LeftRelationMerge87202.owner (.relation 87202) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge87202.deltas
    rows := LeftRelationMerge87202.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge87201.working LeftRelationMerge87202.source
        (relationContext LeftRelationMerge87202.source
          LeftRelationMerge87202.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge87201.working, LeftRelationMerge87202.deltas,
    LeftRelationMerge87202.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 87202)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25066⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25065⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25065⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge87201.working) (working := relationWorking0)
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
end SemanticResult87205

namespace SemanticResult87208
def owner : Owner := ⟨.program ⟨214⟩, ⟨19168⟩⟩
def rawTerms : List Term := Proof.Events340.exact87208RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 87208
def producerEvent : Nat := 87207
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87208.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨9⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨9⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult87208

namespace SemanticResult87212
def owner : Owner := ⟨.program ⟨214⟩, ⟨19170⟩⟩
def rawTerms : List Term := Proof.Events340.exact87212RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 87212
def producerEvent : Nat := 87211
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87212.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 87209 .coefficient) (.value (.predecessor 1 87210 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 87209 .coefficient) (.value (.predecessor 1 87210 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult87212

namespace SemanticResult87290
def owner : Owner := ⟨.program ⟨214⟩, ⟨10977⟩⟩
def rawTerms : List Term := Proof.Events340.exact87290RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 87290
def producerEvent : Nat := 87289
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87290.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 87267, .finite 4, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult87290

namespace SemanticResult87293
def owner : Owner := ⟨.program ⟨214⟩, ⟨10842⟩⟩
def rawTerms : List Term := Proof.Events340.exact87293RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 87293
def producerEvent : Nat := 87292
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87293.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 87267, .finite 4, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult87293

namespace SemanticResult87298
def owner : Owner := ⟨.program ⟨214⟩, ⟨10978⟩⟩
def rawTerms : List Term := Proof.Events341.exact87298RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 87298
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87298.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge87297.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge87297.frameStart)
    (transferEvent := 87296) (owner := owner)
    (leftResult := 87293) (rightResult := 87290)
    (working := LeftOperatorMerge87297.working)
    (reconstruction := LeftOperatorMerge87297.reconstruction)
    (leftReference := .predecessor 0 87294 .coefficient) (rightReference := .predecessor 1 87295 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult87293.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult87290.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge87297.operationAgreement
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
end SemanticResult87298

namespace SemanticResult87309
def owner : Owner := ⟨.program ⟨214⟩, ⟨23038⟩⟩
def rawTerms : List Term := Proof.Events341.exact87309RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 87309
def producerEvent : Nat := 87308
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87309.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 87267, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult87309

namespace SemanticResult87312
def owner : Owner := ⟨.program ⟨214⟩, ⟨25065⟩⟩
def rawTerms : List Term := Proof.Events341.exact87312RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 87312
def producerEvent : Nat := 87311
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult87312.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 87267, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult87312

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
