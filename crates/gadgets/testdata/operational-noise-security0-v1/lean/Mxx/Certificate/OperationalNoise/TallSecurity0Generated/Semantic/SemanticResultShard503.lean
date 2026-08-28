import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard503
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard026
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard097
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard098
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard502

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult70146
def owner : Owner := ⟨.program ⟨214⟩, ⟨11468⟩⟩
def rawTerms : List Term := Proof.Events274.exact70146RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 70146
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70146.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 70143) (survivorTransfer := 70144)
    (survivorEvent := 70145) (resultEvent := resultEvent)
    (rightCoefficientProducer := 11473)
    (owner := owner) (leftOwner := SemanticResult70140.owner)
    (rightOwner := SemanticResult11474.owner)
    (leftResult := 70140) (rightResult := 11474)
    (leftBinding := 70141) (rightBinding := 70142)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11467⟩) (rightExpression := ⟨93⟩)
    (leftActual := SemanticResult70140.actual selector witness)
    (rightActual := SemanticResult11474.actual selector witness)
    (leftRaw := SemanticResult70140.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨93⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound11473.actual selector witness)
    (survivorMagnitude := LeftBound70144.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult70140.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11474.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)
  · exact LeftBound70144.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult70146

namespace SemanticResult70154
def owner : Owner := ⟨.program ⟨214⟩, ⟨14201⟩⟩
def rawTerms : List Term := Proof.Events274.exact70154RawTerms
def summary : Bound := (.finite 14976)
def resultEvent : Nat := 70154
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70154.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨18, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70152.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge70152.frameStart)
    (owner := owner) (leftOwner := SemanticResult70146.owner)
    (rightOwner := SemanticResult3319.owner)
    (leftResult := 70146) (rightResult := 3319)
    (leftActual := SemanticResult70146.actual selector witness)
    (rightActual := SemanticResult3319.actual selector witness)
    (leftRaw := SemanticResult70146.rawTerms)
    (rightRaw := SemanticResult3319.rawTerms)
    (working := LeftOperatorMerge70152.working)
    (leftBinding := 70147) (rightBinding := 70148)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11468⟩) (rightExpression := ⟨14198⟩)
    (coefficientTransfer := 70149) (summaryTransfer := 70151)
    (rightCoefficientProducer := 3318)
    (rightSummaryTransfer := 70150)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨18, by decide⟩)
    (rightRecordedMaximum := 18)
    (rightSummaryMaximum := ⟨18, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge70152.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority3318.actual selector witness)
    (summaryMagnitude := LeftBound70151.actual selector witness)
    (reconstruction := LeftOperatorMerge70152.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult70146.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3319.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3318.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority3318.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge70152.operationAgreement
  · exact LeftBound70151.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70152.working summary) := by
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
end SemanticResult70154

namespace SemanticResult70159
def owner : Owner := ⟨.program ⟨214⟩, ⟨14202⟩⟩
def rawTerms : List Term := Proof.Events274.exact70159RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70159
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70159.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70158.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge70158.frameStart)
    (transferEvent := 70157) (owner := owner)
    (leftResult := 3319) (rightResult := 65295)
    (working := LeftOperatorMerge70158.working)
    (reconstruction := LeftOperatorMerge70158.reconstruction)
    (leftReference := .predecessor 0 70155 .coefficient) (rightReference := .predecessor 1 70156 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3319.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge70158.operationAgreement
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
end SemanticResult70159

namespace SemanticResult70164
def owner : Owner := ⟨.program ⟨214⟩, ⟨7177⟩⟩
def rawTerms : List Term := Proof.Events274.exact70164RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70164
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70164.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70163.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge70163.frameStart)
    (transferEvent := 70162) (owner := owner)
    (leftResult := 65165) (rightResult := 11523)
    (working := LeftOperatorMerge70163.working)
    (reconstruction := LeftOperatorMerge70163.reconstruction)
    (leftReference := .predecessor 0 70160 .coefficient) (rightReference := .predecessor 1 70161 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11523.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge70163.operationAgreement
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
end SemanticResult70164

namespace SemanticResult70168
def owner : Owner := ⟨.program ⟨214⟩, ⟨14203⟩⟩
def rawTerms : List Term := Proof.Events274.exact70168RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70168
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70168.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 70165) (rightBinding := 70166)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7177⟩) (rightExpression := ⟨14202⟩)
    (transferEvent := 70167)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult70164.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70159.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult70168

namespace SemanticResult70174
def owner : Owner := ⟨.program ⟨214⟩, ⟨14204⟩⟩
def rawTerms : List Term := Proof.Events274.exact70174RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 70174
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70174.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 70171) (survivorTransfer := 70172)
    (survivorEvent := 70173) (resultEvent := resultEvent)
    (rightCoefficientProducer := 11514)
    (owner := owner) (leftOwner := SemanticResult70168.owner)
    (rightOwner := SemanticResult11515.owner)
    (leftResult := 70168) (rightResult := 11515)
    (leftBinding := 70169) (rightBinding := 70170)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14203⟩) (rightExpression := ⟨73⟩)
    (leftActual := SemanticResult70168.actual selector witness)
    (rightActual := SemanticResult11515.actual selector witness)
    (leftRaw := SemanticResult70168.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨73⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound11514.actual selector witness)
    (survivorMagnitude := LeftBound70172.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult70168.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11515.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11514.derived selector witness)
  · exact LeftBound70172.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult70174

namespace SemanticResult70184
def owner : Owner := ⟨.program ⟨214⟩, ⟨14205⟩⟩
def rawTerms : List Term := Proof.Events274.exact70184RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 70184
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70184.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70180.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge70180.frameStart)
    (owner := owner) (leftOwner := SemanticResult70174.owner)
    (rightOwner := SemanticResult11512.owner)
    (leftResult := 70174) (rightResult := 11512)
    (leftActual := SemanticResult70174.actual selector witness)
    (rightActual := SemanticResult11512.actual selector witness)
    (leftRaw := SemanticResult70174.rawTerms)
    (rightRaw := SemanticResult11512.rawTerms)
    (working := LeftOperatorMerge70180.working)
    (leftBinding := 70175) (rightBinding := 70176)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14204⟩) (rightExpression := ⟨7853⟩)
    (coefficientTransfer := 70177) (summaryTransfer := 70179)
    (rightCoefficientProducer := 11511)
    (rightSummaryTransfer := 70178)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge70180.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound11511.actual selector witness)
    (summaryMagnitude := LeftBound70179.actual selector witness)
    (reconstruction := LeftOperatorMerge70180.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult70174.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11512.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11511.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound11511.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge70180.operationAgreement
  · exact LeftBound70179.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70180.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 70181 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6779⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6779⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge70180.working
    [{ coefficient := (-1), key := LeftRelationMerge70181.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge70181.frameStart
      LeftRelationMerge70181.owner (.relation 70181) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge70181.deltas
    rows := LeftRelationMerge70181.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge70180.working LeftRelationMerge70181.source
        (relationContext LeftRelationMerge70181.source
          LeftRelationMerge70181.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge70180.working, LeftRelationMerge70181.deltas,
    LeftRelationMerge70181.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 70181)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨14205⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge70180.working) (working := relationWorking0)
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
end SemanticResult70184

namespace SemanticResult70190
def owner : Owner := ⟨.program ⟨214⟩, ⟨14206⟩⟩
def rawTerms : List Term := Proof.Events274.exact70190RawTerms
def summary : Bound := (.finite 95435392)
def resultEvent : Nat := 70190
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70190.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge70188.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult70184.owner)
    (rightOwner := SemanticResult70154.owner)
    (leftResult := 70184) (rightResult := 70154)
    (leftActual := SemanticResult70184.actual selector witness)
    (rightActual := SemanticResult70154.actual selector witness)
    (leftRaw := SemanticResult70184.rawTerms)
    (rightRaw := SemanticResult70154.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 14976) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 70185) (rightBinding := 70186)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14205⟩) (rightExpression := ⟨14201⟩)
    (coefficientTransfer := 70187) (summaryTransfer := 70189)
    (base := LeftOperatorMerge70188.base)
    (reconstruction := LeftOperatorMerge70188.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult70184.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70154.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge70188.operationAgreement
  · rfl
  · decide
end SemanticResult70190

namespace SemanticResult70200
def owner : Owner := ⟨.program ⟨214⟩, ⟨26062⟩⟩
def rawTerms : List Term := Proof.Events274.exact70200RawTerms
def summary : Bound := (.finite 350249415606272)
def resultEvent : Nat := 70200
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70200.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95435392, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70196.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge70196.frameStart)
    (owner := owner) (leftOwner := SemanticResult70190.owner)
    (rightOwner := SemanticResult70126.owner)
    (leftResult := 70190) (rightResult := 70126)
    (leftActual := SemanticResult70190.actual selector witness)
    (rightActual := SemanticResult70126.actual selector witness)
    (leftRaw := SemanticResult70190.rawTerms)
    (rightRaw := SemanticResult70126.rawTerms)
    (working := LeftOperatorMerge70196.working)
    (leftBinding := 70191) (rightBinding := 70192)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14206⟩) (rightExpression := ⟨26061⟩)
    (coefficientTransfer := 70193) (summaryTransfer := 70195)
    (rightCoefficientProducer := 70125)
    (rightSummaryTransfer := 70194)
    (leftMaximum := ⟨95435392, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge70196.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority70125.actual selector witness)
    (summaryMagnitude := LeftBound70195.actual selector witness)
    (reconstruction := LeftOperatorMerge70196.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult70190.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70126.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70125.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority70125.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge70196.operationAgreement
  · exact LeftBound70195.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70196.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 70197 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23582⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23582⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge70196.working
    [{ coefficient := (-1), key := LeftRelationMerge70197.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge70197.frameStart
      LeftRelationMerge70197.owner (.relation 70197) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge70197.deltas
    rows := LeftRelationMerge70197.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge70196.working LeftRelationMerge70197.source
        (relationContext LeftRelationMerge70197.source
          LeftRelationMerge70197.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge70196.working, LeftRelationMerge70197.deltas,
    LeftRelationMerge70197.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 70197)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨26062⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge70196.working) (working := relationWorking0)
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
end SemanticResult70200

namespace SemanticResult70203
def owner : Owner := ⟨.program ⟨214⟩, ⟨19524⟩⟩
def rawTerms : List Term := Proof.Events274.exact70203RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70203
def producerEvent : Nat := 70202
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70203.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨15⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨15⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult70203

namespace SemanticResult70207
def owner : Owner := ⟨.program ⟨214⟩, ⟨19526⟩⟩
def rawTerms : List Term := Proof.Events274.exact70207RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70207
def producerEvent : Nat := 70206
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70207.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 70204 .coefficient) (.value (.predecessor 1 70205 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 70204 .coefficient) (.value (.predecessor 1 70205 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult70207

namespace SemanticResult70285
def owner : Owner := ⟨.program ⟨214⟩, ⟨11465⟩⟩
def rawTerms : List Term := Proof.Events274.exact70285RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70285
def producerEvent : Nat := 70284
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70285.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 70262, .finite 18, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult70285

namespace SemanticResult70288
def owner : Owner := ⟨.program ⟨214⟩, ⟨14198⟩⟩
def rawTerms : List Term := Proof.Events274.exact70288RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70288
def producerEvent : Nat := 70287
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70288.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 70262, .finite 18, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult70288

namespace SemanticResult70293
def owner : Owner := ⟨.program ⟨214⟩, ⟨14199⟩⟩
def rawTerms : List Term := Proof.Events274.exact70293RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70293
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70293.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70292.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge70292.frameStart)
    (transferEvent := 70291) (owner := owner)
    (leftResult := 70288) (rightResult := 70285)
    (working := LeftOperatorMerge70292.working)
    (reconstruction := LeftOperatorMerge70292.reconstruction)
    (leftReference := .predecessor 0 70289 .coefficient) (rightReference := .predecessor 1 70290 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult70288.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70285.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge70292.operationAgreement
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
end SemanticResult70293

namespace SemanticResult70304
def owner : Owner := ⟨.program ⟨214⟩, ⟨23582⟩⟩
def rawTerms : List Term := Proof.Events274.exact70304RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70304
def producerEvent : Nat := 70303
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70304.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 70262, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult70304

namespace SemanticResult70307
def owner : Owner := ⟨.program ⟨214⟩, ⟨26061⟩⟩
def rawTerms : List Term := Proof.Events274.exact70307RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70307
def producerEvent : Nat := 70306
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70307.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 70262, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult70307

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
