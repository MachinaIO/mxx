import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard395
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard089
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard090
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard394

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult54551
def owner : Owner := ⟨.program ⟨214⟩, ⟨11643⟩⟩
def rawTerms : List Term := Proof.Events213.exact54551RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54551
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54551.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 54548) (rightBinding := 54549)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7275⟩) (rightExpression := ⟨11642⟩)
    (transferEvent := 54550)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54547.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult54542.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult54551

namespace SemanticResult54557
def owner : Owner := ⟨.program ⟨214⟩, ⟨11644⟩⟩
def rawTerms : List Term := Proof.Events213.exact54557RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 54557
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54557.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 54554) (survivorTransfer := 54555)
    (survivorEvent := 54556) (resultEvent := resultEvent)
    (rightCoefficientProducer := 10471)
    (owner := owner) (leftOwner := SemanticResult54551.owner)
    (rightOwner := SemanticResult10472.owner)
    (leftResult := 54551) (rightResult := 10472)
    (leftBinding := 54552) (rightBinding := 54553)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11643⟩) (rightExpression := ⟨95⟩)
    (leftActual := SemanticResult54551.actual selector witness)
    (rightActual := SemanticResult10472.actual selector witness)
    (leftRaw := SemanticResult54551.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨95⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound10471.actual selector witness)
    (survivorMagnitude := LeftBound54555.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54551.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10472.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10471.derived selector witness)
  · exact LeftBound54555.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult54557

namespace SemanticResult54565
def owner : Owner := ⟨.program ⟨214⟩, ⟨14653⟩⟩
def rawTerms : List Term := Proof.Events213.exact54565RawTerms
def summary : Bound := (.finite 23296)
def resultEvent : Nat := 54565
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54565.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨28, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54563.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge54563.frameStart)
    (owner := owner) (leftOwner := SemanticResult54557.owner)
    (rightOwner := SemanticResult2525.owner)
    (leftResult := 54557) (rightResult := 2525)
    (leftActual := SemanticResult54557.actual selector witness)
    (rightActual := SemanticResult2525.actual selector witness)
    (leftRaw := SemanticResult54557.rawTerms)
    (rightRaw := SemanticResult2525.rawTerms)
    (working := LeftOperatorMerge54563.working)
    (leftBinding := 54558) (rightBinding := 54559)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11644⟩) (rightExpression := ⟨14650⟩)
    (coefficientTransfer := 54560) (summaryTransfer := 54562)
    (rightCoefficientProducer := 2524)
    (rightSummaryTransfer := 54561)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨28, by decide⟩)
    (rightRecordedMaximum := 28)
    (rightSummaryMaximum := ⟨28, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge54563.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority2524.actual selector witness)
    (summaryMagnitude := LeftBound54562.actual selector witness)
    (reconstruction := LeftOperatorMerge54563.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54557.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2525.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2524.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority2524.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge54563.operationAgreement
  · exact LeftBound54562.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54563.working summary) := by
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
end SemanticResult54565

namespace SemanticResult54570
def owner : Owner := ⟨.program ⟨214⟩, ⟨14654⟩⟩
def rawTerms : List Term := Proof.Events213.exact54570RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54570
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54570.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54569.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge54569.frameStart)
    (transferEvent := 54568) (owner := owner)
    (leftResult := 2525) (rightResult := 50670)
    (working := LeftOperatorMerge54569.working)
    (reconstruction := LeftOperatorMerge54569.reconstruction)
    (leftReference := .predecessor 0 54566 .coefficient) (rightReference := .predecessor 1 54567 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2525.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge54569.operationAgreement
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
end SemanticResult54570

namespace SemanticResult54575
def owner : Owner := ⟨.program ⟨214⟩, ⟨7256⟩⟩
def rawTerms : List Term := Proof.Events213.exact54575RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54575
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54575.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54574.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge54574.frameStart)
    (transferEvent := 54573) (owner := owner)
    (leftResult := 50540) (rightResult := 10521)
    (working := LeftOperatorMerge54574.working)
    (reconstruction := LeftOperatorMerge54574.reconstruction)
    (leftReference := .predecessor 0 54571 .coefficient) (rightReference := .predecessor 1 54572 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10521.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge54574.operationAgreement
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
end SemanticResult54575

namespace SemanticResult54579
def owner : Owner := ⟨.program ⟨214⟩, ⟨14655⟩⟩
def rawTerms : List Term := Proof.Events213.exact54579RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54579
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54579.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 54576) (rightBinding := 54577)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7256⟩) (rightExpression := ⟨14654⟩)
    (transferEvent := 54578)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54575.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult54570.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult54579

namespace SemanticResult54585
def owner : Owner := ⟨.program ⟨214⟩, ⟨14656⟩⟩
def rawTerms : List Term := Proof.Events213.exact54585RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 54585
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54585.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 54582) (survivorTransfer := 54583)
    (survivorEvent := 54584) (resultEvent := resultEvent)
    (rightCoefficientProducer := 10512)
    (owner := owner) (leftOwner := SemanticResult54579.owner)
    (rightOwner := SemanticResult10513.owner)
    (leftResult := 54579) (rightResult := 10513)
    (leftBinding := 54580) (rightBinding := 54581)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14655⟩) (rightExpression := ⟨76⟩)
    (leftActual := SemanticResult54579.actual selector witness)
    (rightActual := SemanticResult10513.actual selector witness)
    (leftRaw := SemanticResult54579.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨76⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound10512.actual selector witness)
    (survivorMagnitude := LeftBound54583.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54579.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10513.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10512.derived selector witness)
  · exact LeftBound54583.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult54585

namespace SemanticResult54595
def owner : Owner := ⟨.program ⟨214⟩, ⟨14657⟩⟩
def rawTerms : List Term := Proof.Events213.exact54595RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 54595
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54595.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54591.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge54591.frameStart)
    (owner := owner) (leftOwner := SemanticResult54585.owner)
    (rightOwner := SemanticResult10510.owner)
    (leftResult := 54585) (rightResult := 10510)
    (leftActual := SemanticResult54585.actual selector witness)
    (rightActual := SemanticResult10510.actual selector witness)
    (leftRaw := SemanticResult54585.rawTerms)
    (rightRaw := SemanticResult10510.rawTerms)
    (working := LeftOperatorMerge54591.working)
    (leftBinding := 54586) (rightBinding := 54587)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14656⟩) (rightExpression := ⟨7859⟩)
    (coefficientTransfer := 54588) (summaryTransfer := 54590)
    (rightCoefficientProducer := 10509)
    (rightSummaryTransfer := 54589)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge54591.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound10509.actual selector witness)
    (summaryMagnitude := LeftBound54590.actual selector witness)
    (reconstruction := LeftOperatorMerge54591.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54585.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10510.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10509.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound10509.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge54591.operationAgreement
  · exact LeftBound54590.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54591.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 54592 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge54591.working
    [{ coefficient := (-1), key := LeftRelationMerge54592.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge54592.frameStart
      LeftRelationMerge54592.owner (.relation 54592) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge54592.deltas
    rows := LeftRelationMerge54592.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge54591.working LeftRelationMerge54592.source
        (relationContext LeftRelationMerge54592.source
          LeftRelationMerge54592.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge54591.working, LeftRelationMerge54592.deltas,
    LeftRelationMerge54592.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 54592)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨14657⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge54591.working) (working := relationWorking0)
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
end SemanticResult54595

namespace SemanticResult54601
def owner : Owner := ⟨.program ⟨214⟩, ⟨14658⟩⟩
def rawTerms : List Term := Proof.Events213.exact54601RawTerms
def summary : Bound := (.finite 95443712)
def resultEvent : Nat := 54601
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54601.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge54599.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult54595.owner)
    (rightOwner := SemanticResult54565.owner)
    (leftResult := 54595) (rightResult := 54565)
    (leftActual := SemanticResult54595.actual selector witness)
    (rightActual := SemanticResult54565.actual selector witness)
    (leftRaw := SemanticResult54595.rawTerms)
    (rightRaw := SemanticResult54565.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 23296) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 54596) (rightBinding := 54597)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14657⟩) (rightExpression := ⟨14653⟩)
    (coefficientTransfer := 54598) (summaryTransfer := 54600)
    (base := LeftOperatorMerge54599.base)
    (reconstruction := LeftOperatorMerge54599.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54595.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult54565.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge54599.operationAgreement
  · rfl
  · decide
end SemanticResult54601

namespace SemanticResult54611
def owner : Owner := ⟨.program ⟨214⟩, ⟨26226⟩⟩
def rawTerms : List Term := Proof.Events213.exact54611RawTerms
def summary : Bound := (.finite 350279950139392)
def resultEvent : Nat := 54611
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54611.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95443712, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54607.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge54607.frameStart)
    (owner := owner) (leftOwner := SemanticResult54601.owner)
    (rightOwner := SemanticResult54537.owner)
    (leftResult := 54601) (rightResult := 54537)
    (leftActual := SemanticResult54601.actual selector witness)
    (rightActual := SemanticResult54537.actual selector witness)
    (leftRaw := SemanticResult54601.rawTerms)
    (rightRaw := SemanticResult54537.rawTerms)
    (working := LeftOperatorMerge54607.working)
    (leftBinding := 54602) (rightBinding := 54603)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14658⟩) (rightExpression := ⟨26225⟩)
    (coefficientTransfer := 54604) (summaryTransfer := 54606)
    (rightCoefficientProducer := 54536)
    (rightSummaryTransfer := 54605)
    (leftMaximum := ⟨95443712, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge54607.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority54536.actual selector witness)
    (summaryMagnitude := LeftBound54606.actual selector witness)
    (reconstruction := LeftOperatorMerge54607.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54601.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult54537.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54536.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority54536.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge54607.operationAgreement
  · exact LeftBound54606.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54607.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 54608 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23670⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge54607.working
    [{ coefficient := (-1), key := LeftRelationMerge54608.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge54608.frameStart
      LeftRelationMerge54608.owner (.relation 54608) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge54608.deltas
    rows := LeftRelationMerge54608.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge54607.working LeftRelationMerge54608.source
        (relationContext LeftRelationMerge54608.source
          LeftRelationMerge54608.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge54607.working, LeftRelationMerge54608.deltas,
    LeftRelationMerge54608.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 54608)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨26226⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge54607.working) (working := relationWorking0)
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
end SemanticResult54611

namespace SemanticResult54614
def owner : Owner := ⟨.program ⟨214⟩, ⟨19676⟩⟩
def rawTerms : List Term := Proof.Events213.exact54614RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54614
def producerEvent : Nat := 54613
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54614.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨17⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨17⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult54614

namespace SemanticResult54618
def owner : Owner := ⟨.program ⟨214⟩, ⟨19678⟩⟩
def rawTerms : List Term := Proof.Events213.exact54618RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54618
def producerEvent : Nat := 54617
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54618.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 54615 .coefficient) (.value (.predecessor 1 54616 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 54615 .coefficient) (.value (.predecessor 1 54616 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult54618

namespace SemanticResult54696
def owner : Owner := ⟨.program ⟨214⟩, ⟨11641⟩⟩
def rawTerms : List Term := Proof.Events213.exact54696RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54696
def producerEvent : Nat := 54695
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54696.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 54673, .finite 28, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult54696

namespace SemanticResult54699
def owner : Owner := ⟨.program ⟨214⟩, ⟨14650⟩⟩
def rawTerms : List Term := Proof.Events213.exact54699RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54699
def producerEvent : Nat := 54698
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54699.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 54673, .finite 28, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult54699

namespace SemanticResult54704
def owner : Owner := ⟨.program ⟨214⟩, ⟨14651⟩⟩
def rawTerms : List Term := Proof.Events213.exact54704RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54704
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54704.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54703.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge54703.frameStart)
    (transferEvent := 54702) (owner := owner)
    (leftResult := 54699) (rightResult := 54696)
    (working := LeftOperatorMerge54703.working)
    (reconstruction := LeftOperatorMerge54703.reconstruction)
    (leftReference := .predecessor 0 54700 .coefficient) (rightReference := .predecessor 1 54701 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult54699.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult54696.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge54703.operationAgreement
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
end SemanticResult54704

namespace SemanticResult54715
def owner : Owner := ⟨.program ⟨214⟩, ⟨23670⟩⟩
def rawTerms : List Term := Proof.Events213.exact54715RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54715
def producerEvent : Nat := 54714
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54715.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 54673, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult54715

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
