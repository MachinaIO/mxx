import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard596
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard032
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard089
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard090
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard566
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard595

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult83776
def owner : Owner := ⟨.program ⟨214⟩, ⟨11638⟩⟩
def rawTerms : List Term := Proof.Events327.exact83776RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83776
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83776.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83775.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge83775.frameStart)
    (transferEvent := 83774) (owner := owner)
    (leftResult := 4012) (rightResult := 79920)
    (working := LeftOperatorMerge83775.working)
    (reconstruction := LeftOperatorMerge83775.reconstruction)
    (leftReference := .predecessor 0 83772 .coefficient) (rightReference := .predecessor 1 83773 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4012.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79920.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge83775.operationAgreement
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
end SemanticResult83776

namespace SemanticResult83781
def owner : Owner := ⟨.program ⟨214⟩, ⟨7237⟩⟩
def rawTerms : List Term := Proof.Events327.exact83781RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83781
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83781.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83780.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge83780.frameStart)
    (transferEvent := 83779) (owner := owner)
    (leftResult := 79790) (rightResult := 10480)
    (working := LeftOperatorMerge83780.working)
    (reconstruction := LeftOperatorMerge83780.reconstruction)
    (leftReference := .predecessor 0 83777 .coefficient) (rightReference := .predecessor 1 83778 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10480.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge83780.operationAgreement
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
end SemanticResult83781

namespace SemanticResult83785
def owner : Owner := ⟨.program ⟨214⟩, ⟨11639⟩⟩
def rawTerms : List Term := Proof.Events327.exact83785RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83785
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83785.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 83782) (rightBinding := 83783)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7237⟩) (rightExpression := ⟨11638⟩)
    (transferEvent := 83784)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83781.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult83776.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult83785

namespace SemanticResult83791
def owner : Owner := ⟨.program ⟨214⟩, ⟨11640⟩⟩
def rawTerms : List Term := Proof.Events327.exact83791RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 83791
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83791.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 83788) (survivorTransfer := 83789)
    (survivorEvent := 83790) (resultEvent := resultEvent)
    (rightCoefficientProducer := 10471)
    (owner := owner) (leftOwner := SemanticResult83785.owner)
    (rightOwner := SemanticResult10472.owner)
    (leftResult := 83785) (rightResult := 10472)
    (leftBinding := 83786) (rightBinding := 83787)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11639⟩) (rightExpression := ⟨95⟩)
    (leftActual := SemanticResult83785.actual selector witness)
    (rightActual := SemanticResult10472.actual selector witness)
    (leftRaw := SemanticResult83785.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨95⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound10471.actual selector witness)
    (survivorMagnitude := LeftBound83789.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83785.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10472.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10471.derived selector witness)
  · exact LeftBound83789.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult83791

namespace SemanticResult83799
def owner : Owner := ⟨.program ⟨214⟩, ⟨14644⟩⟩
def rawTerms : List Term := Proof.Events327.exact83799RawTerms
def summary : Bound := (.finite 23296)
def resultEvent : Nat := 83799
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83799.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨28, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83797.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge83797.frameStart)
    (owner := owner) (leftOwner := SemanticResult83791.owner)
    (rightOwner := SemanticResult4015.owner)
    (leftResult := 83791) (rightResult := 4015)
    (leftActual := SemanticResult83791.actual selector witness)
    (rightActual := SemanticResult4015.actual selector witness)
    (leftRaw := SemanticResult83791.rawTerms)
    (rightRaw := SemanticResult4015.rawTerms)
    (working := LeftOperatorMerge83797.working)
    (leftBinding := 83792) (rightBinding := 83793)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11640⟩) (rightExpression := ⟨14641⟩)
    (coefficientTransfer := 83794) (summaryTransfer := 83796)
    (rightCoefficientProducer := 4014)
    (rightSummaryTransfer := 83795)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨28, by decide⟩)
    (rightRecordedMaximum := 28)
    (rightSummaryMaximum := ⟨28, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge83797.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority4014.actual selector witness)
    (summaryMagnitude := LeftBound83796.actual selector witness)
    (reconstruction := LeftOperatorMerge83797.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83791.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4015.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4014.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority4014.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge83797.operationAgreement
  · exact LeftBound83796.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83797.working summary) := by
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
end SemanticResult83799

namespace SemanticResult83804
def owner : Owner := ⟨.program ⟨214⟩, ⟨14645⟩⟩
def rawTerms : List Term := Proof.Events327.exact83804RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83804
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83804.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83803.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge83803.frameStart)
    (transferEvent := 83802) (owner := owner)
    (leftResult := 4015) (rightResult := 79920)
    (working := LeftOperatorMerge83803.working)
    (reconstruction := LeftOperatorMerge83803.reconstruction)
    (leftReference := .predecessor 0 83800 .coefficient) (rightReference := .predecessor 1 83801 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4015.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79920.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge83803.operationAgreement
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
end SemanticResult83804

namespace SemanticResult83809
def owner : Owner := ⟨.program ⟨214⟩, ⟨7218⟩⟩
def rawTerms : List Term := Proof.Events327.exact83809RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83809
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83809.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83808.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge83808.frameStart)
    (transferEvent := 83807) (owner := owner)
    (leftResult := 79790) (rightResult := 10521)
    (working := LeftOperatorMerge83808.working)
    (reconstruction := LeftOperatorMerge83808.reconstruction)
    (leftReference := .predecessor 0 83805 .coefficient) (rightReference := .predecessor 1 83806 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10521.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge83808.operationAgreement
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
end SemanticResult83809

namespace SemanticResult83813
def owner : Owner := ⟨.program ⟨214⟩, ⟨14646⟩⟩
def rawTerms : List Term := Proof.Events327.exact83813RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83813
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83813.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 83810) (rightBinding := 83811)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7218⟩) (rightExpression := ⟨14645⟩)
    (transferEvent := 83812)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83809.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult83804.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult83813

namespace SemanticResult83819
def owner : Owner := ⟨.program ⟨214⟩, ⟨14647⟩⟩
def rawTerms : List Term := Proof.Events327.exact83819RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 83819
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83819.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 83816) (survivorTransfer := 83817)
    (survivorEvent := 83818) (resultEvent := resultEvent)
    (rightCoefficientProducer := 10512)
    (owner := owner) (leftOwner := SemanticResult83813.owner)
    (rightOwner := SemanticResult10513.owner)
    (leftResult := 83813) (rightResult := 10513)
    (leftBinding := 83814) (rightBinding := 83815)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14646⟩) (rightExpression := ⟨76⟩)
    (leftActual := SemanticResult83813.actual selector witness)
    (rightActual := SemanticResult10513.actual selector witness)
    (leftRaw := SemanticResult83813.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨76⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound10512.actual selector witness)
    (survivorMagnitude := LeftBound83817.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83813.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10513.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10512.derived selector witness)
  · exact LeftBound83817.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult83819

namespace SemanticResult83829
def owner : Owner := ⟨.program ⟨214⟩, ⟨14648⟩⟩
def rawTerms : List Term := Proof.Events327.exact83829RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 83829
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83829.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83825.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge83825.frameStart)
    (owner := owner) (leftOwner := SemanticResult83819.owner)
    (rightOwner := SemanticResult10510.owner)
    (leftResult := 83819) (rightResult := 10510)
    (leftActual := SemanticResult83819.actual selector witness)
    (rightActual := SemanticResult10510.actual selector witness)
    (leftRaw := SemanticResult83819.rawTerms)
    (rightRaw := SemanticResult10510.rawTerms)
    (working := LeftOperatorMerge83825.working)
    (leftBinding := 83820) (rightBinding := 83821)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14647⟩) (rightExpression := ⟨7859⟩)
    (coefficientTransfer := 83822) (summaryTransfer := 83824)
    (rightCoefficientProducer := 10509)
    (rightSummaryTransfer := 83823)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge83825.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound10509.actual selector witness)
    (summaryMagnitude := LeftBound83824.actual selector witness)
    (reconstruction := LeftOperatorMerge83825.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83819.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10510.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10509.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound10509.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge83825.operationAgreement
  · exact LeftBound83824.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83825.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 83826 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge83825.working
    [{ coefficient := (-1), key := LeftRelationMerge83826.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge83826.frameStart
      LeftRelationMerge83826.owner (.relation 83826) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge83826.deltas
    rows := LeftRelationMerge83826.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge83825.working LeftRelationMerge83826.source
        (relationContext LeftRelationMerge83826.source
          LeftRelationMerge83826.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge83825.working, LeftRelationMerge83826.deltas,
    LeftRelationMerge83826.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 83826)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨14648⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge83825.working) (working := relationWorking0)
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
end SemanticResult83829

namespace SemanticResult83835
def owner : Owner := ⟨.program ⟨214⟩, ⟨14649⟩⟩
def rawTerms : List Term := Proof.Events327.exact83835RawTerms
def summary : Bound := (.finite 95443712)
def resultEvent : Nat := 83835
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83835.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge83833.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult83829.owner)
    (rightOwner := SemanticResult83799.owner)
    (leftResult := 83829) (rightResult := 83799)
    (leftActual := SemanticResult83829.actual selector witness)
    (rightActual := SemanticResult83799.actual selector witness)
    (leftRaw := SemanticResult83829.rawTerms)
    (rightRaw := SemanticResult83799.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 23296) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 83830) (rightBinding := 83831)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14648⟩) (rightExpression := ⟨14644⟩)
    (coefficientTransfer := 83832) (summaryTransfer := 83834)
    (base := LeftOperatorMerge83833.base)
    (reconstruction := LeftOperatorMerge83833.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83829.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult83799.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge83833.operationAgreement
  · rfl
  · decide
end SemanticResult83835

namespace SemanticResult83845
def owner : Owner := ⟨.program ⟨214⟩, ⟨26221⟩⟩
def rawTerms : List Term := Proof.Events327.exact83845RawTerms
def summary : Bound := (.finite 350279950139392)
def resultEvent : Nat := 83845
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83845.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95443712, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83841.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge83841.frameStart)
    (owner := owner) (leftOwner := SemanticResult83835.owner)
    (rightOwner := SemanticResult83771.owner)
    (leftResult := 83835) (rightResult := 83771)
    (leftActual := SemanticResult83835.actual selector witness)
    (rightActual := SemanticResult83771.actual selector witness)
    (leftRaw := SemanticResult83835.rawTerms)
    (rightRaw := SemanticResult83771.rawTerms)
    (working := LeftOperatorMerge83841.working)
    (leftBinding := 83836) (rightBinding := 83837)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14649⟩) (rightExpression := ⟨26220⟩)
    (coefficientTransfer := 83838) (summaryTransfer := 83840)
    (rightCoefficientProducer := 83770)
    (rightSummaryTransfer := 83839)
    (leftMaximum := ⟨95443712, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge83841.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority83770.actual selector witness)
    (summaryMagnitude := LeftBound83840.actual selector witness)
    (reconstruction := LeftOperatorMerge83841.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83835.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult83771.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83770.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority83770.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge83841.operationAgreement
  · exact LeftBound83840.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83841.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 83842 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23668⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23668⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge83841.working
    [{ coefficient := (-1), key := LeftRelationMerge83842.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge83842.frameStart
      LeftRelationMerge83842.owner (.relation 83842) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge83842.deltas
    rows := LeftRelationMerge83842.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge83841.working LeftRelationMerge83842.source
        (relationContext LeftRelationMerge83842.source
          LeftRelationMerge83842.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge83841.working, LeftRelationMerge83842.deltas,
    LeftRelationMerge83842.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 83842)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨26221⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge83841.working) (working := relationWorking0)
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
end SemanticResult83845

namespace SemanticResult83848
def owner : Owner := ⟨.program ⟨214⟩, ⟨19672⟩⟩
def rawTerms : List Term := Proof.Events327.exact83848RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83848
def producerEvent : Nat := 83847
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83848.actual selector witness
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
end SemanticResult83848

namespace SemanticResult83852
def owner : Owner := ⟨.program ⟨214⟩, ⟨19674⟩⟩
def rawTerms : List Term := Proof.Events327.exact83852RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83852
def producerEvent : Nat := 83851
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83852.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 83849 .coefficient) (.value (.predecessor 1 83850 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 83849 .coefficient) (.value (.predecessor 1 83850 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult83852

namespace SemanticResult83930
def owner : Owner := ⟨.program ⟨214⟩, ⟨11637⟩⟩
def rawTerms : List Term := Proof.Events327.exact83930RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83930
def producerEvent : Nat := 83929
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83930.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 83907, .finite 28, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult83930

namespace SemanticResult83933
def owner : Owner := ⟨.program ⟨214⟩, ⟨14641⟩⟩
def rawTerms : List Term := Proof.Events327.exact83933RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83933
def producerEvent : Nat := 83932
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83933.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 83907, .finite 28, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult83933

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
