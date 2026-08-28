import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard013
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard052
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard263

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult35920
def owner : Owner := ⟨.program ⟨214⟩, ⟨7289⟩⟩
def rawTerms : List Term := Proof.Events140.exact35920RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 35920
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35920.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge35919.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge35919.frameStart)
    (transferEvent := 35918) (owner := owner)
    (leftResult := 35915) (rightResult := 6034)
    (working := LeftOperatorMerge35919.working)
    (reconstruction := LeftOperatorMerge35919.reconstruction)
    (leftReference := .predecessor 0 35916 .coefficient) (rightReference := .predecessor 1 35917 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6034.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge35919.operationAgreement
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
end SemanticResult35920

namespace SemanticResult35924
def owner : Owner := ⟨.program ⟨214⟩, ⟨7759⟩⟩
def rawTerms : List Term := Proof.Events140.exact35924RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 35924
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35924.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 35921) (rightBinding := 35922)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7289⟩) (rightExpression := ⟨6581⟩)
    (transferEvent := 35923)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35920.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35904.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35924

namespace SemanticResult35930
def owner : Owner := ⟨.program ⟨214⟩, ⟨7760⟩⟩
def rawTerms : List Term := Proof.Events140.exact35930RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 35930
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35930.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 35927) (survivorTransfer := 35928)
    (survivorEvent := 35929) (resultEvent := resultEvent)
    (rightCoefficientProducer := 35877)
    (owner := owner) (leftOwner := SemanticResult35924.owner)
    (rightOwner := SemanticResult35878.owner)
    (leftResult := 35924) (rightResult := 35878)
    (leftBinding := 35925) (rightBinding := 35926)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7759⟩) (rightExpression := ⟨1⟩)
    (leftActual := SemanticResult35924.actual selector witness)
    (rightActual := SemanticResult35878.actual selector witness)
    (leftRaw := SemanticResult35924.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨1⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftAuthority35877.actual selector witness)
    (survivorMagnitude := LeftBound35928.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35924.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35878.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35877.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35877.derived selector witness)
  · exact LeftBound35928.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult35930

namespace SemanticResult36010
def owner : Owner := ⟨.program ⟨214⟩, ⟨18876⟩⟩
def rawTerms : List Term := Proof.Events140.exact36010RawTerms
def summary : Bound := (.finite 6740345342118210980043475264)
def resultEvent : Nat := 36010
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36010.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8101376613122849735629177, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge35972.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge35972.frameStart)
    (owner := owner) (leftOwner := SemanticResult35930.owner)
    (rightOwner := SemanticResult2300.owner)
    (leftResult := 35930) (rightResult := 2300)
    (leftActual := SemanticResult35930.actual selector witness)
    (rightActual := SemanticResult2300.actual selector witness)
    (leftRaw := SemanticResult35930.rawTerms)
    (rightRaw := SemanticResult2300.rawTerms)
    (working := LeftOperatorMerge35972.working)
    (leftBinding := 35931) (rightBinding := 35932)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7760⟩) (rightExpression := ⟨18873⟩)
    (coefficientTransfer := 35933) (summaryTransfer := 35971)
    (rightCoefficientProducer := 2299)
    (rightSummaryTransfer := 35970)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8101376613122849735629179, by decide⟩)
    (rightRecordedMaximum := 8101376613122849735629177)
    (rightSummaryMaximum := ⟨8101376613122849735629177, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge35972.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound2299.actual selector witness)
    (summaryMagnitude := LeftBound35971.actual selector witness)
    (reconstruction := LeftOperatorMerge35972.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35930.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2300.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2299.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound2299.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge35972.operationAgreement
  · exact LeftBound35971.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge35972.working summary) := by
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
end SemanticResult36010

namespace SemanticResult36017
def owner : Owner := ⟨.program ⟨214⟩, ⟨18622⟩⟩
def rawTerms : List Term := Proof.Events140.exact36017RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36017
def producerEvent : Nat := 36016
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36017.actual selector witness
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
end SemanticResult36017

namespace SemanticResult36020
def owner : Owner := ⟨.program ⟨214⟩, ⟨18687⟩⟩
def rawTerms : List Term := Proof.Events140.exact36020RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36020
def producerEvent : Nat := 36019
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36020.actual selector witness
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
end SemanticResult36020

namespace SemanticResult36027
def owner : Owner := ⟨.program ⟨214⟩, ⟨24798⟩⟩
def rawTerms : List Term := Proof.Events140.exact36027RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36027
def producerEvent : Nat := 36026
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36027.actual selector witness
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
end SemanticResult36027

namespace SemanticResult36030
def owner : Owner := ⟨.program ⟨214⟩, ⟨30161⟩⟩
def rawTerms : List Term := Proof.Events140.exact36030RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36030
def producerEvent : Nat := 36029
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36030.actual selector witness
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
end SemanticResult36030

namespace SemanticResult36037
def owner : Owner := ⟨.program ⟨214⟩, ⟨23420⟩⟩
def rawTerms : List Term := Proof.Events140.exact36037RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36037
def producerEvent : Nat := 36036
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36037.actual selector witness
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
end SemanticResult36037

namespace SemanticResult36040
def owner : Owner := ⟨.program ⟨214⟩, ⟨25768⟩⟩
def rawTerms : List Term := Proof.Events140.exact36040RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36040
def producerEvent : Nat := 36039
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36040.actual selector witness
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
end SemanticResult36040

namespace SemanticResult36045
def owner : Owner := ⟨.program ⟨214⟩, ⟨6569⟩⟩
def rawTerms : List Term := Proof.Events140.exact36045RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36045
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36045.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36044.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge36044.frameStart)
    (transferEvent := 36043) (owner := owner)
    (leftResult := 35915) (rightResult := 2)
    (working := LeftOperatorMerge36044.working)
    (reconstruction := LeftOperatorMerge36044.reconstruction)
    (leftReference := .predecessor 0 36041 .coefficient) (rightReference := .predecessor 1 36042 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge36044.operationAgreement
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
end SemanticResult36045

namespace SemanticResult36050
def owner : Owner := ⟨.program ⟨214⟩, ⟨13369⟩⟩
def rawTerms : List Term := Proof.Events140.exact36050RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36050
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36050.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36049.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge36049.frameStart)
    (transferEvent := 36048) (owner := owner)
    (leftResult := 1590) (rightResult := 36045)
    (working := LeftOperatorMerge36049.working)
    (reconstruction := LeftOperatorMerge36049.reconstruction)
    (leftReference := .predecessor 0 36046 .coefficient) (rightReference := .predecessor 1 36047 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1590.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge36049.operationAgreement
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
end SemanticResult36050

namespace SemanticResult36055
def owner : Owner := ⟨.program ⟨214⟩, ⟨7322⟩⟩
def rawTerms : List Term := Proof.Events140.exact36055RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36055
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36055.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36054.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge36054.frameStart)
    (transferEvent := 36053) (owner := owner)
    (leftResult := 35915) (rightResult := 6457)
    (working := LeftOperatorMerge36054.working)
    (reconstruction := LeftOperatorMerge36054.reconstruction)
    (leftReference := .predecessor 0 36051 .coefficient) (rightReference := .predecessor 1 36052 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6457.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge36054.operationAgreement
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
end SemanticResult36055

namespace SemanticResult36059
def owner : Owner := ⟨.program ⟨214⟩, ⟨13370⟩⟩
def rawTerms : List Term := Proof.Events140.exact36059RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36059
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36059.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 36056) (rightBinding := 36057)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7322⟩) (rightExpression := ⟨13369⟩)
    (transferEvent := 36058)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36055.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36050.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult36059

namespace SemanticResult36065
def owner : Owner := ⟨.program ⟨214⟩, ⟨13371⟩⟩
def rawTerms : List Term := Proof.Events140.exact36065RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 36065
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36065.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 36062) (survivorTransfer := 36063)
    (survivorEvent := 36064) (resultEvent := resultEvent)
    (rightCoefficientProducer := 6443)
    (owner := owner) (leftOwner := SemanticResult36059.owner)
    (rightOwner := SemanticResult6444.owner)
    (leftResult := 36059) (rightResult := 6444)
    (leftBinding := 36060) (rightBinding := 36061)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13370⟩) (rightExpression := ⟨104⟩)
    (leftActual := SemanticResult36059.actual selector witness)
    (rightActual := SemanticResult6444.actual selector witness)
    (leftRaw := SemanticResult36059.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨104⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound6443.actual selector witness)
    (survivorMagnitude := LeftBound36063.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36059.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6444.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6443.derived selector witness)
  · exact LeftBound36063.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult36065

namespace SemanticResult36073
def owner : Owner := ⟨.program ⟨214⟩, ⟨13372⟩⟩
def rawTerms : List Term := Proof.Events140.exact36073RawTerms
def summary : Bound := (.finite 49920)
def resultEvent : Nat := 36073
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36073.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨60, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36071.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge36071.frameStart)
    (owner := owner) (leftOwner := SemanticResult36065.owner)
    (rightOwner := SemanticResult1593.owner)
    (leftResult := 36065) (rightResult := 1593)
    (leftActual := SemanticResult36065.actual selector witness)
    (rightActual := SemanticResult1593.actual selector witness)
    (leftRaw := SemanticResult36065.rawTerms)
    (rightRaw := SemanticResult1593.rawTerms)
    (working := LeftOperatorMerge36071.working)
    (leftBinding := 36066) (rightBinding := 36067)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13371⟩) (rightExpression := ⟨10355⟩)
    (coefficientTransfer := 36068) (summaryTransfer := 36070)
    (rightCoefficientProducer := 1592)
    (rightSummaryTransfer := 36069)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨60, by decide⟩)
    (rightRecordedMaximum := 60)
    (rightSummaryMaximum := ⟨60, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge36071.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1592.actual selector witness)
    (summaryMagnitude := LeftBound36070.actual selector witness)
    (reconstruction := LeftOperatorMerge36071.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36065.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1593.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1592.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1592.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge36071.operationAgreement
  · exact LeftBound36070.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36071.working summary) := by
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
end SemanticResult36073

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
