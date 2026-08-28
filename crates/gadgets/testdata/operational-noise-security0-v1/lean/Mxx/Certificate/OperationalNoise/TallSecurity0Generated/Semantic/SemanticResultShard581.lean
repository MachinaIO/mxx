import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard581
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard031
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard073
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard566

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult81838
def owner : Owner := ⟨.program ⟨214⟩, ⟨24540⟩⟩
def rawTerms : List Term := Proof.Events319.exact81838RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81838
def producerEvent : Nat := 81837
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81838.actual selector witness
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
end SemanticResult81838

namespace SemanticResult81841
def owner : Owner := ⟨.program ⟨214⟩, ⟨29168⟩⟩
def rawTerms : List Term := Proof.Events319.exact81841RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81841
def producerEvent : Nat := 81840
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81841.actual selector witness
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
end SemanticResult81841

namespace SemanticResult81848
def owner : Owner := ⟨.program ⟨214⟩, ⟨23248⟩⟩
def rawTerms : List Term := Proof.Events319.exact81848RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81848
def producerEvent : Nat := 81847
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81848.actual selector witness
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
end SemanticResult81848

namespace SemanticResult81851
def owner : Owner := ⟨.program ⟨214⟩, ⟨25450⟩⟩
def rawTerms : List Term := Proof.Events319.exact81851RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81851
def producerEvent : Nat := 81850
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81851.actual selector witness
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
end SemanticResult81851

namespace SemanticResult81856
def owner : Owner := ⟨.program ⟨214⟩, ⟨12569⟩⟩
def rawTerms : List Term := Proof.Events319.exact81856RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81856
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81856.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81855.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge81855.frameStart)
    (transferEvent := 81854) (owner := owner)
    (leftResult := 3920) (rightResult := 79920)
    (working := LeftOperatorMerge81855.working)
    (reconstruction := LeftOperatorMerge81855.reconstruction)
    (leftReference := .predecessor 0 81852 .coefficient) (rightReference := .predecessor 1 81853 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3920.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79920.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge81855.operationAgreement
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
end SemanticResult81856

namespace SemanticResult81861
def owner : Owner := ⟨.program ⟨214⟩, ⟨7242⟩⟩
def rawTerms : List Term := Proof.Events319.exact81861RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81861
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81861.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81860.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge81860.frameStart)
    (transferEvent := 81859) (owner := owner)
    (leftResult := 79790) (rightResult := 8476)
    (working := LeftOperatorMerge81860.working)
    (reconstruction := LeftOperatorMerge81860.reconstruction)
    (leftReference := .predecessor 0 81857 .coefficient) (rightReference := .predecessor 1 81858 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8476.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge81860.operationAgreement
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
end SemanticResult81861

namespace SemanticResult81865
def owner : Owner := ⟨.program ⟨214⟩, ⟨12570⟩⟩
def rawTerms : List Term := Proof.Events319.exact81865RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81865
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81865.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 81862) (rightBinding := 81863)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7242⟩) (rightExpression := ⟨12569⟩)
    (transferEvent := 81864)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult81861.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult81856.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult81865

namespace SemanticResult81871
def owner : Owner := ⟨.program ⟨214⟩, ⟨12571⟩⟩
def rawTerms : List Term := Proof.Events319.exact81871RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 81871
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81871.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 81868) (survivorTransfer := 81869)
    (survivorEvent := 81870) (resultEvent := resultEvent)
    (rightCoefficientProducer := 8467)
    (owner := owner) (leftOwner := SemanticResult81865.owner)
    (rightOwner := SemanticResult8468.owner)
    (leftResult := 81865) (rightResult := 8468)
    (leftBinding := 81866) (rightBinding := 81867)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12570⟩) (rightExpression := ⟨100⟩)
    (leftActual := SemanticResult81865.actual selector witness)
    (rightActual := SemanticResult8468.actual selector witness)
    (leftRaw := SemanticResult81865.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨100⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound8467.actual selector witness)
    (survivorMagnitude := LeftBound81869.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult81865.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8468.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8467.derived selector witness)
  · exact LeftBound81869.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult81871

namespace SemanticResult81879
def owner : Owner := ⟨.program ⟨214⟩, ⟨12572⟩⟩
def rawTerms : List Term := Proof.Events319.exact81879RawTerms
def summary : Bound := (.finite 34944)
def resultEvent : Nat := 81879
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81879.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨42, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81877.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge81877.frameStart)
    (owner := owner) (leftOwner := SemanticResult81871.owner)
    (rightOwner := SemanticResult3923.owner)
    (leftResult := 81871) (rightResult := 3923)
    (leftActual := SemanticResult81871.actual selector witness)
    (rightActual := SemanticResult3923.actual selector witness)
    (leftRaw := SemanticResult81871.rawTerms)
    (rightRaw := SemanticResult3923.rawTerms)
    (working := LeftOperatorMerge81877.working)
    (leftBinding := 81872) (rightBinding := 81873)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12571⟩) (rightExpression := ⟨9925⟩)
    (coefficientTransfer := 81874) (summaryTransfer := 81876)
    (rightCoefficientProducer := 3922)
    (rightSummaryTransfer := 81875)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨42, by decide⟩)
    (rightRecordedMaximum := 42)
    (rightSummaryMaximum := ⟨42, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge81877.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority3922.actual selector witness)
    (summaryMagnitude := LeftBound81876.actual selector witness)
    (reconstruction := LeftOperatorMerge81877.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult81871.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3923.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3922.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority3922.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge81877.operationAgreement
  · exact LeftBound81876.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81877.working summary) := by
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
end SemanticResult81879

namespace SemanticResult81884
def owner : Owner := ⟨.program ⟨214⟩, ⟨9926⟩⟩
def rawTerms : List Term := Proof.Events319.exact81884RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81884
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81884.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81883.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge81883.frameStart)
    (transferEvent := 81882) (owner := owner)
    (leftResult := 3923) (rightResult := 79920)
    (working := LeftOperatorMerge81883.working)
    (reconstruction := LeftOperatorMerge81883.reconstruction)
    (leftReference := .predecessor 0 81880 .coefficient) (rightReference := .predecessor 1 81881 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3923.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79920.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge81883.operationAgreement
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
end SemanticResult81884

namespace SemanticResult81889
def owner : Owner := ⟨.program ⟨214⟩, ⟨7222⟩⟩
def rawTerms : List Term := Proof.Events319.exact81889RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81889
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81889.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81888.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge81888.frameStart)
    (transferEvent := 81887) (owner := owner)
    (leftResult := 79790) (rightResult := 8517)
    (working := LeftOperatorMerge81888.working)
    (reconstruction := LeftOperatorMerge81888.reconstruction)
    (leftReference := .predecessor 0 81885 .coefficient) (rightReference := .predecessor 1 81886 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8517.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge81888.operationAgreement
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
end SemanticResult81889

namespace SemanticResult81893
def owner : Owner := ⟨.program ⟨214⟩, ⟨9927⟩⟩
def rawTerms : List Term := Proof.Events319.exact81893RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81893
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81893.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 81890) (rightBinding := 81891)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7222⟩) (rightExpression := ⟨9926⟩)
    (transferEvent := 81892)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult81889.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult81884.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult81893

namespace SemanticResult81899
def owner : Owner := ⟨.program ⟨214⟩, ⟨9928⟩⟩
def rawTerms : List Term := Proof.Events319.exact81899RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 81899
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81899.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 81896) (survivorTransfer := 81897)
    (survivorEvent := 81898) (resultEvent := resultEvent)
    (rightCoefficientProducer := 8508)
    (owner := owner) (leftOwner := SemanticResult81893.owner)
    (rightOwner := SemanticResult8509.owner)
    (leftResult := 81893) (rightResult := 8509)
    (leftBinding := 81894) (rightBinding := 81895)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9927⟩) (rightExpression := ⟨80⟩)
    (leftActual := SemanticResult81893.actual selector witness)
    (rightActual := SemanticResult8509.actual selector witness)
    (leftRaw := SemanticResult81893.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨80⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound8508.actual selector witness)
    (survivorMagnitude := LeftBound81897.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult81893.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8509.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8508.derived selector witness)
  · exact LeftBound81897.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult81899

namespace SemanticResult81909
def owner : Owner := ⟨.program ⟨214⟩, ⟨9929⟩⟩
def rawTerms : List Term := Proof.Events319.exact81909RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 81909
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81909.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81905.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge81905.frameStart)
    (owner := owner) (leftOwner := SemanticResult81899.owner)
    (rightOwner := SemanticResult8506.owner)
    (leftResult := 81899) (rightResult := 8506)
    (leftActual := SemanticResult81899.actual selector witness)
    (rightActual := SemanticResult8506.actual selector witness)
    (leftRaw := SemanticResult81899.rawTerms)
    (rightRaw := SemanticResult8506.rawTerms)
    (working := LeftOperatorMerge81905.working)
    (leftBinding := 81900) (rightBinding := 81901)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9928⟩) (rightExpression := ⟨7871⟩)
    (coefficientTransfer := 81902) (summaryTransfer := 81904)
    (rightCoefficientProducer := 8505)
    (rightSummaryTransfer := 81903)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge81905.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound8505.actual selector witness)
    (summaryMagnitude := LeftBound81904.actual selector witness)
    (reconstruction := LeftOperatorMerge81905.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult81899.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8506.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8505.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound8505.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge81905.operationAgreement
  · exact LeftBound81904.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81905.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 81906 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9925⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge81905.working
    [{ coefficient := (-1), key := LeftRelationMerge81906.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge81906.frameStart
      LeftRelationMerge81906.owner (.relation 81906) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge81906.deltas
    rows := LeftRelationMerge81906.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge81905.working LeftRelationMerge81906.source
        (relationContext LeftRelationMerge81906.source
          LeftRelationMerge81906.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge81905.working, LeftRelationMerge81906.deltas,
    LeftRelationMerge81906.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 81906)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9929⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9925⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge81905.working) (working := relationWorking0)
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
end SemanticResult81909

namespace SemanticResult81915
def owner : Owner := ⟨.program ⟨214⟩, ⟨12573⟩⟩
def rawTerms : List Term := Proof.Events319.exact81915RawTerms
def summary : Bound := (.finite 95455360)
def resultEvent : Nat := 81915
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81915.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge81913.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult81909.owner)
    (rightOwner := SemanticResult81879.owner)
    (leftResult := 81909) (rightResult := 81879)
    (leftActual := SemanticResult81909.actual selector witness)
    (rightActual := SemanticResult81879.actual selector witness)
    (leftRaw := SemanticResult81909.rawTerms)
    (rightRaw := SemanticResult81879.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 34944) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 81910) (rightBinding := 81911)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9929⟩) (rightExpression := ⟨12572⟩)
    (coefficientTransfer := 81912) (summaryTransfer := 81914)
    (base := LeftOperatorMerge81913.base)
    (reconstruction := LeftOperatorMerge81913.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult81909.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult81879.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge81913.operationAgreement
  · rfl
  · decide
end SemanticResult81915

namespace SemanticResult81925
def owner : Owner := ⟨.program ⟨214⟩, ⟨25451⟩⟩
def rawTerms : List Term := Proof.Events320.exact81925RawTerms
def summary : Bound := (.finite 350322698485760)
def resultEvent : Nat := 81925
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81925.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95455360, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81921.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge81921.frameStart)
    (owner := owner) (leftOwner := SemanticResult81915.owner)
    (rightOwner := SemanticResult81851.owner)
    (leftResult := 81915) (rightResult := 81851)
    (leftActual := SemanticResult81915.actual selector witness)
    (rightActual := SemanticResult81851.actual selector witness)
    (leftRaw := SemanticResult81915.rawTerms)
    (rightRaw := SemanticResult81851.rawTerms)
    (working := LeftOperatorMerge81921.working)
    (leftBinding := 81916) (rightBinding := 81917)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12573⟩) (rightExpression := ⟨25450⟩)
    (coefficientTransfer := 81918) (summaryTransfer := 81920)
    (rightCoefficientProducer := 81850)
    (rightSummaryTransfer := 81919)
    (leftMaximum := ⟨95455360, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge81921.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority81850.actual selector witness)
    (summaryMagnitude := LeftBound81920.actual selector witness)
    (reconstruction := LeftOperatorMerge81921.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult81915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult81851.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81850.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority81850.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge81921.operationAgreement
  · exact LeftBound81920.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81921.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 81922 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23248⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23248⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge81921.working
    [{ coefficient := (-1), key := LeftRelationMerge81922.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge81922.frameStart
      LeftRelationMerge81922.owner (.relation 81922) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge81922.deltas
    rows := LeftRelationMerge81922.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge81921.working LeftRelationMerge81922.source
        (relationContext LeftRelationMerge81922.source
          LeftRelationMerge81922.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge81921.working, LeftRelationMerge81922.deltas,
    LeftRelationMerge81922.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 81922)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25451⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge81921.working) (working := relationWorking0)
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
end SemanticResult81925

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
