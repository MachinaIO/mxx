import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard031
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard053
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard464

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult65154
def owner : Owner := ⟨.program ⟨214⟩, ⟨6578⟩⟩
def rawTerms : List Term := Proof.Events254.exact65154RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65154
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65154.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge65153.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge65153.frameStart)
    (transferEvent := 65152) (owner := owner)
    (leftResult := 65149) (rightResult := 2)
    (working := LeftOperatorMerge65153.working)
    (reconstruction := LeftOperatorMerge65153.reconstruction)
    (leftReference := .predecessor 0 65150 .coefficient) (rightReference := .predecessor 1 65151 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65149.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge65153.operationAgreement
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
end SemanticResult65154

namespace SemanticResult65165
def owner : Owner := ⟨.program ⟨214⟩, ⟨5533⟩⟩
def rawTerms : List Term := Proof.Events254.exact65165RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65165
def producerEvent : Nat := 65164
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65165.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 65159 .coefficient), 0, .finite 1, .identity (.predecessor 0 65159 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult65165

namespace SemanticResult65170
def owner : Owner := ⟨.program ⟨214⟩, ⟨7175⟩⟩
def rawTerms : List Term := Proof.Events254.exact65170RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65170
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65170.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge65169.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge65169.frameStart)
    (transferEvent := 65168) (owner := owner)
    (leftResult := 65165) (rightResult := 6114)
    (working := LeftOperatorMerge65169.working)
    (reconstruction := LeftOperatorMerge65169.reconstruction)
    (leftReference := .predecessor 0 65166 .coefficient) (rightReference := .predecessor 1 65167 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6114.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge65169.operationAgreement
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
end SemanticResult65170

namespace SemanticResult65174
def owner : Owner := ⟨.program ⟨214⟩, ⟨7747⟩⟩
def rawTerms : List Term := Proof.Events254.exact65174RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65174
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65174.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 65171) (rightBinding := 65172)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7175⟩) (rightExpression := ⟨6578⟩)
    (transferEvent := 65173)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult65170.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65154.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult65174

namespace SemanticResult65180
def owner : Owner := ⟨.program ⟨214⟩, ⟨7748⟩⟩
def rawTerms : List Term := Proof.Events254.exact65180RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 65180
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65180.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 65177) (survivorTransfer := 65178)
    (survivorEvent := 65179) (resultEvent := resultEvent)
    (rightCoefficientProducer := 65127)
    (owner := owner) (leftOwner := SemanticResult65174.owner)
    (rightOwner := SemanticResult65128.owner)
    (leftResult := 65174) (rightResult := 65128)
    (leftBinding := 65175) (rightBinding := 65176)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7747⟩) (rightExpression := ⟨66⟩)
    (leftActual := SemanticResult65174.actual selector witness)
    (rightActual := SemanticResult65128.actual selector witness)
    (leftRaw := SemanticResult65174.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨66⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftAuthority65127.actual selector witness)
    (survivorMagnitude := LeftBound65178.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult65174.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65128.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65127.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65127.derived selector witness)
  · exact LeftBound65178.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult65180

namespace SemanticResult65260
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def rawTerms : List Term := Proof.Events254.exact65260RawTerms
def summary : Bound := (.finite 6740345342118210980043475264)
def resultEvent : Nat := 65260
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65260.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8101376613122849735629177, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge65222.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge65222.frameStart)
    (owner := owner) (leftOwner := SemanticResult65180.owner)
    (rightOwner := SemanticResult3796.owner)
    (leftResult := 65180) (rightResult := 3796)
    (leftActual := SemanticResult65180.actual selector witness)
    (rightActual := SemanticResult3796.actual selector witness)
    (leftRaw := SemanticResult65180.rawTerms)
    (rightRaw := SemanticResult3796.rawTerms)
    (working := LeftOperatorMerge65222.working)
    (leftBinding := 65181) (rightBinding := 65182)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7748⟩) (rightExpression := ⟨18828⟩)
    (coefficientTransfer := 65183) (summaryTransfer := 65221)
    (rightCoefficientProducer := 3795)
    (rightSummaryTransfer := 65220)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8101376613122849735629179, by decide⟩)
    (rightRecordedMaximum := 8101376613122849735629177)
    (rightSummaryMaximum := ⟨8101376613122849735629177, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge65222.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound3795.actual selector witness)
    (summaryMagnitude := LeftBound65221.actual selector witness)
    (reconstruction := LeftOperatorMerge65222.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult65180.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3796.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3795.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound3795.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge65222.operationAgreement
  · exact LeftBound65221.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge65222.working summary) := by
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
end SemanticResult65260

namespace SemanticResult65267
def owner : Owner := ⟨.program ⟨214⟩, ⟨18616⟩⟩
def rawTerms : List Term := Proof.Events254.exact65267RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65267
def producerEvent : Nat := 65266
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65267.actual selector witness
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
end SemanticResult65267

namespace SemanticResult65270
def owner : Owner := ⟨.program ⟨214⟩, ⟨18678⟩⟩
def rawTerms : List Term := Proof.Events254.exact65270RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65270
def producerEvent : Nat := 65269
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65270.actual selector witness
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
end SemanticResult65270

namespace SemanticResult65277
def owner : Owner := ⟨.program ⟨214⟩, ⟨24789⟩⟩
def rawTerms : List Term := Proof.Events254.exact65277RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65277
def producerEvent : Nat := 65276
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65277.actual selector witness
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
end SemanticResult65277

namespace SemanticResult65280
def owner : Owner := ⟨.program ⟨214⟩, ⟨30095⟩⟩
def rawTerms : List Term := Proof.Events255.exact65280RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65280
def producerEvent : Nat := 65279
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65280.actual selector witness
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
end SemanticResult65280

namespace SemanticResult65287
def owner : Owner := ⟨.program ⟨214⟩, ⟨23414⟩⟩
def rawTerms : List Term := Proof.Events255.exact65287RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65287
def producerEvent : Nat := 65286
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65287.actual selector witness
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
end SemanticResult65287

namespace SemanticResult65290
def owner : Owner := ⟨.program ⟨214⟩, ⟨25753⟩⟩
def rawTerms : List Term := Proof.Events255.exact65290RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65290
def producerEvent : Nat := 65289
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65290.actual selector witness
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
end SemanticResult65290

namespace SemanticResult65295
def owner : Owner := ⟨.program ⟨214⟩, ⟨6566⟩⟩
def rawTerms : List Term := Proof.Events255.exact65295RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65295
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65295.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge65294.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge65294.frameStart)
    (transferEvent := 65293) (owner := owner)
    (leftResult := 65165) (rightResult := 2)
    (working := LeftOperatorMerge65294.working)
    (reconstruction := LeftOperatorMerge65294.reconstruction)
    (leftReference := .predecessor 0 65291 .coefficient) (rightReference := .predecessor 1 65292 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge65294.operationAgreement
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
end SemanticResult65295

namespace SemanticResult65300
def owner : Owner := ⟨.program ⟨214⟩, ⟨13345⟩⟩
def rawTerms : List Term := Proof.Events255.exact65300RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65300
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65300.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge65299.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge65299.frameStart)
    (transferEvent := 65298) (owner := owner)
    (leftResult := 3086) (rightResult := 65295)
    (working := LeftOperatorMerge65299.working)
    (reconstruction := LeftOperatorMerge65299.reconstruction)
    (leftReference := .predecessor 0 65296 .coefficient) (rightReference := .predecessor 1 65297 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3086.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge65299.operationAgreement
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
end SemanticResult65300

namespace SemanticResult65305
def owner : Owner := ⟨.program ⟨214⟩, ⟨7208⟩⟩
def rawTerms : List Term := Proof.Events255.exact65305RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65305
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65305.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge65304.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge65304.frameStart)
    (transferEvent := 65303) (owner := owner)
    (leftResult := 65165) (rightResult := 6457)
    (working := LeftOperatorMerge65304.working)
    (reconstruction := LeftOperatorMerge65304.reconstruction)
    (leftReference := .predecessor 0 65301 .coefficient) (rightReference := .predecessor 1 65302 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6457.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge65304.operationAgreement
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
end SemanticResult65305

namespace SemanticResult65309
def owner : Owner := ⟨.program ⟨214⟩, ⟨13346⟩⟩
def rawTerms : List Term := Proof.Events255.exact65309RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65309
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65309.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 65306) (rightBinding := 65307)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7208⟩) (rightExpression := ⟨13345⟩)
    (transferEvent := 65308)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult65305.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65300.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult65309

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
