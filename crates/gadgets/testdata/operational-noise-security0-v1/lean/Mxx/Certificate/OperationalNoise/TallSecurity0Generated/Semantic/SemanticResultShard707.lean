import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard707
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard038
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard101
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard102

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult99155
def owner : Owner := ⟨.program ⟨214⟩, ⟨25976⟩⟩
def rawTerms : List Term := Proof.Events387.exact99155RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99155
def producerEvent : Nat := 99154
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99155.actual selector witness
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
end SemanticResult99155

namespace SemanticResult99160
def owner : Owner := ⟨.program ⟨214⟩, ⟨11374⟩⟩
def rawTerms : List Term := Proof.Events387.exact99160RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99160
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99160.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99159.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge99159.frameStart)
    (transferEvent := 99158) (owner := owner)
    (leftResult := 4819) (rightResult := 32)
    (working := LeftOperatorMerge99159.working)
    (reconstruction := LeftOperatorMerge99159.reconstruction)
    (leftReference := .predecessor 0 99156 .coefficient) (rightReference := .predecessor 1 99157 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4819.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge99159.operationAgreement
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
end SemanticResult99160

namespace SemanticResult99165
def owner : Owner := ⟨.program ⟨214⟩, ⟨7115⟩⟩
def rawTerms : List Term := Proof.Events387.exact99165RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99165
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99165.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99164.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge99164.frameStart)
    (transferEvent := 99163) (owner := owner)
    (leftResult := 27) (rightResult := 11983)
    (working := LeftOperatorMerge99164.working)
    (reconstruction := LeftOperatorMerge99164.reconstruction)
    (leftReference := .predecessor 0 99161 .coefficient) (rightReference := .predecessor 1 99162 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11983.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge99164.operationAgreement
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
end SemanticResult99165

namespace SemanticResult99169
def owner : Owner := ⟨.program ⟨214⟩, ⟨11375⟩⟩
def rawTerms : List Term := Proof.Events387.exact99169RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99169
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99169.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 99166) (rightBinding := 99167)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7115⟩) (rightExpression := ⟨11374⟩)
    (transferEvent := 99168)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99160.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99169

namespace SemanticResult99175
def owner : Owner := ⟨.program ⟨214⟩, ⟨11376⟩⟩
def rawTerms : List Term := Proof.Events387.exact99175RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 99175
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99175.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 99172) (survivorTransfer := 99173)
    (survivorEvent := 99174) (resultEvent := resultEvent)
    (rightCoefficientProducer := 11974)
    (owner := owner) (leftOwner := SemanticResult99169.owner)
    (rightOwner := SemanticResult11975.owner)
    (leftResult := 99169) (rightResult := 11975)
    (leftBinding := 99170) (rightBinding := 99171)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11375⟩) (rightExpression := ⟨92⟩)
    (leftActual := SemanticResult99169.actual selector witness)
    (rightActual := SemanticResult11975.actual selector witness)
    (leftRaw := SemanticResult99169.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨92⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound11974.actual selector witness)
    (survivorMagnitude := LeftBound99173.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99169.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11975.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11974.derived selector witness)
  · exact LeftBound99173.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult99175

namespace SemanticResult99183
def owner : Owner := ⟨.program ⟨214⟩, ⟨13966⟩⟩
def rawTerms : List Term := Proof.Events387.exact99183RawTerms
def summary : Bound := (.finite 13312)
def resultEvent : Nat := 99183
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99183.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨16, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99181.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge99181.frameStart)
    (owner := owner) (leftOwner := SemanticResult99175.owner)
    (rightOwner := SemanticResult4822.owner)
    (leftResult := 99175) (rightResult := 4822)
    (leftActual := SemanticResult99175.actual selector witness)
    (rightActual := SemanticResult4822.actual selector witness)
    (leftRaw := SemanticResult99175.rawTerms)
    (rightRaw := SemanticResult4822.rawTerms)
    (working := LeftOperatorMerge99181.working)
    (leftBinding := 99176) (rightBinding := 99177)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11376⟩) (rightExpression := ⟨13963⟩)
    (coefficientTransfer := 99178) (summaryTransfer := 99180)
    (rightCoefficientProducer := 4821)
    (rightSummaryTransfer := 99179)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨16, by decide⟩)
    (rightRecordedMaximum := 16)
    (rightSummaryMaximum := ⟨16, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge99181.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority4821.actual selector witness)
    (summaryMagnitude := LeftBound99180.actual selector witness)
    (reconstruction := LeftOperatorMerge99181.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99175.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4822.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4821.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority4821.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge99181.operationAgreement
  · exact LeftBound99180.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99181.working summary) := by
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
end SemanticResult99183

namespace SemanticResult99188
def owner : Owner := ⟨.program ⟨214⟩, ⟨13967⟩⟩
def rawTerms : List Term := Proof.Events387.exact99188RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99188
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99188.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99187.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge99187.frameStart)
    (transferEvent := 99186) (owner := owner)
    (leftResult := 4822) (rightResult := 32)
    (working := LeftOperatorMerge99187.working)
    (reconstruction := LeftOperatorMerge99187.reconstruction)
    (leftReference := .predecessor 0 99184 .coefficient) (rightReference := .predecessor 1 99185 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4822.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge99187.operationAgreement
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
end SemanticResult99188

namespace SemanticResult99193
def owner : Owner := ⟨.program ⟨214⟩, ⟨7095⟩⟩
def rawTerms : List Term := Proof.Events387.exact99193RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99193
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99193.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99192.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge99192.frameStart)
    (transferEvent := 99191) (owner := owner)
    (leftResult := 27) (rightResult := 12024)
    (working := LeftOperatorMerge99192.working)
    (reconstruction := LeftOperatorMerge99192.reconstruction)
    (leftReference := .predecessor 0 99189 .coefficient) (rightReference := .predecessor 1 99190 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12024.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge99192.operationAgreement
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
end SemanticResult99193

namespace SemanticResult99197
def owner : Owner := ⟨.program ⟨214⟩, ⟨13968⟩⟩
def rawTerms : List Term := Proof.Events387.exact99197RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99197
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99197.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 99194) (rightBinding := 99195)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7095⟩) (rightExpression := ⟨13967⟩)
    (transferEvent := 99196)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99193.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99188.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99197

namespace SemanticResult99203
def owner : Owner := ⟨.program ⟨214⟩, ⟨13969⟩⟩
def rawTerms : List Term := Proof.Events387.exact99203RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 99203
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99203.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 99200) (survivorTransfer := 99201)
    (survivorEvent := 99202) (resultEvent := resultEvent)
    (rightCoefficientProducer := 12015)
    (owner := owner) (leftOwner := SemanticResult99197.owner)
    (rightOwner := SemanticResult12016.owner)
    (leftResult := 99197) (rightResult := 12016)
    (leftBinding := 99198) (rightBinding := 99199)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13968⟩) (rightExpression := ⟨72⟩)
    (leftActual := SemanticResult99197.actual selector witness)
    (rightActual := SemanticResult12016.actual selector witness)
    (leftRaw := SemanticResult99197.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨72⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound12015.actual selector witness)
    (survivorMagnitude := LeftBound99201.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99197.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12016.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12015.derived selector witness)
  · exact LeftBound99201.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult99203

namespace SemanticResult99213
def owner : Owner := ⟨.program ⟨214⟩, ⟨13970⟩⟩
def rawTerms : List Term := Proof.Events387.exact99213RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 99213
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99213.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99209.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge99209.frameStart)
    (owner := owner) (leftOwner := SemanticResult99203.owner)
    (rightOwner := SemanticResult12013.owner)
    (leftResult := 99203) (rightResult := 12013)
    (leftActual := SemanticResult99203.actual selector witness)
    (rightActual := SemanticResult12013.actual selector witness)
    (leftRaw := SemanticResult99203.rawTerms)
    (rightRaw := SemanticResult12013.rawTerms)
    (working := LeftOperatorMerge99209.working)
    (leftBinding := 99204) (rightBinding := 99205)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13969⟩) (rightExpression := ⟨7850⟩)
    (coefficientTransfer := 99206) (summaryTransfer := 99208)
    (rightCoefficientProducer := 12012)
    (rightSummaryTransfer := 99207)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge99209.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound12012.actual selector witness)
    (summaryMagnitude := LeftBound99208.actual selector witness)
    (reconstruction := LeftOperatorMerge99209.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99203.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12013.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12012.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound12012.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge99209.operationAgreement
  · exact LeftBound99208.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99209.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 99210 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13963⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge99209.working
    [{ coefficient := (-1), key := LeftRelationMerge99210.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge99210.frameStart
      LeftRelationMerge99210.owner (.relation 99210) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge99210.deltas
    rows := LeftRelationMerge99210.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge99209.working LeftRelationMerge99210.source
        (relationContext LeftRelationMerge99210.source
          LeftRelationMerge99210.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge99209.working, LeftRelationMerge99210.deltas,
    LeftRelationMerge99210.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 99210)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨13970⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13963⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge99209.working) (working := relationWorking0)
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
end SemanticResult99213

namespace SemanticResult99219
def owner : Owner := ⟨.program ⟨214⟩, ⟨13971⟩⟩
def rawTerms : List Term := Proof.Events387.exact99219RawTerms
def summary : Bound := (.finite 95433728)
def resultEvent : Nat := 99219
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99219.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge99217.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult99213.owner)
    (rightOwner := SemanticResult99183.owner)
    (leftResult := 99213) (rightResult := 99183)
    (leftActual := SemanticResult99213.actual selector witness)
    (rightActual := SemanticResult99183.actual selector witness)
    (leftRaw := SemanticResult99213.rawTerms)
    (rightRaw := SemanticResult99183.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 13312) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 99214) (rightBinding := 99215)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13970⟩) (rightExpression := ⟨13966⟩)
    (coefficientTransfer := 99216) (summaryTransfer := 99218)
    (base := LeftOperatorMerge99217.base)
    (reconstruction := LeftOperatorMerge99217.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99213.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99183.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge99217.operationAgreement
  · rfl
  · decide
end SemanticResult99219

namespace SemanticResult99229
def owner : Owner := ⟨.program ⟨214⟩, ⟨25977⟩⟩
def rawTerms : List Term := Proof.Events387.exact99229RawTerms
def summary : Bound := (.finite 350243308699648)
def resultEvent : Nat := 99229
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99229.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95433728, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99225.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge99225.frameStart)
    (owner := owner) (leftOwner := SemanticResult99219.owner)
    (rightOwner := SemanticResult99155.owner)
    (leftResult := 99219) (rightResult := 99155)
    (leftActual := SemanticResult99219.actual selector witness)
    (rightActual := SemanticResult99155.actual selector witness)
    (leftRaw := SemanticResult99219.rawTerms)
    (rightRaw := SemanticResult99155.rawTerms)
    (working := LeftOperatorMerge99225.working)
    (leftBinding := 99220) (rightBinding := 99221)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13971⟩) (rightExpression := ⟨25976⟩)
    (coefficientTransfer := 99222) (summaryTransfer := 99224)
    (rightCoefficientProducer := 99154)
    (rightSummaryTransfer := 99223)
    (leftMaximum := ⟨95433728, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge99225.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority99154.actual selector witness)
    (summaryMagnitude := LeftBound99224.actual selector witness)
    (reconstruction := LeftOperatorMerge99225.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99219.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99155.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99154.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority99154.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge99225.operationAgreement
  · exact LeftBound99224.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99225.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 99226 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25976⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11373⟩⟩, ⟨.program ⟨214⟩, ⟨13963⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23536⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23536⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge99225.working
    [{ coefficient := (-1), key := LeftRelationMerge99226.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge99226.frameStart
      LeftRelationMerge99226.owner (.relation 99226) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge99226.deltas
    rows := LeftRelationMerge99226.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge99225.working LeftRelationMerge99226.source
        (relationContext LeftRelationMerge99226.source
          LeftRelationMerge99226.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge99225.working, LeftRelationMerge99226.deltas,
    LeftRelationMerge99226.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 99226)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25977⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11373⟩⟩, ⟨.program ⟨214⟩, ⟨13963⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25976⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25976⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge99225.working) (working := relationWorking0)
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
end SemanticResult99229

namespace SemanticResult99232
def owner : Owner := ⟨.program ⟨214⟩, ⟨19445⟩⟩
def rawTerms : List Term := Proof.Events387.exact99232RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99232
def producerEvent : Nat := 99231
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99232.actual selector witness
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
end SemanticResult99232

namespace SemanticResult99236
def owner : Owner := ⟨.program ⟨214⟩, ⟨19447⟩⟩
def rawTerms : List Term := Proof.Events387.exact99236RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99236
def producerEvent : Nat := 99235
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99236.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 99233 .coefficient) (.value (.predecessor 1 99234 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 99233 .coefficient) (.value (.predecessor 1 99234 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult99236

namespace SemanticResult99290
def owner : Owner := ⟨.program ⟨214⟩, ⟨11373⟩⟩
def rawTerms : List Term := Proof.Events387.exact99290RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99290
def producerEvent : Nat := 99289
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99290.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 99279, .finite 16, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult99290

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
