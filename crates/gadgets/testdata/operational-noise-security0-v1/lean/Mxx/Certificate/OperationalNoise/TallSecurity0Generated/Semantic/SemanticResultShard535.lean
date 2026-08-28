import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard535
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard533
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard534

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult75202
def owner : Owner := ⟨.program ⟨214⟩, ⟨18329⟩⟩
def rawTerms : List Term := Proof.Events293.exact75202RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75202
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75202.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75199) (rightBinding := 75200)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18328⟩) (rightExpression := ⟨16305⟩)
    (transferEvent := 75201)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75198.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult74932.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75202

namespace SemanticResult75206
def owner : Owner := ⟨.program ⟨214⟩, ⟨18330⟩⟩
def rawTerms : List Term := Proof.Events293.exact75206RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75206
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75206.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75203) (rightBinding := 75204)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18329⟩) (rightExpression := ⟨17117⟩)
    (transferEvent := 75205)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75202.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult74909.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75206

namespace SemanticResult75210
def owner : Owner := ⟨.program ⟨214⟩, ⟨18331⟩⟩
def rawTerms : List Term := Proof.Events293.exact75210RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75210
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75210.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75207) (rightBinding := 75208)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18330⟩) (rightExpression := ⟨17901⟩)
    (transferEvent := 75209)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75206.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult74886.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75210

namespace SemanticResult75214
def owner : Owner := ⟨.program ⟨214⟩, ⟨18332⟩⟩
def rawTerms : List Term := Proof.Events293.exact75214RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75214
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75214.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75211) (rightBinding := 75212)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18331⟩) (rightExpression := ⟨18202⟩)
    (transferEvent := 75213)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75210.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult74863.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75214

namespace SemanticResult75218
def owner : Owner := ⟨.program ⟨214⟩, ⟨18333⟩⟩
def rawTerms : List Term := Proof.Events293.exact75218RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75218
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75218.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75215) (rightBinding := 75216)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18332⟩) (rightExpression := ⟨16676⟩)
    (transferEvent := 75217)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75214.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult74840.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75218

namespace SemanticResult75222
def owner : Owner := ⟨.program ⟨214⟩, ⟨18334⟩⟩
def rawTerms : List Term := Proof.Events293.exact75222RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75222
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75222.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75219) (rightBinding := 75220)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18333⟩) (rightExpression := ⟨16795⟩)
    (transferEvent := 75221)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75218.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult74817.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75222

namespace SemanticResult75226
def owner : Owner := ⟨.program ⟨214⟩, ⟨18335⟩⟩
def rawTerms : List Term := Proof.Events293.exact75226RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75226
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75226.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75223) (rightBinding := 75224)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18334⟩) (rightExpression := ⟨17082⟩)
    (transferEvent := 75225)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75222.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult74794.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75226

namespace SemanticResult75230
def owner : Owner := ⟨.program ⟨214⟩, ⟨18336⟩⟩
def rawTerms : List Term := Proof.Events293.exact75230RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75230
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75230.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75227) (rightBinding := 75228)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18335⟩) (rightExpression := ⟨18167⟩)
    (transferEvent := 75229)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75226.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult74771.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75230

namespace SemanticResult75241
def owner : Owner := ⟨.program ⟨214⟩, ⟨18616⟩⟩
def rawTerms : List Term := Proof.Events293.exact75241RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75241
def producerEvent : Nat := 75240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75241.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 74728, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult75241

namespace SemanticResult75244
def owner : Owner := ⟨.program ⟨214⟩, ⟨18678⟩⟩
def rawTerms : List Term := Proof.Events293.exact75244RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75244
def producerEvent : Nat := 75243
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75244.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 74728, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult75244

namespace SemanticResult75253
def owner : Owner := ⟨.program ⟨214⟩, ⟨18644⟩⟩
def rawTerms : List Term := Proof.Events293.exact75253RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75253
def producerEvent : Nat := 75252
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75253.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 75251 .coefficient), 74728, .finite 1059, .identity (.predecessor 0 75251 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult75253

namespace SemanticResult75255
def owner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rawTerms : List Term := Proof.Events293.exact75255RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75255
def producerEvent : Nat := 75254
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75255.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 74728, .large, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult75255

namespace SemanticResult75277
def owner : Owner := ⟨.program ⟨214⟩, ⟨18645⟩⟩
def rawTerms : List Term := Proof.Events294.exact75277RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75277
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75277.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge75259.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge75259.frameStart)
    (transferEvent := 75258) (owner := owner)
    (leftResult := 75255) (rightResult := 75253)
    (working := LeftOperatorMerge75259.working)
    (reconstruction := LeftOperatorMerge75259.reconstruction)
    (leftReference := .predecessor 0 75256 .coefficient) (rightReference := .predecessor 1 75257 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult75255.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75253.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge75259.operationAgreement
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
end SemanticResult75277

namespace SemanticResult75280
def owner : Owner := ⟨.program ⟨214⟩, ⟨6743⟩⟩
def rawTerms : List Term := Proof.Events294.exact75280RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75280
def producerEvent : Nat := 75279
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75280.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 74728, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult75280

namespace SemanticResult75283
def owner : Owner := ⟨.program ⟨214⟩, ⟨6741⟩⟩
def rawTerms : List Term := Proof.Events294.exact75283RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75283
def producerEvent : Nat := 75282
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75283.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 74728, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult75283

namespace SemanticResult75286
def owner : Owner := ⟨.program ⟨214⟩, ⟨6739⟩⟩
def rawTerms : List Term := Proof.Events294.exact75286RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75286
def producerEvent : Nat := 75285
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75286.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 74728, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult75286

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
