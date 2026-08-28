import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard055
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard043
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard050
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard051
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard052
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard053
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard054

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult6214
def owner : Owner := ⟨.program ⟨214⟩, ⟨7796⟩⟩
def rawTerms : List Term := Proof.Events024.exact6214RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6214
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6214.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge6195.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge6195.frameStart)
    (transferEvent := 6194) (owner := owner)
    (leftResult := 5954) (rightResult := 6191)
    (working := LeftOperatorMerge6195.working)
    (reconstruction := LeftOperatorMerge6195.reconstruction)
    (leftReference := .predecessor 0 6192 .coefficient) (rightReference := .predecessor 1 6193 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult5954.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6191.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge6195.operationAgreement
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
end SemanticResult6214

namespace SemanticResult6218
def owner : Owner := ⟨.program ⟨214⟩, ⟨7797⟩⟩
def rawTerms : List Term := Proof.Events024.exact6218RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6218
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6218.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6215) (rightBinding := 6216)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7650⟩) (rightExpression := ⟨7796⟩)
    (transferEvent := 6217)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5878.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6214.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6218

namespace SemanticResult6222
def owner : Owner := ⟨.program ⟨214⟩, ⟨7923⟩⟩
def rawTerms : List Term := Proof.Events024.exact6222RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6222
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6222.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6219) (rightBinding := 6220)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7797⟩) (rightExpression := ⟨7922⟩)
    (transferEvent := 6221)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6218.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6179.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6222

namespace SemanticResult6226
def owner : Owner := ⟨.program ⟨214⟩, ⟨7924⟩⟩
def rawTerms : List Term := Proof.Events024.exact6226RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6226
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6226.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6223) (rightBinding := 6224)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7923⟩) (rightExpression := ⟨7921⟩)
    (transferEvent := 6225)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6222.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6139.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6226

namespace SemanticResult6230
def owner : Owner := ⟨.program ⟨214⟩, ⟨7925⟩⟩
def rawTerms : List Term := Proof.Events024.exact6230RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6230
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6230.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6227) (rightBinding := 6228)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7924⟩) (rightExpression := ⟨7920⟩)
    (transferEvent := 6229)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6226.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6099.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6230

namespace SemanticResult6234
def owner : Owner := ⟨.program ⟨214⟩, ⟨7926⟩⟩
def rawTerms : List Term := Proof.Events024.exact6234RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6234
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6234.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6231) (rightBinding := 6232)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7925⟩) (rightExpression := ⟨7919⟩)
    (transferEvent := 6233)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6230.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6059.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6234

namespace SemanticResult6238
def owner : Owner := ⟨.program ⟨214⟩, ⟨7927⟩⟩
def rawTerms : List Term := Proof.Events024.exact6238RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6238
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6238.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6235) (rightBinding := 6236)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7926⟩) (rightExpression := ⟨7918⟩)
    (transferEvent := 6237)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6234.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6019.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6238

namespace SemanticResult6242
def owner : Owner := ⟨.program ⟨214⟩, ⟨7928⟩⟩
def rawTerms : List Term := Proof.Events024.exact6242RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6242
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6242.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6239) (rightBinding := 6240)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7927⟩) (rightExpression := ⟨7917⟩)
    (transferEvent := 6241)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6238.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5979.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6242

namespace SemanticResult6271
def owner : Owner := ⟨.program ⟨214⟩, ⟨7929⟩⟩
def rawTerms : List Term := Proof.Events024.exact6271RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6271
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6271.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge6246.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge6246.frameStart)
    (transferEvent := 6245) (owner := owner)
    (leftResult := 27) (rightResult := 6242)
    (working := LeftOperatorMerge6246.working)
    (reconstruction := LeftOperatorMerge6246.reconstruction)
    (leftReference := .predecessor 0 6243 .coefficient) (rightReference := .predecessor 1 6244 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6242.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge6246.operationAgreement
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
end SemanticResult6271

namespace SemanticResult6275
def owner : Owner := ⟨.program ⟨214⟩, ⟨18909⟩⟩
def rawTerms : List Term := Proof.Events024.exact6275RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6275
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6275.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6272) (rightBinding := 6273)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7929⟩) (rightExpression := ⟨18907⟩)
    (transferEvent := 6274)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6271.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5464.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6275

namespace SemanticResult6277
def owner : Owner := ⟨.program ⟨214⟩, ⟨5⟩⟩
def rawTerms : List Term := Proof.Events024.exact6277RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6277
def producerEvent : Nat := 6276
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6277.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .finite 26, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult6277

namespace SemanticResult6298
def owner : Owner := ⟨.program ⟨214⟩, ⟨5619⟩⟩
def rawTerms : List Term := Proof.Events024.exact6298RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6298
def producerEvent : Nat := 6297
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6298.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 6292 .coefficient), 0, .finite 1, .identity (.predecessor 0 6292 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult6298

namespace SemanticResult6303
def owner : Owner := ⟨.program ⟨214⟩, ⟨6583⟩⟩
def rawTerms : List Term := Proof.Events024.exact6303RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6303
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6303.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge6302.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge6302.frameStart)
    (transferEvent := 6301) (owner := owner)
    (leftResult := 6298) (rightResult := 2)
    (working := LeftOperatorMerge6302.working)
    (reconstruction := LeftOperatorMerge6302.reconstruction)
    (leftReference := .predecessor 0 6299 .coefficient) (rightReference := .predecessor 1 6300 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult6298.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge6302.operationAgreement
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
end SemanticResult6303

namespace SemanticResult6314
def owner : Owner := ⟨.program ⟨214⟩, ⟨5563⟩⟩
def rawTerms : List Term := Proof.Events024.exact6314RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6314
def producerEvent : Nat := 6313
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6314.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 6308 .coefficient), 0, .finite 1, .identity (.predecessor 0 6308 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult6314

namespace SemanticResult6319
def owner : Owner := ⟨.program ⟨214⟩, ⟨7365⟩⟩
def rawTerms : List Term := Proof.Events024.exact6319RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6319
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6319.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge6318.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge6318.frameStart)
    (transferEvent := 6317) (owner := owner)
    (leftResult := 6314) (rightResult := 5480)
    (working := LeftOperatorMerge6318.working)
    (reconstruction := LeftOperatorMerge6318.reconstruction)
    (leftReference := .predecessor 0 6315 .coefficient) (rightReference := .predecessor 1 6316 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult6314.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5480.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge6318.operationAgreement
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
end SemanticResult6319

namespace SemanticResult6323
def owner : Owner := ⟨.program ⟨214⟩, ⟨7767⟩⟩
def rawTerms : List Term := Proof.Events024.exact6323RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6323
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult6323.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6320) (rightBinding := 6321)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7365⟩) (rightExpression := ⟨6583⟩)
    (transferEvent := 6322)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6319.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6303.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6323

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
