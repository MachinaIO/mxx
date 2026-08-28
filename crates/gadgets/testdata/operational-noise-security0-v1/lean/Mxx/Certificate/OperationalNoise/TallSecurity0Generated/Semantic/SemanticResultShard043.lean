import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard043
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard007
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard013
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard031
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard037
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard039
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard042

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult5276
def owner : Owner := ⟨.program ⟨214⟩, ⟨18802⟩⟩
def rawTerms : List Term := Proof.Events020.exact5276RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5276
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5276.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5273) (rightBinding := 5274)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18800⟩) (rightExpression := ⟨18486⟩)
    (transferEvent := 5275)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5272.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5056.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5276

namespace SemanticResult5299
def owner : Owner := ⟨.program ⟨214⟩, ⟨18803⟩⟩
def rawTerms : List Term := Proof.Events020.exact5299RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5299
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5299.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge5280.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge5280.frameStart)
    (transferEvent := 5279) (owner := owner)
    (leftResult := 5276) (rightResult := 4563)
    (working := LeftOperatorMerge5280.working)
    (reconstruction := LeftOperatorMerge5280.reconstruction)
    (leftReference := .predecessor 0 5277 .coefficient) (rightReference := .predecessor 1 5278 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult5276.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4563.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge5280.operationAgreement
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
end SemanticResult5299

namespace SemanticResult5303
def owner : Owner := ⟨.program ⟨214⟩, ⟨18804⟩⟩
def rawTerms : List Term := Proof.Events020.exact5303RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5303
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5303.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5300) (rightBinding := 5301)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6379⟩) (rightExpression := ⟨18803⟩)
    (transferEvent := 5302)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5299.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5303

namespace SemanticResult5307
def owner : Owner := ⟨.program ⟨214⟩, ⟨18844⟩⟩
def rawTerms : List Term := Proof.Events020.exact5307RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5307
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5307.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5304) (rightBinding := 5305)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18804⟩) (rightExpression := ⟨18843⟩)
    (transferEvent := 5306)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5303.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4561.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5307

namespace SemanticResult5311
def owner : Owner := ⟨.program ⟨214⟩, ⟨18845⟩⟩
def rawTerms : List Term := Proof.Events020.exact5311RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5311
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5311.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5308) (rightBinding := 5309)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18844⟩) (rightExpression := ⟨18829⟩)
    (transferEvent := 5310)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5307.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3819.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5311

namespace SemanticResult5315
def owner : Owner := ⟨.program ⟨214⟩, ⟨18860⟩⟩
def rawTerms : List Term := Proof.Events020.exact5315RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5315
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5315.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5312) (rightBinding := 5313)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18845⟩) (rightExpression := ⟨18859⟩)
    (transferEvent := 5314)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5311.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3071.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5315

namespace SemanticResult5319
def owner : Owner := ⟨.program ⟨214⟩, ⟨18875⟩⟩
def rawTerms : List Term := Proof.Events020.exact5319RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5319
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5319.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5316) (rightBinding := 5317)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18860⟩) (rightExpression := ⟨18874⟩)
    (transferEvent := 5318)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5315.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2323.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5319

namespace SemanticResult5323
def owner : Owner := ⟨.program ⟨214⟩, ⟨18890⟩⟩
def rawTerms : List Term := Proof.Events020.exact5323RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5323
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5323.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5320) (rightBinding := 5321)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18875⟩) (rightExpression := ⟨18889⟩)
    (transferEvent := 5322)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5319.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1575.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5323

namespace SemanticResult5327
def owner : Owner := ⟨.program ⟨214⟩, ⟨18905⟩⟩
def rawTerms : List Term := Proof.Events020.exact5327RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5327
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5327.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5324) (rightBinding := 5325)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18890⟩) (rightExpression := ⟨18904⟩)
    (transferEvent := 5326)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5323.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult827.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5327

namespace SemanticResult5464
def owner : Owner := ⟨.program ⟨214⟩, ⟨18907⟩⟩
def rawTerms : List Term := Proof.Events021.exact5464RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5464
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5464.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge5331.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge5331.frameStart)
    (transferEvent := 5330) (owner := owner)
    (leftResult := 5327) (rightResult := 32)
    (working := LeftOperatorMerge5331.working)
    (reconstruction := LeftOperatorMerge5331.reconstruction)
    (leftReference := .predecessor 0 5328 .coefficient) (rightReference := .predecessor 1 5329 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult5327.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge5331.operationAgreement
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
end SemanticResult5464

namespace SemanticResult5469
def owner : Owner := ⟨.program ⟨214⟩, ⟨6594⟩⟩
def rawTerms : List Term := Proof.Events021.exact5469RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5469
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5469.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge5468.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge5468.frameStart)
    (transferEvent := 5467) (owner := owner)
    (leftResult := 2) (rightResult := 34)
    (working := LeftOperatorMerge5468.working)
    (reconstruction := LeftOperatorMerge5468.reconstruction)
    (leftReference := .predecessor 0 5465 .coefficient) (rightReference := .predecessor 1 5466 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult2.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult34.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge5468.operationAgreement
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
end SemanticResult5469

namespace SemanticResult5472
def owner : Owner := ⟨.program ⟨214⟩, ⟨6645⟩⟩
def rawTerms : List Term := Proof.Events021.exact5472RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5472
def producerEvent : Nat := 5471
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5472.actual selector witness
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
end SemanticResult5472

namespace SemanticResult5476
def owner : Owner := ⟨.program ⟨214⟩, ⟨6646⟩⟩
def rawTerms : List Term := Proof.Events021.exact5476RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5476
def producerEvent : Nat := 5475
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5476.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 5473 .coefficient) (.value (.predecessor 1 5474 .coefficient)), 0, .finite 8192, .scale (.predecessor 0 5473 .coefficient) (.value (.predecessor 1 5474 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult5476

namespace SemanticResult5480
def owner : Owner := ⟨.program ⟨214⟩, ⟨6746⟩⟩
def rawTerms : List Term := Proof.Events021.exact5480RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5480
def producerEvent : Nat := 5479
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5480.actual selector witness
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
end SemanticResult5480

namespace SemanticResult5483
def owner : Owner := ⟨.program ⟨214⟩, ⟨7819⟩⟩
def rawTerms : List Term := Proof.Events021.exact5483RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5483
def producerEvent : Nat := 5482
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5483.actual selector witness
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
end SemanticResult5483

namespace SemanticResult5487
def owner : Owner := ⟨.program ⟨214⟩, ⟨7820⟩⟩
def rawTerms : List Term := Proof.Events021.exact5487RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5487
def producerEvent : Nat := 5486
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5487.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 5484 .coefficient) (.value (.predecessor 1 5485 .coefficient)), 0, .finite 8192, .scale (.predecessor 0 5484 .coefficient) (.value (.predecessor 1 5485 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult5487

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
