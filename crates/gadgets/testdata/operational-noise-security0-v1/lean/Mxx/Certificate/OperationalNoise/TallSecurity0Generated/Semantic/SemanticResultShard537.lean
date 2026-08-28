import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard537
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard535
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard536

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult75339
def owner : Owner := ⟨.program ⟨214⟩, ⟨6796⟩⟩
def rawTerms : List Term := Proof.Events294.exact75339RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75339
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75339.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75336) (rightBinding := 75337)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6795⟩) (rightExpression := ⟨6713⟩)
    (transferEvent := 75338)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75335.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75325.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75339

namespace SemanticResult75343
def owner : Owner := ⟨.program ⟨214⟩, ⟨6797⟩⟩
def rawTerms : List Term := Proof.Events294.exact75343RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75343
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75343.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75340) (rightBinding := 75341)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6796⟩) (rightExpression := ⟨6715⟩)
    (transferEvent := 75342)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75339.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75322.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75343

namespace SemanticResult75347
def owner : Owner := ⟨.program ⟨214⟩, ⟨6798⟩⟩
def rawTerms : List Term := Proof.Events294.exact75347RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75347
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75347.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75344) (rightBinding := 75345)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6797⟩) (rightExpression := ⟨6717⟩)
    (transferEvent := 75346)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75343.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75319.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75347

namespace SemanticResult75351
def owner : Owner := ⟨.program ⟨214⟩, ⟨6799⟩⟩
def rawTerms : List Term := Proof.Events294.exact75351RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75351
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75351.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75348) (rightBinding := 75349)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6798⟩) (rightExpression := ⟨6719⟩)
    (transferEvent := 75350)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75347.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75316.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75351

namespace SemanticResult75355
def owner : Owner := ⟨.program ⟨214⟩, ⟨6800⟩⟩
def rawTerms : List Term := Proof.Events294.exact75355RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75355
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75355.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75352) (rightBinding := 75353)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6799⟩) (rightExpression := ⟨6721⟩)
    (transferEvent := 75354)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75351.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75313.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75355

namespace SemanticResult75359
def owner : Owner := ⟨.program ⟨214⟩, ⟨6801⟩⟩
def rawTerms : List Term := Proof.Events294.exact75359RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75359
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75359.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75356) (rightBinding := 75357)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6800⟩) (rightExpression := ⟨6723⟩)
    (transferEvent := 75358)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75355.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75310.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75359

namespace SemanticResult75363
def owner : Owner := ⟨.program ⟨214⟩, ⟨6802⟩⟩
def rawTerms : List Term := Proof.Events294.exact75363RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75363
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75363.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75360) (rightBinding := 75361)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6801⟩) (rightExpression := ⟨6725⟩)
    (transferEvent := 75362)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75359.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75307.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75363

namespace SemanticResult75367
def owner : Owner := ⟨.program ⟨214⟩, ⟨6803⟩⟩
def rawTerms : List Term := Proof.Events294.exact75367RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75367
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75367.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75364) (rightBinding := 75365)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6802⟩) (rightExpression := ⟨6727⟩)
    (transferEvent := 75366)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75363.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75304.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75367

namespace SemanticResult75371
def owner : Owner := ⟨.program ⟨214⟩, ⟨6804⟩⟩
def rawTerms : List Term := Proof.Events294.exact75371RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75371
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75371.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75368) (rightBinding := 75369)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6803⟩) (rightExpression := ⟨6729⟩)
    (transferEvent := 75370)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75367.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75301.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75371

namespace SemanticResult75375
def owner : Owner := ⟨.program ⟨214⟩, ⟨6805⟩⟩
def rawTerms : List Term := Proof.Events294.exact75375RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75375
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75375.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75372) (rightBinding := 75373)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6804⟩) (rightExpression := ⟨6731⟩)
    (transferEvent := 75374)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75371.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75298.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75375

namespace SemanticResult75379
def owner : Owner := ⟨.program ⟨214⟩, ⟨6806⟩⟩
def rawTerms : List Term := Proof.Events294.exact75379RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75379
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75379.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75376) (rightBinding := 75377)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6805⟩) (rightExpression := ⟨6733⟩)
    (transferEvent := 75378)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75375.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75295.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75379

namespace SemanticResult75383
def owner : Owner := ⟨.program ⟨214⟩, ⟨6807⟩⟩
def rawTerms : List Term := Proof.Events294.exact75383RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75383
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75383.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75380) (rightBinding := 75381)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6806⟩) (rightExpression := ⟨6735⟩)
    (transferEvent := 75382)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75379.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75292.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75383

namespace SemanticResult75387
def owner : Owner := ⟨.program ⟨214⟩, ⟨6808⟩⟩
def rawTerms : List Term := Proof.Events294.exact75387RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75387
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75387.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75384) (rightBinding := 75385)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6807⟩) (rightExpression := ⟨6737⟩)
    (transferEvent := 75386)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75383.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75289.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75387

namespace SemanticResult75391
def owner : Owner := ⟨.program ⟨214⟩, ⟨6809⟩⟩
def rawTerms : List Term := Proof.Events294.exact75391RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75391
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75391.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75388) (rightBinding := 75389)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6808⟩) (rightExpression := ⟨6739⟩)
    (transferEvent := 75390)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75387.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75286.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75391

namespace SemanticResult75395
def owner : Owner := ⟨.program ⟨214⟩, ⟨6810⟩⟩
def rawTerms : List Term := Proof.Events294.exact75395RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75395
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75395.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75392) (rightBinding := 75393)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6809⟩) (rightExpression := ⟨6741⟩)
    (transferEvent := 75394)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75391.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75283.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75395

namespace SemanticResult75399
def owner : Owner := ⟨.program ⟨214⟩, ⟨6811⟩⟩
def rawTerms : List Term := Proof.Events294.exact75399RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 75399
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult75399.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 75396) (rightBinding := 75397)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6810⟩) (rightExpression := ⟨6743⟩)
    (transferEvent := 75398)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75395.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75280.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75399

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
