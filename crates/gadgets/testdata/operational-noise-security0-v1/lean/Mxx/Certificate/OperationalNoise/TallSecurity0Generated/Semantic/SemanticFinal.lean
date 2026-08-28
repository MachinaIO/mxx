import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResult
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.StatementResidual
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Proof

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticFinal

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

def owner : Owner := ⟨.program ⟨214⟩, ⟨30220⟩⟩
def resultEvent : Nat := 107564
def preFoldEvent : Nat := 107565
def endEvent : Nat := 107566
def rawTerms : List Term := []
def summary : Bound := .finite 25317157507886064950797272225391822339692950454324
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.statementResidual (some selector) witness

theorem resultAt : history.lookup resultEvent = some
    ⟨.resultExact owner rawTerms .large 107405
      summary none, 0⟩ := by
  rfl

theorem resultClaimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact SemanticResult.resultClaimSound selector selectorLower selectorUpper witness

theorem preFoldAt : history.lookup preFoldEvent = some
    ⟨.preFoldPolynomial resultEvent rawTerms summary
      (some (.result resultEvent .summary)), 0⟩ := by
  rfl

theorem invocationEndAt : history.lookup endEvent = some
    ⟨.invocationEndExact owner preFoldEvent rawTerms .large
      107405 summary none, 0⟩ := by
  rfl

theorem invocationEndClaimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact (rawTerms.map Term.toExact) summary) := by
  exact invocationEndSound 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
    (rawTerms.map Term.toExact) (rawTerms.map Term.toExact) summary summary
    (resultClaimSound selector selectorLower selectorUpper witness).claim rfl rfl

theorem strictBoundSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    2 * 268435009 * centeredNorm 100418593683253592432016548326729029359133068138294319235841 (actual selector witness) < 100418593683253592432016548326729029359133068138294319235841 := by
  apply finalStrictBound_of_empty_finite_claim 268435009 100418593683253592432016548326729029359133068138294319235841 witness.env
    (actual selector witness) 25317157507886064950797272225391822339692950454324
  · simpa [rawTerms, summary] using
      invocationEndClaimSound selector selectorLower selectorUpper witness
  · decide
  · decide

theorem fixedSemanticSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    history.lookup resultEvent = some
        ⟨.resultExact owner rawTerms .large 107405
          summary none, 0⟩ ∧
      history.lookup preFoldEvent = some
        ⟨.preFoldPolynomial resultEvent rawTerms summary
          (some (.result resultEvent .summary)), 0⟩ ∧
      history.lookup endEvent = some
        ⟨.invocationEndExact owner preFoldEvent rawTerms .large
          107405 summary none, 0⟩ ∧
      ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
        (.exact (rawTerms.map Term.toExact) summary) ∧
      2 * 268435009 * centeredNorm 100418593683253592432016548326729029359133068138294319235841 (actual selector witness) < 100418593683253592432016548326729029359133068138294319235841 := by
  exact ⟨resultAt, preFoldAt, invocationEndAt,
    invocationEndClaimSound selector selectorLower selectorUpper witness,
    strictBoundSound selector selectorLower selectorUpper witness⟩

theorem fixedAcceptance :
    OperationalCertificateAccepted document history 268435009 100418593683253592432016548326729029359133068138294319235841 ringDimension endEvent preFoldEvent resultEvent
      25317157507886064950797272225391822339692950454324 owner rawTerms .large 107405
      summary none Cert.statementResidual := by
  refine ⟨proofValid, ?_⟩
  refine ⟨rfl, ?_⟩
  refine ⟨rfl, ?_⟩
  refine ⟨rfl, ?_⟩
  refine ⟨rfl, ?_⟩
  refine ⟨rfl, ?_⟩
  refine ⟨⟨0, resultAt, preFoldAt, invocationEndAt⟩, ?_⟩
  change ∀ selector, selectorMinimum ≤ selector → selector < selectorMaximum →
    ∀ witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841, _
  intro selector selectorLower selectorUpper witness
  exact ⟨by
    simpa [actual, Cert.statementResidual] using
      invocationEndClaimSound selector selectorLower selectorUpper witness, by
    simpa [actual, Cert.statementResidual] using
      strictBoundSound selector selectorLower selectorUpper witness⟩

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticFinal
