import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResult
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.StatementResidual
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Proof

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticFinal

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

def owner : Owner := ⟨.program ⟨257⟩, ⟨71547⟩⟩
def resultEvent : Nat := 308622
def preFoldEvent : Nat := 308623
def endEvent : Nat := 308624
def rawTerms : List Term := []
def summary : Bound := .finite 146340160251294585514145619529726732840708448851938772156114662662487408692
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.statementResidual (some selector) witness

theorem resultAt : history.lookup resultEvent = some
    ⟨.resultExact owner rawTerms .large 308203
      summary none, 0⟩ := by
  rfl

theorem resultClaimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact SemanticResult.resultClaimSound selector selectorLower selectorUpper witness

theorem preFoldAt : history.lookup preFoldEvent = some
    ⟨.preFoldPolynomial resultEvent rawTerms summary
      (some (.result resultEvent .summary)), 0⟩ := by
  rfl

theorem invocationEndAt : history.lookup endEvent = some
    ⟨.invocationEndExact owner preFoldEvent rawTerms .large
      308203 summary none, 0⟩ := by
  rfl

theorem invocationEndClaimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact (rawTerms.map Term.toExact) summary) := by
  exact invocationEndSound 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
    (rawTerms.map Term.toExact) (rawTerms.map Term.toExact) summary summary
    (resultClaimSound selector selectorLower selectorUpper witness).claim rfl rfl

theorem strictBoundSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    2 * 268369921 * centeredNorm 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 (actual selector witness) < 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 := by
  apply finalStrictBound_of_empty_finite_claim 268369921 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env
    (actual selector witness) 146340160251294585514145619529726732840708448851938772156114662662487408692
  · simpa [rawTerms, summary] using
      invocationEndClaimSound selector selectorLower selectorUpper witness
  · decide
  · decide

theorem fixedSemanticSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    history.lookup resultEvent = some
        ⟨.resultExact owner rawTerms .large 308203
          summary none, 0⟩ ∧
      history.lookup preFoldEvent = some
        ⟨.preFoldPolynomial resultEvent rawTerms summary
          (some (.result resultEvent .summary)), 0⟩ ∧
      history.lookup endEvent = some
        ⟨.invocationEndExact owner preFoldEvent rawTerms .large
          308203 summary none, 0⟩ ∧
      ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
        (.exact (rawTerms.map Term.toExact) summary) ∧
      2 * 268369921 * centeredNorm 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 (actual selector witness) < 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 := by
  exact ⟨resultAt, preFoldAt, invocationEndAt,
    invocationEndClaimSound selector selectorLower selectorUpper witness,
    strictBoundSound selector selectorLower selectorUpper witness⟩

theorem fixedAcceptance :
    OperationalCertificateAccepted document history 268369921 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 ringDimension endEvent preFoldEvent resultEvent
      146340160251294585514145619529726732840708448851938772156114662662487408692 owner rawTerms .large 308203
      summary none Cert.statementResidual := by
  refine ⟨proofValid, ?_⟩
  refine ⟨rfl, ?_⟩
  refine ⟨rfl, ?_⟩
  refine ⟨rfl, ?_⟩
  refine ⟨rfl, ?_⟩
  refine ⟨rfl, ?_⟩
  refine ⟨⟨0, resultAt, preFoldAt, invocationEndAt⟩, ?_⟩
  change ∀ selector, selectorMinimum ≤ selector → selector < selectorMaximum →
    ∀ witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817, _
  intro selector selectorLower selectorUpper witness
  exact ⟨by
    simpa [actual, Cert.statementResidual] using
      invocationEndClaimSound selector selectorLower selectorUpper witness, by
    simpa [actual, Cert.statementResidual] using
      strictBoundSound selector selectorLower selectorUpper witness⟩

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticFinal
