import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard123
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard124
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard125
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard126
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard127

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult15896
def owner : Owner := ⟨.program ⟨257⟩, ⟨7292⟩⟩
def rawTerms : List Term := Proof.Events062.exact15896RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15896
def producerEvent : Nat := 15895
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15896.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 15894 .coefficient), 0, .large, .identity (.predecessor 0 15894 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult15896

namespace SemanticResult15901
def owner : Owner := ⟨.program ⟨257⟩, ⟨9128⟩⟩
def rawTerms : List Term := Proof.Events062.exact15901RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15901
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15901.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge15900.working .exactZero) := by
  apply operatorSubMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge15900.frameStart)
    (transferEvent := 15899) (owner := owner)
    (leftResult := 15896) (rightResult := 15896)
    (working := LeftOperatorMerge15900.working)
    (reconstruction := LeftOperatorMerge15900.reconstruction)
    (leftReference := .predecessor 0 15897 .coefficient) (rightReference := .predecessor 1 15898 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult15896.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15896.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge15900.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult15901

namespace SemanticResult15905
def owner : Owner := ⟨.program ⟨257⟩, ⟨9129⟩⟩
def rawTerms : List Term := Proof.Events062.exact15905RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15905
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15905.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15902) (rightBinding := 15903)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9128⟩) (rightExpression := ⟨9109⟩)
    (transferEvent := 15904)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15901.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15890.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15905

namespace SemanticResult15909
def owner : Owner := ⟨.program ⟨257⟩, ⟨9130⟩⟩
def rawTerms : List Term := Proof.Events062.exact15909RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15909
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15909.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15906) (rightBinding := 15907)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9129⟩) (rightExpression := ⟨9110⟩)
    (transferEvent := 15908)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15905.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15870.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15909

namespace SemanticResult15913
def owner : Owner := ⟨.program ⟨257⟩, ⟨9131⟩⟩
def rawTerms : List Term := Proof.Events062.exact15913RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15913
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15913.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15910) (rightBinding := 15911)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9130⟩) (rightExpression := ⟨9111⟩)
    (transferEvent := 15912)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15909.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15850.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15913

namespace SemanticResult15917
def owner : Owner := ⟨.program ⟨257⟩, ⟨9132⟩⟩
def rawTerms : List Term := Proof.Events062.exact15917RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15917
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15917.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15914) (rightBinding := 15915)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9131⟩) (rightExpression := ⟨9112⟩)
    (transferEvent := 15916)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15913.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15830.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15917

namespace SemanticResult15921
def owner : Owner := ⟨.program ⟨257⟩, ⟨9133⟩⟩
def rawTerms : List Term := Proof.Events062.exact15921RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15921
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15921.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15918) (rightBinding := 15919)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9132⟩) (rightExpression := ⟨9113⟩)
    (transferEvent := 15920)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15917.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15810.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15921

namespace SemanticResult15925
def owner : Owner := ⟨.program ⟨257⟩, ⟨9134⟩⟩
def rawTerms : List Term := Proof.Events062.exact15925RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15925
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15925.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15922) (rightBinding := 15923)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9133⟩) (rightExpression := ⟨9114⟩)
    (transferEvent := 15924)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15921.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15790.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15925

namespace SemanticResult15929
def owner : Owner := ⟨.program ⟨257⟩, ⟨9135⟩⟩
def rawTerms : List Term := Proof.Events062.exact15929RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15929
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15929.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15926) (rightBinding := 15927)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9134⟩) (rightExpression := ⟨9115⟩)
    (transferEvent := 15928)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15925.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15770.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15929

namespace SemanticResult15933
def owner : Owner := ⟨.program ⟨257⟩, ⟨9136⟩⟩
def rawTerms : List Term := Proof.Events062.exact15933RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15933
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15933.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15930) (rightBinding := 15931)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9135⟩) (rightExpression := ⟨9116⟩)
    (transferEvent := 15932)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15929.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15750.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15933

namespace SemanticResult15937
def owner : Owner := ⟨.program ⟨257⟩, ⟨9137⟩⟩
def rawTerms : List Term := Proof.Events062.exact15937RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15937
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15937.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15934) (rightBinding := 15935)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9136⟩) (rightExpression := ⟨9117⟩)
    (transferEvent := 15936)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15933.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15730.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15937

namespace SemanticResult15941
def owner : Owner := ⟨.program ⟨257⟩, ⟨9138⟩⟩
def rawTerms : List Term := Proof.Events062.exact15941RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15941
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15941.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15938) (rightBinding := 15939)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9137⟩) (rightExpression := ⟨9118⟩)
    (transferEvent := 15940)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15937.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15710.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15941

namespace SemanticResult15945
def owner : Owner := ⟨.program ⟨257⟩, ⟨9139⟩⟩
def rawTerms : List Term := Proof.Events062.exact15945RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15945
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15945.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15942) (rightBinding := 15943)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9138⟩) (rightExpression := ⟨9119⟩)
    (transferEvent := 15944)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15941.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15690.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15945

namespace SemanticResult15949
def owner : Owner := ⟨.program ⟨257⟩, ⟨9140⟩⟩
def rawTerms : List Term := Proof.Events062.exact15949RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15949
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15949.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15946) (rightBinding := 15947)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9139⟩) (rightExpression := ⟨9120⟩)
    (transferEvent := 15948)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15945.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15670.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15949

namespace SemanticResult15953
def owner : Owner := ⟨.program ⟨257⟩, ⟨9141⟩⟩
def rawTerms : List Term := Proof.Events062.exact15953RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15953
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15953.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15950) (rightBinding := 15951)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9140⟩) (rightExpression := ⟨9121⟩)
    (transferEvent := 15952)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15949.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15650.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15953

namespace SemanticResult15957
def owner : Owner := ⟨.program ⟨257⟩, ⟨9142⟩⟩
def rawTerms : List Term := Proof.Events062.exact15957RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15957
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult15957.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15954) (rightBinding := 15955)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9141⟩) (rightExpression := ⟨9122⟩)
    (transferEvent := 15956)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15953.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15630.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15957

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
