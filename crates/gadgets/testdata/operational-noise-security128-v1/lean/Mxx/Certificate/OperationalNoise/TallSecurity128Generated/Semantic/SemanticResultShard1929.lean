import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1929
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1927
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1928

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult275919
def owner : Owner := ⟨.program ⟨257⟩, ⟨56965⟩⟩
def rawTerms : List Term := Proof.Events1077.exact275919RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275919
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275919.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 275916) (rightBinding := 275917)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53985⟩) (rightExpression := ⟨56964⟩)
    (transferEvent := 275918)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult275915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult275757.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult275919

namespace SemanticResult275923
def owner : Owner := ⟨.program ⟨257⟩, ⟨59945⟩⟩
def rawTerms : List Term := Proof.Events1077.exact275923RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275923
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275923.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 275920) (rightBinding := 275921)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56965⟩) (rightExpression := ⟨59944⟩)
    (transferEvent := 275922)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult275919.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult275734.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult275923

namespace SemanticResult275927
def owner : Owner := ⟨.program ⟨257⟩, ⟨62925⟩⟩
def rawTerms : List Term := Proof.Events1077.exact275927RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275927
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275927.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 275924) (rightBinding := 275925)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59945⟩) (rightExpression := ⟨62924⟩)
    (transferEvent := 275926)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult275923.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult275711.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult275927

namespace SemanticResult275931
def owner : Owner := ⟨.program ⟨257⟩, ⟨66020⟩⟩
def rawTerms : List Term := Proof.Events1077.exact275931RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275931
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275931.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 275928) (rightBinding := 275929)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62925⟩) (rightExpression := ⟨66019⟩)
    (transferEvent := 275930)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult275927.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult275688.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult275931

namespace SemanticResult275935
def owner : Owner := ⟨.program ⟨257⟩, ⟨66021⟩⟩
def rawTerms : List Term := Proof.Events1077.exact275935RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275935
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275935.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 275932) (rightBinding := 275933)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66020⟩) (rightExpression := ⟨26512⟩)
    (transferEvent := 275934)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult275931.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult275665.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult275935

namespace SemanticResult275939
def owner : Owner := ⟨.program ⟨257⟩, ⟨66022⟩⟩
def rawTerms : List Term := Proof.Events1077.exact275939RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275939
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275939.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 275936) (rightBinding := 275937)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66021⟩) (rightExpression := ⟨29192⟩)
    (transferEvent := 275938)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult275935.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult275642.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult275939

namespace SemanticResult275943
def owner : Owner := ⟨.program ⟨257⟩, ⟨66023⟩⟩
def rawTerms : List Term := Proof.Events1077.exact275943RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275943
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275943.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 275940) (rightBinding := 275941)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66022⟩) (rightExpression := ⟨34856⟩)
    (transferEvent := 275942)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult275939.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult275619.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult275943

namespace SemanticResult275947
def owner : Owner := ⟨.program ⟨257⟩, ⟨66024⟩⟩
def rawTerms : List Term := Proof.Events1077.exact275947RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275947
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275947.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 275944) (rightBinding := 275945)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66023⟩) (rightExpression := ⟨37536⟩)
    (transferEvent := 275946)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult275943.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult275596.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult275947

namespace SemanticResult275951
def owner : Owner := ⟨.program ⟨257⟩, ⟨66025⟩⟩
def rawTerms : List Term := Proof.Events1077.exact275951RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275951
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275951.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 275948) (rightBinding := 275949)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66024⟩) (rightExpression := ⟨40212⟩)
    (transferEvent := 275950)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult275947.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult275573.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult275951

namespace SemanticResult275955
def owner : Owner := ⟨.program ⟨257⟩, ⟨66026⟩⟩
def rawTerms : List Term := Proof.Events1077.exact275955RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275955
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275955.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 275952) (rightBinding := 275953)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66025⟩) (rightExpression := ⟨42892⟩)
    (transferEvent := 275954)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult275951.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult275550.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult275955

namespace SemanticResult275959
def owner : Owner := ⟨.program ⟨257⟩, ⟨66027⟩⟩
def rawTerms : List Term := Proof.Events1077.exact275959RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275959
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275959.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 275956) (rightBinding := 275957)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66026⟩) (rightExpression := ⟨45576⟩)
    (transferEvent := 275958)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult275955.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult275527.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult275959

namespace SemanticResult275963
def owner : Owner := ⟨.program ⟨257⟩, ⟨66028⟩⟩
def rawTerms : List Term := Proof.Events1077.exact275963RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275963
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275963.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 275960) (rightBinding := 275961)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66027⟩) (rightExpression := ⟨48256⟩)
    (transferEvent := 275962)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult275959.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult275504.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult275963

namespace SemanticResult275974
def owner : Owner := ⟨.program ⟨257⟩, ⟨68780⟩⟩
def rawTerms : List Term := Proof.Events1078.exact275974RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275974
def producerEvent : Nat := 275973
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275974.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 275461, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult275974

namespace SemanticResult275977
def owner : Owner := ⟨.program ⟨257⟩, ⟨70979⟩⟩
def rawTerms : List Term := Proof.Events1078.exact275977RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275977
def producerEvent : Nat := 275976
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275977.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 275461, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult275977

namespace SemanticResult275986
def owner : Owner := ⟨.program ⟨257⟩, ⟨69056⟩⟩
def rawTerms : List Term := Proof.Events1078.exact275986RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275986
def producerEvent : Nat := 275985
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275986.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 275984 .coefficient), 275461, .finite 1059, .identity (.predecessor 0 275984 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult275986

namespace SemanticResult275988
def owner : Owner := ⟨.program ⟨257⟩, ⟨6908⟩⟩
def rawTerms : List Term := Proof.Events1078.exact275988RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 275988
def producerEvent : Nat := 275987
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult275988.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 275461, .large, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult275988

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
