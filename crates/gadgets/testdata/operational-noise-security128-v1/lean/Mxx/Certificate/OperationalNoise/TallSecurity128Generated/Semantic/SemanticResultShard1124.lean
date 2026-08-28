import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1124
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1123

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult158849
def owner : Owner := ⟨.program ⟨257⟩, ⟨22029⟩⟩
def rawTerms : List Term := Proof.Events620.exact158849RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158849
def producerEvent : Nat := 158848
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158849.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 158461, .finite 51, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult158849

namespace SemanticResult158872
def owner : Owner := ⟨.program ⟨257⟩, ⟨18809⟩⟩
def rawTerms : List Term := Proof.Events620.exact158872RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158872
def producerEvent : Nat := 158871
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158872.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 158461, .finite 48, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult158872

namespace SemanticResult158895
def owner : Owner := ⟨.program ⟨257⟩, ⟨15987⟩⟩
def rawTerms : List Term := Proof.Events620.exact158895RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158895
def producerEvent : Nat := 158894
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158895.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 158461, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult158895

namespace SemanticResult158899
def owner : Owner := ⟨.program ⟨257⟩, ⟨18810⟩⟩
def rawTerms : List Term := Proof.Events620.exact158899RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158899
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158899.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 158896) (rightBinding := 158897)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15987⟩) (rightExpression := ⟨18809⟩)
    (transferEvent := 158898)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult158895.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult158872.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult158899

namespace SemanticResult158903
def owner : Owner := ⟨.program ⟨257⟩, ⟨22030⟩⟩
def rawTerms : List Term := Proof.Events620.exact158903RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158903
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158903.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 158900) (rightBinding := 158901)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18810⟩) (rightExpression := ⟨22029⟩)
    (transferEvent := 158902)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult158899.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult158849.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult158903

namespace SemanticResult158907
def owner : Owner := ⟨.program ⟨257⟩, ⟨32050⟩⟩
def rawTerms : List Term := Proof.Events620.exact158907RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158907
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158907.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 158904) (rightBinding := 158905)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22030⟩) (rightExpression := ⟨32049⟩)
    (transferEvent := 158906)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult158903.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult158826.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult158907

namespace SemanticResult158911
def owner : Owner := ⟨.program ⟨257⟩, ⟨51105⟩⟩
def rawTerms : List Term := Proof.Events620.exact158911RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158911
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158911.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 158908) (rightBinding := 158909)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32050⟩) (rightExpression := ⟨51104⟩)
    (transferEvent := 158910)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult158907.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult158803.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult158911

namespace SemanticResult158915
def owner : Owner := ⟨.program ⟨257⟩, ⟨54085⟩⟩
def rawTerms : List Term := Proof.Events620.exact158915RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158915
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158915.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 158912) (rightBinding := 158913)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51105⟩) (rightExpression := ⟨54084⟩)
    (transferEvent := 158914)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult158911.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult158780.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult158915

namespace SemanticResult158919
def owner : Owner := ⟨.program ⟨257⟩, ⟨57065⟩⟩
def rawTerms : List Term := Proof.Events620.exact158919RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158919
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158919.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 158916) (rightBinding := 158917)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54085⟩) (rightExpression := ⟨57064⟩)
    (transferEvent := 158918)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult158915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult158757.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult158919

namespace SemanticResult158923
def owner : Owner := ⟨.program ⟨257⟩, ⟨60045⟩⟩
def rawTerms : List Term := Proof.Events620.exact158923RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158923
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158923.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 158920) (rightBinding := 158921)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57065⟩) (rightExpression := ⟨60044⟩)
    (transferEvent := 158922)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult158919.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult158734.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult158923

namespace SemanticResult158927
def owner : Owner := ⟨.program ⟨257⟩, ⟨63025⟩⟩
def rawTerms : List Term := Proof.Events620.exact158927RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158927
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158927.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 158924) (rightBinding := 158925)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60045⟩) (rightExpression := ⟨63024⟩)
    (transferEvent := 158926)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult158923.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult158711.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult158927

namespace SemanticResult158931
def owner : Owner := ⟨.program ⟨257⟩, ⟨66392⟩⟩
def rawTerms : List Term := Proof.Events620.exact158931RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158931
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158931.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 158928) (rightBinding := 158929)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63025⟩) (rightExpression := ⟨66391⟩)
    (transferEvent := 158930)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult158927.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult158688.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult158931

namespace SemanticResult158935
def owner : Owner := ⟨.program ⟨257⟩, ⟨66393⟩⟩
def rawTerms : List Term := Proof.Events620.exact158935RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158935
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158935.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 158932) (rightBinding := 158933)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66392⟩) (rightExpression := ⟨26580⟩)
    (transferEvent := 158934)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult158931.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult158665.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult158935

namespace SemanticResult158939
def owner : Owner := ⟨.program ⟨257⟩, ⟨66394⟩⟩
def rawTerms : List Term := Proof.Events620.exact158939RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158939
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158939.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 158936) (rightBinding := 158937)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66393⟩) (rightExpression := ⟨29260⟩)
    (transferEvent := 158938)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult158935.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult158642.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult158939

namespace SemanticResult158943
def owner : Owner := ⟨.program ⟨257⟩, ⟨66395⟩⟩
def rawTerms : List Term := Proof.Events620.exact158943RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158943
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158943.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 158940) (rightBinding := 158941)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66394⟩) (rightExpression := ⟨34924⟩)
    (transferEvent := 158942)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult158939.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult158619.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult158943

namespace SemanticResult158947
def owner : Owner := ⟨.program ⟨257⟩, ⟨66396⟩⟩
def rawTerms : List Term := Proof.Events620.exact158947RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 158947
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult158947.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 158944) (rightBinding := 158945)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66395⟩) (rightExpression := ⟨37604⟩)
    (transferEvent := 158946)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult158943.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult158596.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult158947

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
