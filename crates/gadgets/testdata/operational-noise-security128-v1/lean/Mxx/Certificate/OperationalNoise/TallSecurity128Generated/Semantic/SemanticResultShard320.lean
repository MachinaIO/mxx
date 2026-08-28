import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard320
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard318
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard319

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult41919
def owner : Owner := ⟨.program ⟨257⟩, ⟨57293⟩⟩
def rawTerms : List Term := Proof.Events163.exact41919RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41919
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41919.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41916) (rightBinding := 41917)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54313⟩) (rightExpression := ⟨57292⟩)
    (transferEvent := 41918)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41757.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41919

namespace SemanticResult41923
def owner : Owner := ⟨.program ⟨257⟩, ⟨60273⟩⟩
def rawTerms : List Term := Proof.Events163.exact41923RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41923
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41923.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41920) (rightBinding := 41921)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57293⟩) (rightExpression := ⟨60272⟩)
    (transferEvent := 41922)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41919.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41734.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41923

namespace SemanticResult41927
def owner : Owner := ⟨.program ⟨257⟩, ⟨63253⟩⟩
def rawTerms : List Term := Proof.Events163.exact41927RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41927
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41927.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41924) (rightBinding := 41925)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60273⟩) (rightExpression := ⟨63252⟩)
    (transferEvent := 41926)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41923.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41711.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41927

namespace SemanticResult41931
def owner : Owner := ⟨.program ⟨257⟩, ⟨67232⟩⟩
def rawTerms : List Term := Proof.Events163.exact41931RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41931
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41931.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41928) (rightBinding := 41929)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63253⟩) (rightExpression := ⟨67231⟩)
    (transferEvent := 41930)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41927.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41688.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41931

namespace SemanticResult41935
def owner : Owner := ⟨.program ⟨257⟩, ⟨67233⟩⟩
def rawTerms : List Term := Proof.Events163.exact41935RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41935
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41935.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41932) (rightBinding := 41933)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67232⟩) (rightExpression := ⟨26736⟩)
    (transferEvent := 41934)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41931.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41665.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41935

namespace SemanticResult41939
def owner : Owner := ⟨.program ⟨257⟩, ⟨67234⟩⟩
def rawTerms : List Term := Proof.Events163.exact41939RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41939
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41939.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41936) (rightBinding := 41937)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67233⟩) (rightExpression := ⟨29416⟩)
    (transferEvent := 41938)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41935.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41642.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41939

namespace SemanticResult41943
def owner : Owner := ⟨.program ⟨257⟩, ⟨67235⟩⟩
def rawTerms : List Term := Proof.Events163.exact41943RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41943
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41943.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41940) (rightBinding := 41941)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67234⟩) (rightExpression := ⟨35080⟩)
    (transferEvent := 41942)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41939.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41619.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41943

namespace SemanticResult41947
def owner : Owner := ⟨.program ⟨257⟩, ⟨67236⟩⟩
def rawTerms : List Term := Proof.Events163.exact41947RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41947
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41947.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41944) (rightBinding := 41945)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67235⟩) (rightExpression := ⟨37760⟩)
    (transferEvent := 41946)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41943.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41596.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41947

namespace SemanticResult41951
def owner : Owner := ⟨.program ⟨257⟩, ⟨67237⟩⟩
def rawTerms : List Term := Proof.Events163.exact41951RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41951
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41951.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41948) (rightBinding := 41949)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67236⟩) (rightExpression := ⟨40436⟩)
    (transferEvent := 41950)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41947.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41573.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41951

namespace SemanticResult41955
def owner : Owner := ⟨.program ⟨257⟩, ⟨67238⟩⟩
def rawTerms : List Term := Proof.Events163.exact41955RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41955
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41955.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41952) (rightBinding := 41953)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67237⟩) (rightExpression := ⟨43116⟩)
    (transferEvent := 41954)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41951.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41550.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41955

namespace SemanticResult41959
def owner : Owner := ⟨.program ⟨257⟩, ⟨67239⟩⟩
def rawTerms : List Term := Proof.Events163.exact41959RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41959
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41959.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41956) (rightBinding := 41957)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67238⟩) (rightExpression := ⟨45800⟩)
    (transferEvent := 41958)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41955.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41527.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41959

namespace SemanticResult41963
def owner : Owner := ⟨.program ⟨257⟩, ⟨67240⟩⟩
def rawTerms : List Term := Proof.Events163.exact41963RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41963
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41963.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41960) (rightBinding := 41961)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67239⟩) (rightExpression := ⟨48480⟩)
    (transferEvent := 41962)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41959.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41504.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41963

namespace SemanticResult41974
def owner : Owner := ⟨.program ⟨257⟩, ⟨68884⟩⟩
def rawTerms : List Term := Proof.Events163.exact41974RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41974
def producerEvent : Nat := 41973
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41974.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 41461, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult41974

namespace SemanticResult41977
def owner : Owner := ⟨.program ⟨257⟩, ⟨71534⟩⟩
def rawTerms : List Term := Proof.Events163.exact41977RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41977
def producerEvent : Nat := 41976
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41977.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 41461, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult41977

namespace SemanticResult41986
def owner : Owner := ⟨.program ⟨257⟩, ⟨69124⟩⟩
def rawTerms : List Term := Proof.Events164.exact41986RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41986
def producerEvent : Nat := 41985
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41986.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 41984 .coefficient), 41461, .finite 1059, .identity (.predecessor 0 41984 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult41986

namespace SemanticResult41988
def owner : Owner := ⟨.program ⟨257⟩, ⟨6908⟩⟩
def rawTerms : List Term := Proof.Events164.exact41988RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41988
def producerEvent : Nat := 41987
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult41988.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 41461, .large, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult41988

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
