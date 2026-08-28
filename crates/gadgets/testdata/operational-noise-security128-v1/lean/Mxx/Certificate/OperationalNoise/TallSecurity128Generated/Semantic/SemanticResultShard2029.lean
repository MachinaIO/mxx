import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard2029
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard2028

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult290415
def owner : Owner := ⟨.program ⟨257⟩, ⟨31992⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290415RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290415
def producerEvent : Nat := 290414
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290415.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 290050, .finite 55, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult290415

namespace SemanticResult290438
def owner : Owner := ⟨.program ⟨257⟩, ⟨21972⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290438RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290438
def producerEvent : Nat := 290437
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290438.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 290050, .finite 51, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult290438

namespace SemanticResult290461
def owner : Owner := ⟨.program ⟨257⟩, ⟨18752⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290461RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290461
def producerEvent : Nat := 290460
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290461.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 290050, .finite 48, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult290461

namespace SemanticResult290484
def owner : Owner := ⟨.program ⟨257⟩, ⟨15939⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290484RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290484
def producerEvent : Nat := 290483
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290484.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 290050, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult290484

namespace SemanticResult290488
def owner : Owner := ⟨.program ⟨257⟩, ⟨18753⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290488RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290488
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290488.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 290485) (rightBinding := 290486)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15939⟩) (rightExpression := ⟨18752⟩)
    (transferEvent := 290487)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult290484.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult290461.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult290488

namespace SemanticResult290492
def owner : Owner := ⟨.program ⟨257⟩, ⟨21973⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290492RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290492
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290492.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 290489) (rightBinding := 290490)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18753⟩) (rightExpression := ⟨21972⟩)
    (transferEvent := 290491)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult290488.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult290438.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult290492

namespace SemanticResult290496
def owner : Owner := ⟨.program ⟨257⟩, ⟨31993⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290496RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290496
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290496.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 290493) (rightBinding := 290494)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21973⟩) (rightExpression := ⟨31992⟩)
    (transferEvent := 290495)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult290492.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult290415.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult290496

namespace SemanticResult290500
def owner : Owner := ⟨.program ⟨257⟩, ⟨51048⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290500RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290500
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290500.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 290497) (rightBinding := 290498)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨31993⟩) (rightExpression := ⟨51047⟩)
    (transferEvent := 290499)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult290496.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult290392.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult290500

namespace SemanticResult290504
def owner : Owner := ⟨.program ⟨257⟩, ⟨54028⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290504RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290504
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290504.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 290501) (rightBinding := 290502)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51048⟩) (rightExpression := ⟨54027⟩)
    (transferEvent := 290503)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult290500.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult290369.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult290504

namespace SemanticResult290508
def owner : Owner := ⟨.program ⟨257⟩, ⟨57008⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290508RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290508
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290508.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 290505) (rightBinding := 290506)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54028⟩) (rightExpression := ⟨57007⟩)
    (transferEvent := 290507)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult290504.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult290346.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult290508

namespace SemanticResult290512
def owner : Owner := ⟨.program ⟨257⟩, ⟨59988⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290512RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290512
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290512.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 290509) (rightBinding := 290510)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57008⟩) (rightExpression := ⟨59987⟩)
    (transferEvent := 290511)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult290508.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult290323.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult290512

namespace SemanticResult290516
def owner : Owner := ⟨.program ⟨257⟩, ⟨62968⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290516RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290516
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290516.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 290513) (rightBinding := 290514)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59988⟩) (rightExpression := ⟨62967⟩)
    (transferEvent := 290515)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult290512.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult290300.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult290516

namespace SemanticResult290520
def owner : Owner := ⟨.program ⟨257⟩, ⟨66182⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290520RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290520
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290520.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 290517) (rightBinding := 290518)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62968⟩) (rightExpression := ⟨66181⟩)
    (transferEvent := 290519)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult290516.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult290277.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult290520

namespace SemanticResult290524
def owner : Owner := ⟨.program ⟨257⟩, ⟨66183⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290524RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290524
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290524.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 290521) (rightBinding := 290522)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66182⟩) (rightExpression := ⟨26541⟩)
    (transferEvent := 290523)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult290520.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult290254.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult290524

namespace SemanticResult290528
def owner : Owner := ⟨.program ⟨257⟩, ⟨66184⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290528RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290528
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290528.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 290525) (rightBinding := 290526)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66183⟩) (rightExpression := ⟨29221⟩)
    (transferEvent := 290527)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult290524.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult290231.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult290528

namespace SemanticResult290532
def owner : Owner := ⟨.program ⟨257⟩, ⟨66185⟩⟩
def rawTerms : List Term := Proof.Events1134.exact290532RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 290532
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult290532.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 290529) (rightBinding := 290530)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66184⟩) (rightExpression := ⟨34885⟩)
    (transferEvent := 290531)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult290528.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult290208.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult290532

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
