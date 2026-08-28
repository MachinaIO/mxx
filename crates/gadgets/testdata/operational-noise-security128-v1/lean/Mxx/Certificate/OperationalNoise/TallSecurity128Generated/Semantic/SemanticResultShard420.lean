import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard420
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard419

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult56451
def owner : Owner := ⟨.program ⟨257⟩, ⟨32258⟩⟩
def rawTerms : List Term := Proof.Events220.exact56451RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56451
def producerEvent : Nat := 56450
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56451.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 56086, .finite 55, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult56451

namespace SemanticResult56474
def owner : Owner := ⟨.program ⟨257⟩, ⟨22238⟩⟩
def rawTerms : List Term := Proof.Events220.exact56474RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56474
def producerEvent : Nat := 56473
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56474.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 56086, .finite 51, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult56474

namespace SemanticResult56497
def owner : Owner := ⟨.program ⟨257⟩, ⟨19018⟩⟩
def rawTerms : List Term := Proof.Events220.exact56497RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56497
def producerEvent : Nat := 56496
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56497.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 56086, .finite 48, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult56497

namespace SemanticResult56520
def owner : Owner := ⟨.program ⟨257⟩, ⟨16163⟩⟩
def rawTerms : List Term := Proof.Events220.exact56520RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56520
def producerEvent : Nat := 56519
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56520.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 56086, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult56520

namespace SemanticResult56524
def owner : Owner := ⟨.program ⟨257⟩, ⟨19019⟩⟩
def rawTerms : List Term := Proof.Events220.exact56524RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56524
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56524.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 56521) (rightBinding := 56522)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16163⟩) (rightExpression := ⟨19018⟩)
    (transferEvent := 56523)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56520.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56497.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult56524

namespace SemanticResult56528
def owner : Owner := ⟨.program ⟨257⟩, ⟨22239⟩⟩
def rawTerms : List Term := Proof.Events220.exact56528RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56528
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56528.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 56525) (rightBinding := 56526)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨19019⟩) (rightExpression := ⟨22238⟩)
    (transferEvent := 56527)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56524.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56474.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult56528

namespace SemanticResult56532
def owner : Owner := ⟨.program ⟨257⟩, ⟨32259⟩⟩
def rawTerms : List Term := Proof.Events220.exact56532RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56532
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56532.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 56529) (rightBinding := 56530)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22239⟩) (rightExpression := ⟨32258⟩)
    (transferEvent := 56531)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56528.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56451.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult56532

namespace SemanticResult56536
def owner : Owner := ⟨.program ⟨257⟩, ⟨51314⟩⟩
def rawTerms : List Term := Proof.Events220.exact56536RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56536
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56536.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 56533) (rightBinding := 56534)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32259⟩) (rightExpression := ⟨51313⟩)
    (transferEvent := 56535)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56532.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56428.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult56536

namespace SemanticResult56540
def owner : Owner := ⟨.program ⟨257⟩, ⟨54294⟩⟩
def rawTerms : List Term := Proof.Events220.exact56540RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56540
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56540.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 56537) (rightBinding := 56538)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51314⟩) (rightExpression := ⟨54293⟩)
    (transferEvent := 56539)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56536.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56405.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult56540

namespace SemanticResult56544
def owner : Owner := ⟨.program ⟨257⟩, ⟨57274⟩⟩
def rawTerms : List Term := Proof.Events220.exact56544RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56544
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56544.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 56541) (rightBinding := 56542)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54294⟩) (rightExpression := ⟨57273⟩)
    (transferEvent := 56543)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56382.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult56544

namespace SemanticResult56548
def owner : Owner := ⟨.program ⟨257⟩, ⟨60254⟩⟩
def rawTerms : List Term := Proof.Events220.exact56548RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56548
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56548.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 56545) (rightBinding := 56546)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57274⟩) (rightExpression := ⟨60253⟩)
    (transferEvent := 56547)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56544.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56359.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult56548

namespace SemanticResult56552
def owner : Owner := ⟨.program ⟨257⟩, ⟨63234⟩⟩
def rawTerms : List Term := Proof.Events220.exact56552RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56552
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56552.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 56549) (rightBinding := 56550)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60254⟩) (rightExpression := ⟨63233⟩)
    (transferEvent := 56551)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56548.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56336.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult56552

namespace SemanticResult56556
def owner : Owner := ⟨.program ⟨257⟩, ⟨67162⟩⟩
def rawTerms : List Term := Proof.Events220.exact56556RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56556
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56556.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 56553) (rightBinding := 56554)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63234⟩) (rightExpression := ⟨67161⟩)
    (transferEvent := 56555)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56552.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56313.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult56556

namespace SemanticResult56560
def owner : Owner := ⟨.program ⟨257⟩, ⟨67163⟩⟩
def rawTerms : List Term := Proof.Events220.exact56560RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56560
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56560.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 56557) (rightBinding := 56558)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67162⟩) (rightExpression := ⟨26723⟩)
    (transferEvent := 56559)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56556.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56290.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult56560

namespace SemanticResult56564
def owner : Owner := ⟨.program ⟨257⟩, ⟨67164⟩⟩
def rawTerms : List Term := Proof.Events220.exact56564RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56564
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56564.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 56561) (rightBinding := 56562)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67163⟩) (rightExpression := ⟨29403⟩)
    (transferEvent := 56563)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56560.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56267.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult56564

namespace SemanticResult56568
def owner : Owner := ⟨.program ⟨257⟩, ⟨67165⟩⟩
def rawTerms : List Term := Proof.Events220.exact56568RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56568
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult56568.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 56565) (rightBinding := 56566)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67164⟩) (rightExpression := ⟨35067⟩)
    (transferEvent := 56567)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56564.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56244.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult56568

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
