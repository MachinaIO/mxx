import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard751
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard734
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard735
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard737
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard738
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard739
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard741
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard742
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard743
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard745
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard746
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard748
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard749
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard750

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult104655
def owner : Owner := ⟨.program ⟨257⟩, ⟨9949⟩⟩
def rawTerms : List Term := Proof.Events408.exact104655RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 104655
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104655.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 104652) (rightBinding := 104653)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9926⟩) (rightExpression := ⟨9948⟩)
    (transferEvent := 104654)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104651.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult104646.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult104655

namespace SemanticResult104661
def owner : Owner := ⟨.program ⟨257⟩, ⟨9950⟩⟩
def rawTerms : List Term := Proof.Events408.exact104661RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 104661
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104661.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 104658) (survivorTransfer := 104659)
    (survivorEvent := 104660) (resultEvent := resultEvent)
    (rightCoefficientProducer := 31515)
    (owner := owner) (leftOwner := SemanticResult104655.owner)
    (rightOwner := SemanticResult31516.owner)
    (leftResult := 104655) (rightResult := 31516)
    (leftBinding := 104656) (rightBinding := 104657)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9949⟩) (rightExpression := ⟨118⟩)
    (leftActual := SemanticResult104655.actual selector witness)
    (rightActual := SemanticResult31516.actual selector witness)
    (leftRaw := SemanticResult104655.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound31515.actual selector witness)
    (survivorMagnitude := LeftBound104659.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104655.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31516.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)
  · exact LeftBound104659.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult104661

namespace SemanticResult104668
def owner : Owner := ⟨.program ⟨257⟩, ⟨9951⟩⟩
def rawTerms : List Term := Proof.Events408.exact104668RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 104668
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104668.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge104665.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult104661.owner)
    (rightOwner := SemanticResult104661.owner)
    (leftResult := 104661) (rightResult := 104661)
    (leftActual := SemanticResult104661.actual selector witness)
    (rightActual := SemanticResult104661.actual selector witness)
    (leftRaw := SemanticResult104661.rawTerms)
    (rightRaw := SemanticResult104661.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 104662) (rightBinding := 104663)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9950⟩) (rightExpression := ⟨9950⟩)
    (coefficientTransfer := 104664) (summaryTransfer := 104667)
    (base := LeftOperatorMerge104665.base)
    (reconstruction := LeftOperatorMerge104665.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104661.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult104661.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge104665.operationAgreement
  · rfl
  · decide
end SemanticResult104668

namespace SemanticResult104673
def owner : Owner := ⟨.program ⟨257⟩, ⟨17899⟩⟩
def rawTerms : List Term := Proof.Events408.exact104673RawTerms
def summary : Bound := (.finite 345624685687166110058245054666339432529972)
def resultEvent : Nat := 104673
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104673.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult104668.owner)
    (rightOwner := SemanticResult104641.owner)
    (leftResult := 104668) (rightResult := 104641)
    (leftActual := SemanticResult104668.actual selector witness)
    (rightActual := SemanticResult104641.actual selector witness)
    (leftRaw := SemanticResult104668.rawTerms)
    (rightRaw := SemanticResult104641.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 345624685687166110058245054666339432529920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 104669) (rightBinding := 104670)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9951⟩) (rightExpression := ⟨17898⟩)
    (transferEvent := 104671) (summaryTransferEvent := 104672)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104668.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult104641.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult104673

namespace SemanticResult104678
def owner : Owner := ⟨.program ⟨257⟩, ⟨20805⟩⟩
def rawTerms : List Term := Proof.Events408.exact104678RawTerms
def summary : Bound := (.finite 691250426059631610003352154589745737891892)
def resultEvent : Nat := 104678
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104678.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult104673.owner)
    (rightOwner := SemanticResult104429.owner)
    (leftResult := 104673) (rightResult := 104429)
    (leftActual := SemanticResult104673.actual selector witness)
    (rightActual := SemanticResult104429.actual selector witness)
    (leftRaw := SemanticResult104673.rawTerms)
    (rightRaw := SemanticResult104429.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 345624685687166110058245054666339432529972)
    (rightMaximum := 345625740372465499945107099923406305361920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 104674) (rightBinding := 104675)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17899⟩) (rightExpression := ⟨20804⟩)
    (transferEvent := 104676) (summaryTransferEvent := 104677)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104673.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult104429.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult104678

namespace SemanticResult104683
def owner : Owner := ⟨.program ⟨257⟩, ⟨24025⟩⟩
def rawTerms : List Term := Proof.Events408.exact104683RawTerms
def summary : Bound := (.finite 1036877221117396499835321299770218916085812)
def resultEvent : Nat := 104683
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104683.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult104678.owner)
    (rightOwner := SemanticResult104217.owner)
    (leftResult := 104678) (rightResult := 104217)
    (leftActual := SemanticResult104678.actual selector witness)
    (rightActual := SemanticResult104217.actual selector witness)
    (leftRaw := SemanticResult104678.rawTerms)
    (rightRaw := SemanticResult104217.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 691250426059631610003352154589745737891892)
    (rightMaximum := 345626795057764889831969145180473178193920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 104679) (rightBinding := 104680)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20805⟩) (rightExpression := ⟨24024⟩)
    (transferEvent := 104681) (summaryTransferEvent := 104682)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104678.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult104217.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult104683

namespace SemanticResult104688
def owner : Owner := ⟨.program ⟨257⟩, ⟨34045⟩⟩
def rawTerms : List Term := Proof.Events408.exact104688RawTerms
def summary : Bound := (.finite 1382506125545760169441014535464825839943732)
def resultEvent : Nat := 104688
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104688.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult104683.owner)
    (rightOwner := SemanticResult104005.owner)
    (leftResult := 104683) (rightResult := 104005)
    (leftActual := SemanticResult104683.actual selector witness)
    (rightActual := SemanticResult104005.actual selector witness)
    (leftRaw := SemanticResult104683.rawTerms)
    (rightRaw := SemanticResult104005.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1036877221117396499835321299770218916085812)
    (rightMaximum := 345628904428363669605693235694606923857920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 104684) (rightBinding := 104685)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨24025⟩) (rightExpression := ⟨34044⟩)
    (transferEvent := 104686) (summaryTransferEvent := 104687)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104683.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult104005.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult104688

namespace SemanticResult104693
def owner : Owner := ⟨.program ⟨257⟩, ⟨53105⟩⟩
def rawTerms : List Term := Proof.Events408.exact104693RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 104693
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104693.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult104688.owner)
    (rightOwner := SemanticResult103793.owner)
    (leftResult := 104688) (rightResult := 103793)
    (leftActual := SemanticResult104688.actual selector witness)
    (rightActual := SemanticResult103793.actual selector witness)
    (leftRaw := SemanticResult104688.rawTerms)
    (rightRaw := SemanticResult103793.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 104689) (rightBinding := 104690)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨34045⟩) (rightExpression := ⟨53104⟩)
    (transferEvent := 104691) (summaryTransferEvent := 104692)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104688.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103793.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult104693

namespace SemanticResult104698
def owner : Owner := ⟨.program ⟨257⟩, ⟨56085⟩⟩
def rawTerms : List Term := Proof.Events408.exact104698RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 104698
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104698.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult104693.owner)
    (rightOwner := SemanticResult103581.owner)
    (leftResult := 104693) (rightResult := 103581)
    (leftActual := SemanticResult104693.actual selector witness)
    (rightActual := SemanticResult103581.actual selector witness)
    (leftRaw := SemanticResult104693.rawTerms)
    (rightRaw := SemanticResult103581.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 104694) (rightBinding := 104695)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53105⟩) (rightExpression := ⟨56084⟩)
    (transferEvent := 104696) (summaryTransferEvent := 104697)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104693.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103581.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult104698

namespace SemanticResult104703
def owner : Owner := ⟨.program ⟨257⟩, ⟨59065⟩⟩
def rawTerms : List Term := Proof.Events408.exact104703RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 104703
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104703.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult104698.owner)
    (rightOwner := SemanticResult103369.owner)
    (leftResult := 104698) (rightResult := 103369)
    (leftActual := SemanticResult104698.actual selector witness)
    (rightActual := SemanticResult103369.actual selector witness)
    (leftRaw := SemanticResult104698.rawTerms)
    (rightRaw := SemanticResult103369.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 104699) (rightBinding := 104700)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56085⟩) (rightExpression := ⟨59064⟩)
    (transferEvent := 104701) (summaryTransferEvent := 104702)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104698.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103369.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult104703

namespace SemanticResult104708
def owner : Owner := ⟨.program ⟨257⟩, ⟨62045⟩⟩
def rawTerms : List Term := Proof.Events409.exact104708RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 104708
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104708.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult104703.owner)
    (rightOwner := SemanticResult103157.owner)
    (leftResult := 104703) (rightResult := 103157)
    (leftActual := SemanticResult104703.actual selector witness)
    (rightActual := SemanticResult103157.actual selector witness)
    (leftRaw := SemanticResult104703.rawTerms)
    (rightRaw := SemanticResult103157.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 104704) (rightBinding := 104705)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59065⟩) (rightExpression := ⟨62044⟩)
    (transferEvent := 104706) (summaryTransferEvent := 104707)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104703.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103157.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult104708

namespace SemanticResult104713
def owner : Owner := ⟨.program ⟨257⟩, ⟨65025⟩⟩
def rawTerms : List Term := Proof.Events409.exact104713RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 104713
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104713.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult104708.owner)
    (rightOwner := SemanticResult102945.owner)
    (leftResult := 104708) (rightResult := 102945)
    (leftActual := SemanticResult104708.actual selector witness)
    (rightActual := SemanticResult102945.actual selector witness)
    (leftRaw := SemanticResult104708.rawTerms)
    (rightRaw := SemanticResult102945.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 104709) (rightBinding := 104710)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62045⟩) (rightExpression := ⟨65024⟩)
    (transferEvent := 104711) (summaryTransferEvent := 104712)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104708.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult102945.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult104713

namespace SemanticResult104718
def owner : Owner := ⟨.program ⟨257⟩, ⟨70562⟩⟩
def rawTerms : List Term := Proof.Events409.exact104718RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 104718
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104718.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult104713.owner)
    (rightOwner := SemanticResult102733.owner)
    (leftResult := 104713) (rightResult := 102733)
    (leftActual := SemanticResult104713.actual selector witness)
    (rightActual := SemanticResult102733.actual selector witness)
    (leftRaw := SemanticResult104713.rawTerms)
    (rightRaw := SemanticResult102733.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 104714) (rightBinding := 104715)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65025⟩) (rightExpression := ⟨70561⟩)
    (transferEvent := 104716) (summaryTransferEvent := 104717)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104713.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult102733.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult104718

namespace SemanticResult104723
def owner : Owner := ⟨.program ⟨257⟩, ⟨70563⟩⟩
def rawTerms : List Term := Proof.Events409.exact104723RawTerms
def summary : Bound := (.finite 3802007596962448506045899439491360353157172)
def resultEvent : Nat := 104723
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104723.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult104718.owner)
    (rightOwner := SemanticResult102521.owner)
    (leftResult := 104718) (rightResult := 102521)
    (leftActual := SemanticResult104718.actual selector witness)
    (rightActual := SemanticResult102521.actual selector witness)
    (leftRaw := SemanticResult104718.rawTerms)
    (rightRaw := SemanticResult102521.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3456353380086899479155517117627148481331252)
    (rightMaximum := 345654216875549026890382321864211871825920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 104719) (rightBinding := 104720)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70562⟩) (rightExpression := ⟨28412⟩)
    (transferEvent := 104721) (summaryTransferEvent := 104722)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104718.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult102521.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult104723

namespace SemanticResult104728
def owner : Owner := ⟨.program ⟨257⟩, ⟨70564⟩⟩
def rawTerms : List Term := Proof.Events409.exact104728RawTerms
def summary : Bound := (.finite 4147668141949793872257454032897973461975092)
def resultEvent : Nat := 104728
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104728.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult104723.owner)
    (rightOwner := SemanticResult102309.owner)
    (leftResult := 104723) (rightResult := 102309)
    (leftActual := SemanticResult104723.actual selector witness)
    (rightActual := SemanticResult102309.actual selector witness)
    (leftRaw := SemanticResult104723.rawTerms)
    (rightRaw := SemanticResult102309.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3802007596962448506045899439491360353157172)
    (rightMaximum := 345660544987345366211554593406613108817920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 104724) (rightBinding := 104725)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70563⟩) (rightExpression := ⟨31092⟩)
    (transferEvent := 104726) (summaryTransferEvent := 104727)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104723.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult102309.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult104728

namespace SemanticResult104733
def owner : Owner := ⟨.program ⟨257⟩, ⟨70565⟩⟩
def rawTerms : List Term := Proof.Events409.exact104733RawTerms
def summary : Bound := (.finite 4493332905678336798016456807332854062121012)
def resultEvent : Nat := 104733
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult104733.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult104728.owner)
    (rightOwner := SemanticResult102097.owner)
    (leftResult := 104728) (rightResult := 102097)
    (leftActual := SemanticResult104728.actual selector witness)
    (rightActual := SemanticResult102097.actual selector witness)
    (leftRaw := SemanticResult104728.rawTerms)
    (rightRaw := SemanticResult102097.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4147668141949793872257454032897973461975092)
    (rightMaximum := 345664763728542925759002774434880600145920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 104729) (rightBinding := 104730)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70564⟩) (rightExpression := ⟨36752⟩)
    (transferEvent := 104731) (summaryTransferEvent := 104732)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult104728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult102097.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult104733

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
