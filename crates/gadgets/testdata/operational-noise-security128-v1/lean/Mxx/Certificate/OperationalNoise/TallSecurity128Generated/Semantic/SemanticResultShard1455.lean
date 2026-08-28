import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1455
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1436
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1438
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1439
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1441
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1442
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1443
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1445
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1446
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1447
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1449
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1450
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1452
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1453
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1454

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult207036
def owner : Owner := ⟨.program ⟨257⟩, ⟨9414⟩⟩
def rawTerms : List Term := Proof.Events808.exact207036RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 207036
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207036.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 207033) (survivorTransfer := 207034)
    (survivorEvent := 207035) (resultEvent := resultEvent)
    (rightCoefficientProducer := 31515)
    (owner := owner) (leftOwner := SemanticResult207030.owner)
    (rightOwner := SemanticResult31516.owner)
    (leftResult := 207030) (rightResult := 31516)
    (leftBinding := 207031) (rightBinding := 207032)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9413⟩) (rightExpression := ⟨118⟩)
    (leftActual := SemanticResult207030.actual selector witness)
    (rightActual := SemanticResult31516.actual selector witness)
    (leftRaw := SemanticResult207030.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound31515.actual selector witness)
    (survivorMagnitude := LeftBound207034.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207030.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31516.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)
  · exact LeftBound207034.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult207036

namespace SemanticResult207043
def owner : Owner := ⟨.program ⟨257⟩, ⟨9483⟩⟩
def rawTerms : List Term := Proof.Events808.exact207043RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 207043
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207043.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge207040.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult207036.owner)
    (rightOwner := SemanticResult207036.owner)
    (leftResult := 207036) (rightResult := 207036)
    (leftActual := SemanticResult207036.actual selector witness)
    (rightActual := SemanticResult207036.actual selector witness)
    (leftRaw := SemanticResult207036.rawTerms)
    (rightRaw := SemanticResult207036.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 207037) (rightBinding := 207038)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9414⟩) (rightExpression := ⟨9414⟩)
    (coefficientTransfer := 207039) (summaryTransfer := 207042)
    (base := LeftOperatorMerge207040.base)
    (reconstruction := LeftOperatorMerge207040.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207036.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult207036.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge207040.operationAgreement
  · rfl
  · decide
end SemanticResult207043

namespace SemanticResult207048
def owner : Owner := ⟨.program ⟨257⟩, ⟨17815⟩⟩
def rawTerms : List Term := Proof.Events808.exact207048RawTerms
def summary : Bound := (.finite 345624685687166110058245054666339432529972)
def resultEvent : Nat := 207048
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207048.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult207043.owner)
    (rightOwner := SemanticResult207016.owner)
    (leftResult := 207043) (rightResult := 207016)
    (leftActual := SemanticResult207043.actual selector witness)
    (rightActual := SemanticResult207016.actual selector witness)
    (leftRaw := SemanticResult207043.rawTerms)
    (rightRaw := SemanticResult207016.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 345624685687166110058245054666339432529920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 207044) (rightBinding := 207045)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9483⟩) (rightExpression := ⟨17814⟩)
    (transferEvent := 207046) (summaryTransferEvent := 207047)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207043.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult207016.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult207048

namespace SemanticResult207053
def owner : Owner := ⟨.program ⟨257⟩, ⟨20712⟩⟩
def rawTerms : List Term := Proof.Events808.exact207053RawTerms
def summary : Bound := (.finite 691250426059631610003352154589745737891892)
def resultEvent : Nat := 207053
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207053.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult207048.owner)
    (rightOwner := SemanticResult206804.owner)
    (leftResult := 207048) (rightResult := 206804)
    (leftActual := SemanticResult207048.actual selector witness)
    (rightActual := SemanticResult206804.actual selector witness)
    (leftRaw := SemanticResult207048.rawTerms)
    (rightRaw := SemanticResult206804.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 345624685687166110058245054666339432529972)
    (rightMaximum := 345625740372465499945107099923406305361920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 207049) (rightBinding := 207050)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17815⟩) (rightExpression := ⟨20711⟩)
    (transferEvent := 207051) (summaryTransferEvent := 207052)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207048.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult206804.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult207053

namespace SemanticResult207058
def owner : Owner := ⟨.program ⟨257⟩, ⟨23932⟩⟩
def rawTerms : List Term := Proof.Events808.exact207058RawTerms
def summary : Bound := (.finite 1036877221117396499835321299770218916085812)
def resultEvent : Nat := 207058
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207058.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult207053.owner)
    (rightOwner := SemanticResult206592.owner)
    (leftResult := 207053) (rightResult := 206592)
    (leftActual := SemanticResult207053.actual selector witness)
    (rightActual := SemanticResult206592.actual selector witness)
    (leftRaw := SemanticResult207053.rawTerms)
    (rightRaw := SemanticResult206592.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 691250426059631610003352154589745737891892)
    (rightMaximum := 345626795057764889831969145180473178193920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 207054) (rightBinding := 207055)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20712⟩) (rightExpression := ⟨23931⟩)
    (transferEvent := 207056) (summaryTransferEvent := 207057)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207053.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult206592.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult207058

namespace SemanticResult207063
def owner : Owner := ⟨.program ⟨257⟩, ⟨33952⟩⟩
def rawTerms : List Term := Proof.Events808.exact207063RawTerms
def summary : Bound := (.finite 1382506125545760169441014535464825839943732)
def resultEvent : Nat := 207063
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207063.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult207058.owner)
    (rightOwner := SemanticResult206380.owner)
    (leftResult := 207058) (rightResult := 206380)
    (leftActual := SemanticResult207058.actual selector witness)
    (rightActual := SemanticResult206380.actual selector witness)
    (leftRaw := SemanticResult207058.rawTerms)
    (rightRaw := SemanticResult206380.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1036877221117396499835321299770218916085812)
    (rightMaximum := 345628904428363669605693235694606923857920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 207059) (rightBinding := 207060)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23932⟩) (rightExpression := ⟨33951⟩)
    (transferEvent := 207061) (summaryTransferEvent := 207062)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207058.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult206380.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult207063

namespace SemanticResult207068
def owner : Owner := ⟨.program ⟨257⟩, ⟨53012⟩⟩
def rawTerms : List Term := Proof.Events808.exact207068RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 207068
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207068.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult207063.owner)
    (rightOwner := SemanticResult206168.owner)
    (leftResult := 207063) (rightResult := 206168)
    (leftActual := SemanticResult207063.actual selector witness)
    (rightActual := SemanticResult206168.actual selector witness)
    (leftRaw := SemanticResult207063.rawTerms)
    (rightRaw := SemanticResult206168.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 207064) (rightBinding := 207065)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33952⟩) (rightExpression := ⟨53011⟩)
    (transferEvent := 207066) (summaryTransferEvent := 207067)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207063.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult206168.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult207068

namespace SemanticResult207073
def owner : Owner := ⟨.program ⟨257⟩, ⟨55992⟩⟩
def rawTerms : List Term := Proof.Events808.exact207073RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 207073
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207073.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult207068.owner)
    (rightOwner := SemanticResult205956.owner)
    (leftResult := 207068) (rightResult := 205956)
    (leftActual := SemanticResult207068.actual selector witness)
    (rightActual := SemanticResult205956.actual selector witness)
    (leftRaw := SemanticResult207068.rawTerms)
    (rightRaw := SemanticResult205956.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 207069) (rightBinding := 207070)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53012⟩) (rightExpression := ⟨55991⟩)
    (transferEvent := 207071) (summaryTransferEvent := 207072)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207068.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult205956.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult207073

namespace SemanticResult207078
def owner : Owner := ⟨.program ⟨257⟩, ⟨58972⟩⟩
def rawTerms : List Term := Proof.Events808.exact207078RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 207078
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207078.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult207073.owner)
    (rightOwner := SemanticResult205744.owner)
    (leftResult := 207073) (rightResult := 205744)
    (leftActual := SemanticResult207073.actual selector witness)
    (rightActual := SemanticResult205744.actual selector witness)
    (leftRaw := SemanticResult207073.rawTerms)
    (rightRaw := SemanticResult205744.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 207074) (rightBinding := 207075)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55992⟩) (rightExpression := ⟨58971⟩)
    (transferEvent := 207076) (summaryTransferEvent := 207077)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207073.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult205744.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult207078

namespace SemanticResult207083
def owner : Owner := ⟨.program ⟨257⟩, ⟨61952⟩⟩
def rawTerms : List Term := Proof.Events808.exact207083RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 207083
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207083.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult207078.owner)
    (rightOwner := SemanticResult205532.owner)
    (leftResult := 207078) (rightResult := 205532)
    (leftActual := SemanticResult207078.actual selector witness)
    (rightActual := SemanticResult205532.actual selector witness)
    (leftRaw := SemanticResult207078.rawTerms)
    (rightRaw := SemanticResult205532.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 207079) (rightBinding := 207080)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58972⟩) (rightExpression := ⟨61951⟩)
    (transferEvent := 207081) (summaryTransferEvent := 207082)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207078.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult205532.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult207083

namespace SemanticResult207088
def owner : Owner := ⟨.program ⟨257⟩, ⟨64932⟩⟩
def rawTerms : List Term := Proof.Events808.exact207088RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 207088
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207088.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult207083.owner)
    (rightOwner := SemanticResult205320.owner)
    (leftResult := 207083) (rightResult := 205320)
    (leftActual := SemanticResult207083.actual selector witness)
    (rightActual := SemanticResult205320.actual selector witness)
    (leftRaw := SemanticResult207083.rawTerms)
    (rightRaw := SemanticResult205320.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 207084) (rightBinding := 207085)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61952⟩) (rightExpression := ⟨64931⟩)
    (transferEvent := 207086) (summaryTransferEvent := 207087)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207083.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult205320.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult207088

namespace SemanticResult207093
def owner : Owner := ⟨.program ⟨257⟩, ⟨70325⟩⟩
def rawTerms : List Term := Proof.Events808.exact207093RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 207093
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207093.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult207088.owner)
    (rightOwner := SemanticResult205108.owner)
    (leftResult := 207088) (rightResult := 205108)
    (leftActual := SemanticResult207088.actual selector witness)
    (rightActual := SemanticResult205108.actual selector witness)
    (leftRaw := SemanticResult207088.rawTerms)
    (rightRaw := SemanticResult205108.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 207089) (rightBinding := 207090)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64932⟩) (rightExpression := ⟨70324⟩)
    (transferEvent := 207091) (summaryTransferEvent := 207092)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207088.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult205108.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult207093

namespace SemanticResult207098
def owner : Owner := ⟨.program ⟨257⟩, ⟨70326⟩⟩
def rawTerms : List Term := Proof.Events808.exact207098RawTerms
def summary : Bound := (.finite 3802007596962448506045899439491360353157172)
def resultEvent : Nat := 207098
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207098.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult207093.owner)
    (rightOwner := SemanticResult204896.owner)
    (leftResult := 207093) (rightResult := 204896)
    (leftActual := SemanticResult207093.actual selector witness)
    (rightActual := SemanticResult204896.actual selector witness)
    (leftRaw := SemanticResult207093.rawTerms)
    (rightRaw := SemanticResult204896.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3456353380086899479155517117627148481331252)
    (rightMaximum := 345654216875549026890382321864211871825920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 207094) (rightBinding := 207095)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70325⟩) (rightExpression := ⟨28337⟩)
    (transferEvent := 207096) (summaryTransferEvent := 207097)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207093.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult204896.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult207098

namespace SemanticResult207103
def owner : Owner := ⟨.program ⟨257⟩, ⟨70327⟩⟩
def rawTerms : List Term := Proof.Events808.exact207103RawTerms
def summary : Bound := (.finite 4147668141949793872257454032897973461975092)
def resultEvent : Nat := 207103
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207103.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult207098.owner)
    (rightOwner := SemanticResult204684.owner)
    (leftResult := 207098) (rightResult := 204684)
    (leftActual := SemanticResult207098.actual selector witness)
    (rightActual := SemanticResult204684.actual selector witness)
    (leftRaw := SemanticResult207098.rawTerms)
    (rightRaw := SemanticResult204684.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3802007596962448506045899439491360353157172)
    (rightMaximum := 345660544987345366211554593406613108817920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 207099) (rightBinding := 207100)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70326⟩) (rightExpression := ⟨31017⟩)
    (transferEvent := 207101) (summaryTransferEvent := 207102)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207098.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult204684.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult207103

namespace SemanticResult207108
def owner : Owner := ⟨.program ⟨257⟩, ⟨70328⟩⟩
def rawTerms : List Term := Proof.Events809.exact207108RawTerms
def summary : Bound := (.finite 4493332905678336798016456807332854062121012)
def resultEvent : Nat := 207108
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207108.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult207103.owner)
    (rightOwner := SemanticResult204472.owner)
    (leftResult := 207103) (rightResult := 204472)
    (leftActual := SemanticResult207103.actual selector witness)
    (rightActual := SemanticResult204472.actual selector witness)
    (leftRaw := SemanticResult207103.rawTerms)
    (rightRaw := SemanticResult204472.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4147668141949793872257454032897973461975092)
    (rightMaximum := 345664763728542925759002774434880600145920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 207104) (rightBinding := 207105)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70327⟩) (rightExpression := ⟨36677⟩)
    (transferEvent := 207106) (summaryTransferEvent := 207107)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207103.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult204472.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult207108

namespace SemanticResult207113
def owner : Owner := ⟨.program ⟨257⟩, ⟨70329⟩⟩
def rawTerms : List Term := Proof.Events809.exact207113RawTerms
def summary : Bound := (.finite 4838999778777478503549183672281868407930932)
def resultEvent : Nat := 207113
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult207113.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult207108.owner)
    (rightOwner := SemanticResult204260.owner)
    (leftResult := 207108) (rightResult := 204260)
    (leftActual := SemanticResult207108.actual selector witness)
    (rightActual := SemanticResult204260.actual selector witness)
    (leftRaw := SemanticResult207108.rawTerms)
    (rightRaw := SemanticResult204260.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4493332905678336798016456807332854062121012)
    (rightMaximum := 345666873099141705532726864949014345809920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 207109) (rightBinding := 207110)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70328⟩) (rightExpression := ⟨39357⟩)
    (transferEvent := 207111) (summaryTransferEvent := 207112)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207108.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult204260.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult207113

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
