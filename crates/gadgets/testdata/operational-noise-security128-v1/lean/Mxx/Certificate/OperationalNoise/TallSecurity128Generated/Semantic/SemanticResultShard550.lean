import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard550
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard530
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard531
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard533
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard534
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard535
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard537
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard538
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard540
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard541
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard542
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard544
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard545
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard546
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard548
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard549

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult75418
def owner : Owner := ⟨.program ⟨257⟩, ⟨10799⟩⟩
def rawTerms : List Term := Proof.Events294.exact75418RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 75418
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75418.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge75415.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75411.owner)
    (rightOwner := SemanticResult75411.owner)
    (leftResult := 75411) (rightResult := 75411)
    (leftActual := SemanticResult75411.actual selector witness)
    (rightActual := SemanticResult75411.actual selector witness)
    (leftRaw := SemanticResult75411.rawTerms)
    (rightRaw := SemanticResult75411.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75412) (rightBinding := 75413)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10798⟩) (rightExpression := ⟨10798⟩)
    (coefficientTransfer := 75414) (summaryTransfer := 75417)
    (base := LeftOperatorMerge75415.base)
    (reconstruction := LeftOperatorMerge75415.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75411.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75411.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge75415.operationAgreement
  · rfl
  · decide
end SemanticResult75418

namespace SemanticResult75423
def owner : Owner := ⟨.program ⟨257⟩, ⟨17955⟩⟩
def rawTerms : List Term := Proof.Events294.exact75423RawTerms
def summary : Bound := (.finite 345624685687166110058245054666339432529972)
def resultEvent : Nat := 75423
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75423.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75418.owner)
    (rightOwner := SemanticResult75391.owner)
    (leftResult := 75418) (rightResult := 75391)
    (leftActual := SemanticResult75418.actual selector witness)
    (rightActual := SemanticResult75391.actual selector witness)
    (leftRaw := SemanticResult75418.rawTerms)
    (rightRaw := SemanticResult75391.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 345624685687166110058245054666339432529920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75419) (rightBinding := 75420)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10799⟩) (rightExpression := ⟨17954⟩)
    (transferEvent := 75421) (summaryTransferEvent := 75422)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75418.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75391.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75423

namespace SemanticResult75428
def owner : Owner := ⟨.program ⟨257⟩, ⟨20867⟩⟩
def rawTerms : List Term := Proof.Events294.exact75428RawTerms
def summary : Bound := (.finite 691250426059631610003352154589745737891892)
def resultEvent : Nat := 75428
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75428.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75423.owner)
    (rightOwner := SemanticResult75179.owner)
    (leftResult := 75423) (rightResult := 75179)
    (leftActual := SemanticResult75423.actual selector witness)
    (rightActual := SemanticResult75179.actual selector witness)
    (leftRaw := SemanticResult75423.rawTerms)
    (rightRaw := SemanticResult75179.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 345624685687166110058245054666339432529972)
    (rightMaximum := 345625740372465499945107099923406305361920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75424) (rightBinding := 75425)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17955⟩) (rightExpression := ⟨20866⟩)
    (transferEvent := 75426) (summaryTransferEvent := 75427)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75423.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75179.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75428

namespace SemanticResult75433
def owner : Owner := ⟨.program ⟨257⟩, ⟨24087⟩⟩
def rawTerms : List Term := Proof.Events294.exact75433RawTerms
def summary : Bound := (.finite 1036877221117396499835321299770218916085812)
def resultEvent : Nat := 75433
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75433.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75428.owner)
    (rightOwner := SemanticResult74967.owner)
    (leftResult := 75428) (rightResult := 74967)
    (leftActual := SemanticResult75428.actual selector witness)
    (rightActual := SemanticResult74967.actual selector witness)
    (leftRaw := SemanticResult75428.rawTerms)
    (rightRaw := SemanticResult74967.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 691250426059631610003352154589745737891892)
    (rightMaximum := 345626795057764889831969145180473178193920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75429) (rightBinding := 75430)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20867⟩) (rightExpression := ⟨24086⟩)
    (transferEvent := 75431) (summaryTransferEvent := 75432)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75428.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult74967.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75433

namespace SemanticResult75438
def owner : Owner := ⟨.program ⟨257⟩, ⟨34107⟩⟩
def rawTerms : List Term := Proof.Events294.exact75438RawTerms
def summary : Bound := (.finite 1382506125545760169441014535464825839943732)
def resultEvent : Nat := 75438
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75438.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75433.owner)
    (rightOwner := SemanticResult74755.owner)
    (leftResult := 75433) (rightResult := 74755)
    (leftActual := SemanticResult75433.actual selector witness)
    (rightActual := SemanticResult74755.actual selector witness)
    (leftRaw := SemanticResult75433.rawTerms)
    (rightRaw := SemanticResult74755.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1036877221117396499835321299770218916085812)
    (rightMaximum := 345628904428363669605693235694606923857920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75434) (rightBinding := 75435)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨24087⟩) (rightExpression := ⟨34106⟩)
    (transferEvent := 75436) (summaryTransferEvent := 75437)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75433.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult74755.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75438

namespace SemanticResult75443
def owner : Owner := ⟨.program ⟨257⟩, ⟨53167⟩⟩
def rawTerms : List Term := Proof.Events294.exact75443RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 75443
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75443.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75438.owner)
    (rightOwner := SemanticResult74543.owner)
    (leftResult := 75438) (rightResult := 74543)
    (leftActual := SemanticResult75438.actual selector witness)
    (rightActual := SemanticResult74543.actual selector witness)
    (leftRaw := SemanticResult75438.rawTerms)
    (rightRaw := SemanticResult74543.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75439) (rightBinding := 75440)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨34107⟩) (rightExpression := ⟨53166⟩)
    (transferEvent := 75441) (summaryTransferEvent := 75442)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75438.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult74543.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75443

namespace SemanticResult75448
def owner : Owner := ⟨.program ⟨257⟩, ⟨56147⟩⟩
def rawTerms : List Term := Proof.Events294.exact75448RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 75448
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75448.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75443.owner)
    (rightOwner := SemanticResult74331.owner)
    (leftResult := 75443) (rightResult := 74331)
    (leftActual := SemanticResult75443.actual selector witness)
    (rightActual := SemanticResult74331.actual selector witness)
    (leftRaw := SemanticResult75443.rawTerms)
    (rightRaw := SemanticResult74331.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75444) (rightBinding := 75445)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53167⟩) (rightExpression := ⟨56146⟩)
    (transferEvent := 75446) (summaryTransferEvent := 75447)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75443.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult74331.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75448

namespace SemanticResult75453
def owner : Owner := ⟨.program ⟨257⟩, ⟨59127⟩⟩
def rawTerms : List Term := Proof.Events294.exact75453RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 75453
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75453.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75448.owner)
    (rightOwner := SemanticResult74119.owner)
    (leftResult := 75448) (rightResult := 74119)
    (leftActual := SemanticResult75448.actual selector witness)
    (rightActual := SemanticResult74119.actual selector witness)
    (leftRaw := SemanticResult75448.rawTerms)
    (rightRaw := SemanticResult74119.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75449) (rightBinding := 75450)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56147⟩) (rightExpression := ⟨59126⟩)
    (transferEvent := 75451) (summaryTransferEvent := 75452)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75448.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult74119.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75453

namespace SemanticResult75458
def owner : Owner := ⟨.program ⟨257⟩, ⟨62107⟩⟩
def rawTerms : List Term := Proof.Events294.exact75458RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 75458
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75458.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75453.owner)
    (rightOwner := SemanticResult73907.owner)
    (leftResult := 75453) (rightResult := 73907)
    (leftActual := SemanticResult75453.actual selector witness)
    (rightActual := SemanticResult73907.actual selector witness)
    (leftRaw := SemanticResult75453.rawTerms)
    (rightRaw := SemanticResult73907.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75454) (rightBinding := 75455)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59127⟩) (rightExpression := ⟨62106⟩)
    (transferEvent := 75456) (summaryTransferEvent := 75457)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75453.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73907.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75458

namespace SemanticResult75463
def owner : Owner := ⟨.program ⟨257⟩, ⟨65087⟩⟩
def rawTerms : List Term := Proof.Events294.exact75463RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 75463
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75463.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75458.owner)
    (rightOwner := SemanticResult73695.owner)
    (leftResult := 75458) (rightResult := 73695)
    (leftActual := SemanticResult75458.actual selector witness)
    (rightActual := SemanticResult73695.actual selector witness)
    (leftRaw := SemanticResult75458.rawTerms)
    (rightRaw := SemanticResult73695.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75459) (rightBinding := 75460)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62107⟩) (rightExpression := ⟨65086⟩)
    (transferEvent := 75461) (summaryTransferEvent := 75462)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75458.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73695.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75463

namespace SemanticResult75468
def owner : Owner := ⟨.program ⟨257⟩, ⟨70720⟩⟩
def rawTerms : List Term := Proof.Events294.exact75468RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 75468
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75468.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75463.owner)
    (rightOwner := SemanticResult73483.owner)
    (leftResult := 75463) (rightResult := 73483)
    (leftActual := SemanticResult75463.actual selector witness)
    (rightActual := SemanticResult73483.actual selector witness)
    (leftRaw := SemanticResult75463.rawTerms)
    (rightRaw := SemanticResult73483.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75464) (rightBinding := 75465)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65087⟩) (rightExpression := ⟨70719⟩)
    (transferEvent := 75466) (summaryTransferEvent := 75467)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75463.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73483.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75468

namespace SemanticResult75473
def owner : Owner := ⟨.program ⟨257⟩, ⟨70721⟩⟩
def rawTerms : List Term := Proof.Events294.exact75473RawTerms
def summary : Bound := (.finite 3802007596962448506045899439491360353157172)
def resultEvent : Nat := 75473
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75473.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75468.owner)
    (rightOwner := SemanticResult73271.owner)
    (leftResult := 75468) (rightResult := 73271)
    (leftActual := SemanticResult75468.actual selector witness)
    (rightActual := SemanticResult73271.actual selector witness)
    (leftRaw := SemanticResult75468.rawTerms)
    (rightRaw := SemanticResult73271.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3456353380086899479155517117627148481331252)
    (rightMaximum := 345654216875549026890382321864211871825920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75469) (rightBinding := 75470)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70720⟩) (rightExpression := ⟨28462⟩)
    (transferEvent := 75471) (summaryTransferEvent := 75472)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75468.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73271.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75473

namespace SemanticResult75478
def owner : Owner := ⟨.program ⟨257⟩, ⟨70722⟩⟩
def rawTerms : List Term := Proof.Events294.exact75478RawTerms
def summary : Bound := (.finite 4147668141949793872257454032897973461975092)
def resultEvent : Nat := 75478
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75478.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75473.owner)
    (rightOwner := SemanticResult73059.owner)
    (leftResult := 75473) (rightResult := 73059)
    (leftActual := SemanticResult75473.actual selector witness)
    (rightActual := SemanticResult73059.actual selector witness)
    (leftRaw := SemanticResult75473.rawTerms)
    (rightRaw := SemanticResult73059.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3802007596962448506045899439491360353157172)
    (rightMaximum := 345660544987345366211554593406613108817920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75474) (rightBinding := 75475)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70721⟩) (rightExpression := ⟨31142⟩)
    (transferEvent := 75476) (summaryTransferEvent := 75477)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75473.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73059.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75478

namespace SemanticResult75483
def owner : Owner := ⟨.program ⟨257⟩, ⟨70723⟩⟩
def rawTerms : List Term := Proof.Events294.exact75483RawTerms
def summary : Bound := (.finite 4493332905678336798016456807332854062121012)
def resultEvent : Nat := 75483
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75483.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75478.owner)
    (rightOwner := SemanticResult72847.owner)
    (leftResult := 75478) (rightResult := 72847)
    (leftActual := SemanticResult75478.actual selector witness)
    (rightActual := SemanticResult72847.actual selector witness)
    (leftRaw := SemanticResult75478.rawTerms)
    (rightRaw := SemanticResult72847.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4147668141949793872257454032897973461975092)
    (rightMaximum := 345664763728542925759002774434880600145920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75479) (rightBinding := 75480)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70722⟩) (rightExpression := ⟨36802⟩)
    (transferEvent := 75481) (summaryTransferEvent := 75482)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75478.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult72847.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75483

namespace SemanticResult75488
def owner : Owner := ⟨.program ⟨257⟩, ⟨70724⟩⟩
def rawTerms : List Term := Proof.Events294.exact75488RawTerms
def summary : Bound := (.finite 4838999778777478503549183672281868407930932)
def resultEvent : Nat := 75488
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75488.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75483.owner)
    (rightOwner := SemanticResult72635.owner)
    (leftResult := 75483) (rightResult := 72635)
    (leftActual := SemanticResult75483.actual selector witness)
    (rightActual := SemanticResult72635.actual selector witness)
    (leftRaw := SemanticResult75483.rawTerms)
    (rightRaw := SemanticResult72635.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4493332905678336798016456807332854062121012)
    (rightMaximum := 345666873099141705532726864949014345809920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75484) (rightBinding := 75485)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70723⟩) (rightExpression := ⟨39482⟩)
    (transferEvent := 75486) (summaryTransferEvent := 75487)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75483.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult72635.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75488

namespace SemanticResult75493
def owner : Owner := ⟨.program ⟨257⟩, ⟨70725⟩⟩
def rawTerms : List Term := Proof.Events294.exact75493RawTerms
def summary : Bound := (.finite 5184670870617817768629358718259150245068852)
def resultEvent : Nat := 75493
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult75493.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult75488.owner)
    (rightOwner := SemanticResult72423.owner)
    (leftResult := 75488) (rightResult := 72423)
    (leftActual := SemanticResult75488.actual selector witness)
    (rightActual := SemanticResult72423.actual selector witness)
    (leftRaw := SemanticResult75488.rawTerms)
    (rightRaw := SemanticResult72423.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4838999778777478503549183672281868407930932)
    (rightMaximum := 345671091840339265080175045977281837137920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 75489) (rightBinding := 75490)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70724⟩) (rightExpression := ⟨42162⟩)
    (transferEvent := 75491) (summaryTransferEvent := 75492)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75488.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult72423.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult75493

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
