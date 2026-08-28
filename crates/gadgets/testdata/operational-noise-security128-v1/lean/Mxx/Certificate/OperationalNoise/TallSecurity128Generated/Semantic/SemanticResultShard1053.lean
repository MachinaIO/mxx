import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1053
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1029
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1030
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1031
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1033
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1034
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1036
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1037
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1038
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1040
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1041
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1042
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1044
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1045
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1047
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1048
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1049
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1052

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult148558
def owner : Owner := ⟨.program ⟨257⟩, ⟨23653⟩⟩
def rawTerms : List Term := Proof.Events580.exact148558RawTerms
def summary : Bound := (.finite 1036877221117396499835321299770218916085812)
def resultEvent : Nat := 148558
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148558.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148553.owner)
    (rightOwner := SemanticResult148092.owner)
    (leftResult := 148553) (rightResult := 148092)
    (leftActual := SemanticResult148553.actual selector witness)
    (rightActual := SemanticResult148092.actual selector witness)
    (leftRaw := SemanticResult148553.rawTerms)
    (rightRaw := SemanticResult148092.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 691250426059631610003352154589745737891892)
    (rightMaximum := 345626795057764889831969145180473178193920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148554) (rightBinding := 148555)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20433⟩) (rightExpression := ⟨23652⟩)
    (transferEvent := 148556) (summaryTransferEvent := 148557)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148553.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult148092.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148558

namespace SemanticResult148563
def owner : Owner := ⟨.program ⟨257⟩, ⟨33673⟩⟩
def rawTerms : List Term := Proof.Events580.exact148563RawTerms
def summary : Bound := (.finite 1382506125545760169441014535464825839943732)
def resultEvent : Nat := 148563
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148563.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148558.owner)
    (rightOwner := SemanticResult147880.owner)
    (leftResult := 148558) (rightResult := 147880)
    (leftActual := SemanticResult148558.actual selector witness)
    (rightActual := SemanticResult147880.actual selector witness)
    (leftRaw := SemanticResult148558.rawTerms)
    (rightRaw := SemanticResult147880.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1036877221117396499835321299770218916085812)
    (rightMaximum := 345628904428363669605693235694606923857920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148559) (rightBinding := 148560)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23653⟩) (rightExpression := ⟨33672⟩)
    (transferEvent := 148561) (summaryTransferEvent := 148562)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148558.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult147880.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148563

namespace SemanticResult148568
def owner : Owner := ⟨.program ⟨257⟩, ⟨52733⟩⟩
def rawTerms : List Term := Proof.Events580.exact148568RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 148568
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148568.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148563.owner)
    (rightOwner := SemanticResult147668.owner)
    (leftResult := 148563) (rightResult := 147668)
    (leftActual := SemanticResult148563.actual selector witness)
    (rightActual := SemanticResult147668.actual selector witness)
    (leftRaw := SemanticResult148563.rawTerms)
    (rightRaw := SemanticResult147668.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148564) (rightBinding := 148565)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33673⟩) (rightExpression := ⟨52732⟩)
    (transferEvent := 148566) (summaryTransferEvent := 148567)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148563.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult147668.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148568

namespace SemanticResult148573
def owner : Owner := ⟨.program ⟨257⟩, ⟨55713⟩⟩
def rawTerms : List Term := Proof.Events580.exact148573RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 148573
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148573.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148568.owner)
    (rightOwner := SemanticResult147456.owner)
    (leftResult := 148568) (rightResult := 147456)
    (leftActual := SemanticResult148568.actual selector witness)
    (rightActual := SemanticResult147456.actual selector witness)
    (leftRaw := SemanticResult148568.rawTerms)
    (rightRaw := SemanticResult147456.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148569) (rightBinding := 148570)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨52733⟩) (rightExpression := ⟨55712⟩)
    (transferEvent := 148571) (summaryTransferEvent := 148572)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148568.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult147456.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148573

namespace SemanticResult148578
def owner : Owner := ⟨.program ⟨257⟩, ⟨58693⟩⟩
def rawTerms : List Term := Proof.Events580.exact148578RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 148578
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148578.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148573.owner)
    (rightOwner := SemanticResult147244.owner)
    (leftResult := 148573) (rightResult := 147244)
    (leftActual := SemanticResult148573.actual selector witness)
    (rightActual := SemanticResult147244.actual selector witness)
    (leftRaw := SemanticResult148573.rawTerms)
    (rightRaw := SemanticResult147244.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148574) (rightBinding := 148575)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55713⟩) (rightExpression := ⟨58692⟩)
    (transferEvent := 148576) (summaryTransferEvent := 148577)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148573.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult147244.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148578

namespace SemanticResult148583
def owner : Owner := ⟨.program ⟨257⟩, ⟨61673⟩⟩
def rawTerms : List Term := Proof.Events580.exact148583RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 148583
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148583.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148578.owner)
    (rightOwner := SemanticResult147032.owner)
    (leftResult := 148578) (rightResult := 147032)
    (leftActual := SemanticResult148578.actual selector witness)
    (rightActual := SemanticResult147032.actual selector witness)
    (leftRaw := SemanticResult148578.rawTerms)
    (rightRaw := SemanticResult147032.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148579) (rightBinding := 148580)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58693⟩) (rightExpression := ⟨61672⟩)
    (transferEvent := 148581) (summaryTransferEvent := 148582)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148578.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult147032.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148583

namespace SemanticResult148588
def owner : Owner := ⟨.program ⟨257⟩, ⟨64653⟩⟩
def rawTerms : List Term := Proof.Events580.exact148588RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 148588
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148588.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148583.owner)
    (rightOwner := SemanticResult146820.owner)
    (leftResult := 148583) (rightResult := 146820)
    (leftActual := SemanticResult148583.actual selector witness)
    (rightActual := SemanticResult146820.actual selector witness)
    (leftRaw := SemanticResult148583.rawTerms)
    (rightRaw := SemanticResult146820.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148584) (rightBinding := 148585)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61673⟩) (rightExpression := ⟨64652⟩)
    (transferEvent := 148586) (summaryTransferEvent := 148587)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148583.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult146820.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148588

namespace SemanticResult148593
def owner : Owner := ⟨.program ⟨257⟩, ⟨69614⟩⟩
def rawTerms : List Term := Proof.Events580.exact148593RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 148593
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148593.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148588.owner)
    (rightOwner := SemanticResult146608.owner)
    (leftResult := 148588) (rightResult := 146608)
    (leftActual := SemanticResult148588.actual selector witness)
    (rightActual := SemanticResult146608.actual selector witness)
    (leftRaw := SemanticResult148588.rawTerms)
    (rightRaw := SemanticResult146608.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148589) (rightBinding := 148590)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64653⟩) (rightExpression := ⟨69613⟩)
    (transferEvent := 148591) (summaryTransferEvent := 148592)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148588.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult146608.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148593

namespace SemanticResult148598
def owner : Owner := ⟨.program ⟨257⟩, ⟨69615⟩⟩
def rawTerms : List Term := Proof.Events580.exact148598RawTerms
def summary : Bound := (.finite 3802007596962448506045899439491360353157172)
def resultEvent : Nat := 148598
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148598.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148593.owner)
    (rightOwner := SemanticResult146396.owner)
    (leftResult := 148593) (rightResult := 146396)
    (leftActual := SemanticResult148593.actual selector witness)
    (rightActual := SemanticResult146396.actual selector witness)
    (leftRaw := SemanticResult148593.rawTerms)
    (rightRaw := SemanticResult146396.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3456353380086899479155517117627148481331252)
    (rightMaximum := 345654216875549026890382321864211871825920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148594) (rightBinding := 148595)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69614⟩) (rightExpression := ⟨28112⟩)
    (transferEvent := 148596) (summaryTransferEvent := 148597)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148593.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult146396.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148598

namespace SemanticResult148603
def owner : Owner := ⟨.program ⟨257⟩, ⟨69616⟩⟩
def rawTerms : List Term := Proof.Events580.exact148603RawTerms
def summary : Bound := (.finite 4147668141949793872257454032897973461975092)
def resultEvent : Nat := 148603
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148603.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148598.owner)
    (rightOwner := SemanticResult146184.owner)
    (leftResult := 148598) (rightResult := 146184)
    (leftActual := SemanticResult148598.actual selector witness)
    (rightActual := SemanticResult146184.actual selector witness)
    (leftRaw := SemanticResult148598.rawTerms)
    (rightRaw := SemanticResult146184.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3802007596962448506045899439491360353157172)
    (rightMaximum := 345660544987345366211554593406613108817920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148599) (rightBinding := 148600)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69615⟩) (rightExpression := ⟨30792⟩)
    (transferEvent := 148601) (summaryTransferEvent := 148602)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148598.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult146184.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148603

namespace SemanticResult148608
def owner : Owner := ⟨.program ⟨257⟩, ⟨69617⟩⟩
def rawTerms : List Term := Proof.Events580.exact148608RawTerms
def summary : Bound := (.finite 4493332905678336798016456807332854062121012)
def resultEvent : Nat := 148608
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148608.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148603.owner)
    (rightOwner := SemanticResult145972.owner)
    (leftResult := 148603) (rightResult := 145972)
    (leftActual := SemanticResult148603.actual selector witness)
    (rightActual := SemanticResult145972.actual selector witness)
    (leftRaw := SemanticResult148603.rawTerms)
    (rightRaw := SemanticResult145972.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4147668141949793872257454032897973461975092)
    (rightMaximum := 345664763728542925759002774434880600145920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148604) (rightBinding := 148605)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69616⟩) (rightExpression := ⟨36452⟩)
    (transferEvent := 148606) (summaryTransferEvent := 148607)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148603.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult145972.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148608

namespace SemanticResult148613
def owner : Owner := ⟨.program ⟨257⟩, ⟨69618⟩⟩
def rawTerms : List Term := Proof.Events580.exact148613RawTerms
def summary : Bound := (.finite 4838999778777478503549183672281868407930932)
def resultEvent : Nat := 148613
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148613.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148608.owner)
    (rightOwner := SemanticResult145760.owner)
    (leftResult := 148608) (rightResult := 145760)
    (leftActual := SemanticResult148608.actual selector witness)
    (rightActual := SemanticResult145760.actual selector witness)
    (leftRaw := SemanticResult148608.rawTerms)
    (rightRaw := SemanticResult145760.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4493332905678336798016456807332854062121012)
    (rightMaximum := 345666873099141705532726864949014345809920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148609) (rightBinding := 148610)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69617⟩) (rightExpression := ⟨39132⟩)
    (transferEvent := 148611) (summaryTransferEvent := 148612)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148608.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult145760.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148613

namespace SemanticResult148618
def owner : Owner := ⟨.program ⟨257⟩, ⟨69619⟩⟩
def rawTerms : List Term := Proof.Events580.exact148618RawTerms
def summary : Bound := (.finite 5184670870617817768629358718259150245068852)
def resultEvent : Nat := 148618
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148618.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148613.owner)
    (rightOwner := SemanticResult145548.owner)
    (leftResult := 148613) (rightResult := 145548)
    (leftActual := SemanticResult148613.actual selector witness)
    (rightActual := SemanticResult145548.actual selector witness)
    (leftRaw := SemanticResult148613.rawTerms)
    (rightRaw := SemanticResult145548.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4838999778777478503549183672281868407930932)
    (rightMaximum := 345671091840339265080175045977281837137920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148614) (rightBinding := 148615)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69618⟩) (rightExpression := ⟨41812⟩)
    (transferEvent := 148616) (summaryTransferEvent := 148617)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148613.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult145548.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148618

namespace SemanticResult148623
def owner : Owner := ⟨.program ⟨257⟩, ⟨69620⟩⟩
def rawTerms : List Term := Proof.Events580.exact148623RawTerms
def summary : Bound := (.finite 5530348290569953373030706035778833319198772)
def resultEvent : Nat := 148623
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148623.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148618.owner)
    (rightOwner := SemanticResult145336.owner)
    (leftResult := 148618) (rightResult := 145336)
    (leftActual := SemanticResult148618.actual selector witness)
    (rightActual := SemanticResult145336.actual selector witness)
    (leftRaw := SemanticResult148618.rawTerms)
    (rightRaw := SemanticResult145336.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5184670870617817768629358718259150245068852)
    (rightMaximum := 345677419952135604401347317519683074129920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148619) (rightBinding := 148620)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69619⟩) (rightExpression := ⟨44492⟩)
    (transferEvent := 148621) (summaryTransferEvent := 148622)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148618.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult145336.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148623

namespace SemanticResult148628
def owner : Owner := ⟨.program ⟨257⟩, ⟨69621⟩⟩
def rawTerms : List Term := Proof.Events580.exact148628RawTerms
def summary : Bound := (.finite 5876032038633885316753225624840917630320692)
def resultEvent : Nat := 148628
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148628.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148623.owner)
    (rightOwner := SemanticResult145124.owner)
    (leftResult := 148623) (rightResult := 145124)
    (leftActual := SemanticResult148623.actual selector witness)
    (rightActual := SemanticResult145124.actual selector witness)
    (leftRaw := SemanticResult148623.rawTerms)
    (rightRaw := SemanticResult145124.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5530348290569953373030706035778833319198772)
    (rightMaximum := 345683748063931943722519589062084311121920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148624) (rightBinding := 148625)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69620⟩) (rightExpression := ⟨47172⟩)
    (transferEvent := 148626) (summaryTransferEvent := 148627)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148623.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult145124.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148628

namespace SemanticResult148633
def owner : Owner := ⟨.program ⟨257⟩, ⟨69622⟩⟩
def rawTerms : List Term := Proof.Events580.exact148633RawTerms
def summary : Bound := (.finite 6221717896068416040249469304417135687106612)
def resultEvent : Nat := 148633
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult148633.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult148628.owner)
    (rightOwner := SemanticResult144912.owner)
    (leftResult := 148628) (rightResult := 144912)
    (leftActual := SemanticResult148628.actual selector witness)
    (rightActual := SemanticResult144912.actual selector witness)
    (leftRaw := SemanticResult148628.rawTerms)
    (rightRaw := SemanticResult144912.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5876032038633885316753225624840917630320692)
    (rightMaximum := 345685857434530723496243679576218056785920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 148629) (rightBinding := 148630)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69621⟩) (rightExpression := ⟨49852⟩)
    (transferEvent := 148631) (summaryTransferEvent := 148632)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult148628.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult144912.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult148633

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
