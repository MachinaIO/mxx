import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1757
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1731
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1733
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1734
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1735
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1737
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1738
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1739
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1741
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1742
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1744
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1745
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1746
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1748
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1749
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1750
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1752
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1756

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult250938
def owner : Owner := ⟨.program ⟨257⟩, ⟨33828⟩⟩
def rawTerms : List Term := Proof.Events980.exact250938RawTerms
def summary : Bound := (.finite 1382506125545760169441014535464825839943732)
def resultEvent : Nat := 250938
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult250938.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult250933.owner)
    (rightOwner := SemanticResult250255.owner)
    (leftResult := 250933) (rightResult := 250255)
    (leftActual := SemanticResult250933.actual selector witness)
    (rightActual := SemanticResult250255.actual selector witness)
    (leftRaw := SemanticResult250933.rawTerms)
    (rightRaw := SemanticResult250255.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1036877221117396499835321299770218916085812)
    (rightMaximum := 345628904428363669605693235694606923857920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 250934) (rightBinding := 250935)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23808⟩) (rightExpression := ⟨33827⟩)
    (transferEvent := 250936) (summaryTransferEvent := 250937)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult250933.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult250255.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult250938

namespace SemanticResult250943
def owner : Owner := ⟨.program ⟨257⟩, ⟨52888⟩⟩
def rawTerms : List Term := Proof.Events980.exact250943RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 250943
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult250943.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult250938.owner)
    (rightOwner := SemanticResult250043.owner)
    (leftResult := 250938) (rightResult := 250043)
    (leftActual := SemanticResult250938.actual selector witness)
    (rightActual := SemanticResult250043.actual selector witness)
    (leftRaw := SemanticResult250938.rawTerms)
    (rightRaw := SemanticResult250043.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 250939) (rightBinding := 250940)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33828⟩) (rightExpression := ⟨52887⟩)
    (transferEvent := 250941) (summaryTransferEvent := 250942)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult250938.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult250043.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult250943

namespace SemanticResult250948
def owner : Owner := ⟨.program ⟨257⟩, ⟨55868⟩⟩
def rawTerms : List Term := Proof.Events980.exact250948RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 250948
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult250948.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult250943.owner)
    (rightOwner := SemanticResult249831.owner)
    (leftResult := 250943) (rightResult := 249831)
    (leftActual := SemanticResult250943.actual selector witness)
    (rightActual := SemanticResult249831.actual selector witness)
    (leftRaw := SemanticResult250943.rawTerms)
    (rightRaw := SemanticResult249831.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 250944) (rightBinding := 250945)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨52888⟩) (rightExpression := ⟨55867⟩)
    (transferEvent := 250946) (summaryTransferEvent := 250947)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult250943.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult249831.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult250948

namespace SemanticResult250953
def owner : Owner := ⟨.program ⟨257⟩, ⟨58848⟩⟩
def rawTerms : List Term := Proof.Events980.exact250953RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 250953
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult250953.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult250948.owner)
    (rightOwner := SemanticResult249619.owner)
    (leftResult := 250948) (rightResult := 249619)
    (leftActual := SemanticResult250948.actual selector witness)
    (rightActual := SemanticResult249619.actual selector witness)
    (leftRaw := SemanticResult250948.rawTerms)
    (rightRaw := SemanticResult249619.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 250949) (rightBinding := 250950)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55868⟩) (rightExpression := ⟨58847⟩)
    (transferEvent := 250951) (summaryTransferEvent := 250952)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult250948.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult249619.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult250953

namespace SemanticResult250958
def owner : Owner := ⟨.program ⟨257⟩, ⟨61828⟩⟩
def rawTerms : List Term := Proof.Events980.exact250958RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 250958
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult250958.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult250953.owner)
    (rightOwner := SemanticResult249407.owner)
    (leftResult := 250953) (rightResult := 249407)
    (leftActual := SemanticResult250953.actual selector witness)
    (rightActual := SemanticResult249407.actual selector witness)
    (leftRaw := SemanticResult250953.rawTerms)
    (rightRaw := SemanticResult249407.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 250954) (rightBinding := 250955)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58848⟩) (rightExpression := ⟨61827⟩)
    (transferEvent := 250956) (summaryTransferEvent := 250957)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult250953.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult249407.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult250958

namespace SemanticResult250963
def owner : Owner := ⟨.program ⟨257⟩, ⟨64808⟩⟩
def rawTerms : List Term := Proof.Events980.exact250963RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 250963
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult250963.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult250958.owner)
    (rightOwner := SemanticResult249195.owner)
    (leftResult := 250958) (rightResult := 249195)
    (leftActual := SemanticResult250958.actual selector witness)
    (rightActual := SemanticResult249195.actual selector witness)
    (leftRaw := SemanticResult250958.rawTerms)
    (rightRaw := SemanticResult249195.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 250959) (rightBinding := 250960)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61828⟩) (rightExpression := ⟨64807⟩)
    (transferEvent := 250961) (summaryTransferEvent := 250962)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult250958.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult249195.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult250963

namespace SemanticResult250968
def owner : Owner := ⟨.program ⟨257⟩, ⟨70009⟩⟩
def rawTerms : List Term := Proof.Events980.exact250968RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 250968
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult250968.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult250963.owner)
    (rightOwner := SemanticResult248983.owner)
    (leftResult := 250963) (rightResult := 248983)
    (leftActual := SemanticResult250963.actual selector witness)
    (rightActual := SemanticResult248983.actual selector witness)
    (leftRaw := SemanticResult250963.rawTerms)
    (rightRaw := SemanticResult248983.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 250964) (rightBinding := 250965)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64808⟩) (rightExpression := ⟨70008⟩)
    (transferEvent := 250966) (summaryTransferEvent := 250967)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult250963.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult248983.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult250968

namespace SemanticResult250973
def owner : Owner := ⟨.program ⟨257⟩, ⟨70010⟩⟩
def rawTerms : List Term := Proof.Events980.exact250973RawTerms
def summary : Bound := (.finite 3802007596962448506045899439491360353157172)
def resultEvent : Nat := 250973
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult250973.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult250968.owner)
    (rightOwner := SemanticResult248771.owner)
    (leftResult := 250968) (rightResult := 248771)
    (leftActual := SemanticResult250968.actual selector witness)
    (rightActual := SemanticResult248771.actual selector witness)
    (leftRaw := SemanticResult250968.rawTerms)
    (rightRaw := SemanticResult248771.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3456353380086899479155517117627148481331252)
    (rightMaximum := 345654216875549026890382321864211871825920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 250969) (rightBinding := 250970)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70009⟩) (rightExpression := ⟨28237⟩)
    (transferEvent := 250971) (summaryTransferEvent := 250972)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult250968.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult248771.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult250973

namespace SemanticResult250978
def owner : Owner := ⟨.program ⟨257⟩, ⟨70011⟩⟩
def rawTerms : List Term := Proof.Events980.exact250978RawTerms
def summary : Bound := (.finite 4147668141949793872257454032897973461975092)
def resultEvent : Nat := 250978
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult250978.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult250973.owner)
    (rightOwner := SemanticResult248559.owner)
    (leftResult := 250973) (rightResult := 248559)
    (leftActual := SemanticResult250973.actual selector witness)
    (rightActual := SemanticResult248559.actual selector witness)
    (leftRaw := SemanticResult250973.rawTerms)
    (rightRaw := SemanticResult248559.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3802007596962448506045899439491360353157172)
    (rightMaximum := 345660544987345366211554593406613108817920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 250974) (rightBinding := 250975)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70010⟩) (rightExpression := ⟨30917⟩)
    (transferEvent := 250976) (summaryTransferEvent := 250977)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult250973.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult248559.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult250978

namespace SemanticResult250983
def owner : Owner := ⟨.program ⟨257⟩, ⟨70012⟩⟩
def rawTerms : List Term := Proof.Events980.exact250983RawTerms
def summary : Bound := (.finite 4493332905678336798016456807332854062121012)
def resultEvent : Nat := 250983
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult250983.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult250978.owner)
    (rightOwner := SemanticResult248347.owner)
    (leftResult := 250978) (rightResult := 248347)
    (leftActual := SemanticResult250978.actual selector witness)
    (rightActual := SemanticResult248347.actual selector witness)
    (leftRaw := SemanticResult250978.rawTerms)
    (rightRaw := SemanticResult248347.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4147668141949793872257454032897973461975092)
    (rightMaximum := 345664763728542925759002774434880600145920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 250979) (rightBinding := 250980)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70011⟩) (rightExpression := ⟨36577⟩)
    (transferEvent := 250981) (summaryTransferEvent := 250982)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult250978.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult248347.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult250983

namespace SemanticResult250988
def owner : Owner := ⟨.program ⟨257⟩, ⟨70013⟩⟩
def rawTerms : List Term := Proof.Events980.exact250988RawTerms
def summary : Bound := (.finite 4838999778777478503549183672281868407930932)
def resultEvent : Nat := 250988
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult250988.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult250983.owner)
    (rightOwner := SemanticResult248135.owner)
    (leftResult := 250983) (rightResult := 248135)
    (leftActual := SemanticResult250983.actual selector witness)
    (rightActual := SemanticResult248135.actual selector witness)
    (leftRaw := SemanticResult250983.rawTerms)
    (rightRaw := SemanticResult248135.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4493332905678336798016456807332854062121012)
    (rightMaximum := 345666873099141705532726864949014345809920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 250984) (rightBinding := 250985)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70012⟩) (rightExpression := ⟨39257⟩)
    (transferEvent := 250986) (summaryTransferEvent := 250987)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult250983.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult248135.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult250988

namespace SemanticResult250993
def owner : Owner := ⟨.program ⟨257⟩, ⟨70014⟩⟩
def rawTerms : List Term := Proof.Events980.exact250993RawTerms
def summary : Bound := (.finite 5184670870617817768629358718259150245068852)
def resultEvent : Nat := 250993
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult250993.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult250988.owner)
    (rightOwner := SemanticResult247923.owner)
    (leftResult := 250988) (rightResult := 247923)
    (leftActual := SemanticResult250988.actual selector witness)
    (rightActual := SemanticResult247923.actual selector witness)
    (leftRaw := SemanticResult250988.rawTerms)
    (rightRaw := SemanticResult247923.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4838999778777478503549183672281868407930932)
    (rightMaximum := 345671091840339265080175045977281837137920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 250989) (rightBinding := 250990)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70013⟩) (rightExpression := ⟨41937⟩)
    (transferEvent := 250991) (summaryTransferEvent := 250992)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult250988.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult247923.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult250993

namespace SemanticResult250998
def owner : Owner := ⟨.program ⟨257⟩, ⟨70015⟩⟩
def rawTerms : List Term := Proof.Events980.exact250998RawTerms
def summary : Bound := (.finite 5530348290569953373030706035778833319198772)
def resultEvent : Nat := 250998
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult250998.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult250993.owner)
    (rightOwner := SemanticResult247711.owner)
    (leftResult := 250993) (rightResult := 247711)
    (leftActual := SemanticResult250993.actual selector witness)
    (rightActual := SemanticResult247711.actual selector witness)
    (leftRaw := SemanticResult250993.rawTerms)
    (rightRaw := SemanticResult247711.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5184670870617817768629358718259150245068852)
    (rightMaximum := 345677419952135604401347317519683074129920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 250994) (rightBinding := 250995)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70014⟩) (rightExpression := ⟨44617⟩)
    (transferEvent := 250996) (summaryTransferEvent := 250997)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult250993.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult247711.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult250998

namespace SemanticResult251003
def owner : Owner := ⟨.program ⟨257⟩, ⟨70016⟩⟩
def rawTerms : List Term := Proof.Events980.exact251003RawTerms
def summary : Bound := (.finite 5876032038633885316753225624840917630320692)
def resultEvent : Nat := 251003
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult251003.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult250998.owner)
    (rightOwner := SemanticResult247499.owner)
    (leftResult := 250998) (rightResult := 247499)
    (leftActual := SemanticResult250998.actual selector witness)
    (rightActual := SemanticResult247499.actual selector witness)
    (leftRaw := SemanticResult250998.rawTerms)
    (rightRaw := SemanticResult247499.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5530348290569953373030706035778833319198772)
    (rightMaximum := 345683748063931943722519589062084311121920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 250999) (rightBinding := 251000)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70015⟩) (rightExpression := ⟨47297⟩)
    (transferEvent := 251001) (summaryTransferEvent := 251002)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult250998.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult247499.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult251003

namespace SemanticResult251008
def owner : Owner := ⟨.program ⟨257⟩, ⟨70017⟩⟩
def rawTerms : List Term := Proof.Events980.exact251008RawTerms
def summary : Bound := (.finite 6221717896068416040249469304417135687106612)
def resultEvent : Nat := 251008
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult251008.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult251003.owner)
    (rightOwner := SemanticResult247287.owner)
    (leftResult := 251003) (rightResult := 247287)
    (leftActual := SemanticResult251003.actual selector witness)
    (rightActual := SemanticResult247287.actual selector witness)
    (leftRaw := SemanticResult251003.rawTerms)
    (rightRaw := SemanticResult247287.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5876032038633885316753225624840917630320692)
    (rightMaximum := 345685857434530723496243679576218056785920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 251004) (rightBinding := 251005)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70016⟩) (rightExpression := ⟨49977⟩)
    (transferEvent := 251006) (summaryTransferEvent := 251007)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult251003.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult247287.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult251008

namespace SemanticResult251013
def owner : Owner := ⟨.program ⟨257⟩, ⟨71178⟩⟩
def rawTerms : List Term := Proof.Events980.exact251013RawTerms
def summary : Bound := (.finite 66805187227601152574551644069558752530002096506798132)
def resultEvent : Nat := 251013
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult251013.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult251008.owner)
    (rightOwner := SemanticResult247075.owner)
    (leftResult := 251008) (rightResult := 247075)
    (leftActual := SemanticResult251008.actual selector witness)
    (rightActual := SemanticResult247075.actual selector witness)
    (leftRaw := SemanticResult251008.rawTerms)
    (rightRaw := SemanticResult247075.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 6221717896068416040249469304417135687106612)
    (rightMaximum := 66805187221379434678483228029309283225584960819691520) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 251009) (rightBinding := 251010)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70017⟩) (rightExpression := ⟨71176⟩)
    (transferEvent := 251011) (summaryTransferEvent := 251012)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult251008.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult247075.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult251013

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
