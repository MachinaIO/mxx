import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1524
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1468
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1472
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1476
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1479
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1483
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1487
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1490
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1494
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1498
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1501
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1505
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1509
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1512
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1516
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1520
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1522
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1523

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult216195
def owner : Owner := ⟨.program ⟨257⟩, ⟨17764⟩⟩
def rawTerms : List Term := Proof.Events844.exact216195RawTerms
def summary : Bound := (.finite 32188807212483706889510625476608)
def resultEvent : Nat := 216195
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216195.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge216192.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216188.owner)
    (rightOwner := SemanticResult216010.owner)
    (leftResult := 216188) (rightResult := 216010)
    (leftActual := SemanticResult216188.actual selector witness)
    (rightActual := SemanticResult216010.actual selector witness)
    (leftRaw := SemanticResult216188.rawTerms)
    (rightRaw := SemanticResult216010.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32188807212483504816668771614720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216189) (rightBinding := 216190)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16599⟩) (rightExpression := ⟨17763⟩)
    (coefficientTransfer := 216191) (summaryTransfer := 216194)
    (base := LeftOperatorMerge216192.base)
    (reconstruction := LeftOperatorMerge216192.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216188.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult216010.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge216192.operationAgreement
  · rfl
  · decide
end SemanticResult216195

namespace SemanticResult216200
def owner : Owner := ⟨.program ⟨257⟩, ⟨20656⟩⟩
def rawTerms : List Term := Proof.Events844.exact216200RawTerms
def summary : Bound := (.finite 64377712650190257467641695830016)
def resultEvent : Nat := 216200
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216200.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216195.owner)
    (rightOwner := SemanticResult215713.owner)
    (leftResult := 216195) (rightResult := 215713)
    (leftActual := SemanticResult216195.actual selector witness)
    (rightActual := SemanticResult215713.actual selector witness)
    (leftRaw := SemanticResult216195.rawTerms)
    (rightRaw := SemanticResult215713.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 32188807212483706889510625476608)
    (rightMaximum := 32188905437706550578131070353408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216196) (rightBinding := 216197)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17764⟩) (rightExpression := ⟨20655⟩)
    (transferEvent := 216198) (summaryTransferEvent := 216199)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216195.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult215713.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult216200

namespace SemanticResult216205
def owner : Owner := ⟨.program ⟨257⟩, ⟨23876⟩⟩
def rawTerms : List Term := Proof.Events844.exact216205RawTerms
def summary : Bound := (.finite 96566716313119651734393211060224)
def resultEvent : Nat := 216205
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216205.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216200.owner)
    (rightOwner := SemanticResult215231.owner)
    (leftResult := 216200) (rightResult := 215231)
    (leftActual := SemanticResult216200.actual selector witness)
    (rightActual := SemanticResult215231.actual selector witness)
    (leftRaw := SemanticResult216200.rawTerms)
    (rightRaw := SemanticResult215231.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 64377712650190257467641695830016)
    (rightMaximum := 32189003662929394266751515230208) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216201) (rightBinding := 216202)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20656⟩) (rightExpression := ⟨23875⟩)
    (transferEvent := 216203) (summaryTransferEvent := 216204)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216200.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult215231.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult216205

namespace SemanticResult216210
def owner : Owner := ⟨.program ⟨257⟩, ⟨33896⟩⟩
def rawTerms : List Term := Proof.Events844.exact216210RawTerms
def summary : Bound := (.finite 128755916426494733378385616044032)
def resultEvent : Nat := 216210
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216210.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216205.owner)
    (rightOwner := SemanticResult214749.owner)
    (leftResult := 216205) (rightResult := 214749)
    (leftActual := SemanticResult216205.actual selector witness)
    (rightActual := SemanticResult214749.actual selector witness)
    (leftRaw := SemanticResult216205.rawTerms)
    (rightRaw := SemanticResult214749.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 96566716313119651734393211060224)
    (rightMaximum := 32189200113375081643992404983808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216206) (rightBinding := 216207)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23876⟩) (rightExpression := ⟨33895⟩)
    (transferEvent := 216208) (summaryTransferEvent := 216209)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216205.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult214749.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult216210

namespace SemanticResult216215
def owner : Owner := ⟨.program ⟨257⟩, ⟨52956⟩⟩
def rawTerms : List Term := Proof.Events844.exact216215RawTerms
def summary : Bound := (.finite 160945509440761189776859800535040)
def resultEvent : Nat := 216215
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216215.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216210.owner)
    (rightOwner := SemanticResult214267.owner)
    (leftResult := 216210) (rightResult := 214267)
    (leftActual := SemanticResult216210.actual selector witness)
    (rightActual := SemanticResult214267.actual selector witness)
    (leftRaw := SemanticResult216210.rawTerms)
    (rightRaw := SemanticResult214267.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 128755916426494733378385616044032)
    (rightMaximum := 32189593014266456398474184491008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216211) (rightBinding := 216212)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33896⟩) (rightExpression := ⟨52955⟩)
    (transferEvent := 216213) (summaryTransferEvent := 216214)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216210.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult214267.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult216215

namespace SemanticResult216220
def owner : Owner := ⟨.program ⟨257⟩, ⟨55936⟩⟩
def rawTerms : List Term := Proof.Events844.exact216220RawTerms
def summary : Bound := (.finite 193135298905473333552574874779648)
def resultEvent : Nat := 216220
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216220.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216215.owner)
    (rightOwner := SemanticResult213785.owner)
    (leftResult := 216215) (rightResult := 213785)
    (leftActual := SemanticResult216215.actual selector witness)
    (rightActual := SemanticResult213785.actual selector witness)
    (leftRaw := SemanticResult216215.rawTerms)
    (rightRaw := SemanticResult213785.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 160945509440761189776859800535040)
    (rightMaximum := 32189789464712143775715074244608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216216) (rightBinding := 216217)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨52956⟩) (rightExpression := ⟨55935⟩)
    (transferEvent := 216218) (summaryTransferEvent := 216219)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216215.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult213785.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult216220

namespace SemanticResult216225
def owner : Owner := ⟨.program ⟨257⟩, ⟨58916⟩⟩
def rawTerms : List Term := Proof.Events844.exact216225RawTerms
def summary : Bound := (.finite 225325481271076852082771728531456)
def resultEvent : Nat := 216225
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216225.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216220.owner)
    (rightOwner := SemanticResult213303.owner)
    (leftResult := 216220) (rightResult := 213303)
    (leftActual := SemanticResult216220.actual selector witness)
    (rightActual := SemanticResult213303.actual selector witness)
    (leftRaw := SemanticResult216220.rawTerms)
    (rightRaw := SemanticResult213303.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 193135298905473333552574874779648)
    (rightMaximum := 32190182365603518530196853751808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216221) (rightBinding := 216222)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55936⟩) (rightExpression := ⟨58915⟩)
    (transferEvent := 216223) (summaryTransferEvent := 216224)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216220.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult213303.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult216225

namespace SemanticResult216230
def owner : Owner := ⟨.program ⟨257⟩, ⟨61896⟩⟩
def rawTerms : List Term := Proof.Events844.exact216230RawTerms
def summary : Bound := (.finite 257515860087126057990209472036864)
def resultEvent : Nat := 216230
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216230.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216225.owner)
    (rightOwner := SemanticResult212821.owner)
    (leftResult := 216225) (rightResult := 212821)
    (leftActual := SemanticResult216225.actual selector witness)
    (rightActual := SemanticResult212821.actual selector witness)
    (leftRaw := SemanticResult216225.rawTerms)
    (rightRaw := SemanticResult212821.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 225325481271076852082771728531456)
    (rightMaximum := 32190378816049205907437743505408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216226) (rightBinding := 216227)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58916⟩) (rightExpression := ⟨61895⟩)
    (transferEvent := 216228) (summaryTransferEvent := 216229)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216225.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult212821.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult216230

namespace SemanticResult216235
def owner : Owner := ⟨.program ⟨257⟩, ⟨64876⟩⟩
def rawTerms : List Term := Proof.Events844.exact216235RawTerms
def summary : Bound := (.finite 289706631804066638652128995049472)
def resultEvent : Nat := 216235
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216235.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216230.owner)
    (rightOwner := SemanticResult212339.owner)
    (leftResult := 216230) (rightResult := 212339)
    (leftActual := SemanticResult216230.actual selector witness)
    (rightActual := SemanticResult212339.actual selector witness)
    (leftRaw := SemanticResult216230.rawTerms)
    (rightRaw := SemanticResult212339.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 257515860087126057990209472036864)
    (rightMaximum := 32190771716940580661919523012608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216231) (rightBinding := 216232)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61896⟩) (rightExpression := ⟨64875⟩)
    (transferEvent := 216233) (summaryTransferEvent := 216234)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216230.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult212339.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult216235

namespace SemanticResult216240
def owner : Owner := ⟨.program ⟨257⟩, ⟨70181⟩⟩
def rawTerms : List Term := Proof.Events844.exact216240RawTerms
def summary : Bound := (.finite 321897992872344281445771187322880)
def resultEvent : Nat := 216240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216240.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216235.owner)
    (rightOwner := SemanticResult211857.owner)
    (leftResult := 216235) (rightResult := 211857)
    (leftActual := SemanticResult216235.actual selector witness)
    (rightActual := SemanticResult211857.actual selector witness)
    (leftRaw := SemanticResult216235.rawTerms)
    (rightRaw := SemanticResult211857.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 289706631804066638652128995049472)
    (rightMaximum := 32191361068277642793642192273408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216236) (rightBinding := 216237)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64876⟩) (rightExpression := ⟨70180⟩)
    (transferEvent := 216238) (summaryTransferEvent := 216239)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216235.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult211857.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult216240

namespace SemanticResult216245
def owner : Owner := ⟨.program ⟨257⟩, ⟨70182⟩⟩
def rawTerms : List Term := Proof.Events844.exact216245RawTerms
def summary : Bound := (.finite 354089550391067611616654269349888)
def resultEvent : Nat := 216245
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216245.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216240.owner)
    (rightOwner := SemanticResult211375.owner)
    (leftResult := 216240) (rightResult := 211375)
    (leftActual := SemanticResult216240.actual selector witness)
    (rightActual := SemanticResult211375.actual selector witness)
    (leftRaw := SemanticResult216240.rawTerms)
    (rightRaw := SemanticResult211375.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 321897992872344281445771187322880)
    (rightMaximum := 32191557518723330170883082027008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216241) (rightBinding := 216242)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70181⟩) (rightExpression := ⟨28292⟩)
    (transferEvent := 216243) (summaryTransferEvent := 216244)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216240.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult211375.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult216245

namespace SemanticResult216250
def owner : Owner := ⟨.program ⟨257⟩, ⟨70183⟩⟩
def rawTerms : List Term := Proof.Events844.exact216250RawTerms
def summary : Bound := (.finite 386281697261128003919260020637696)
def resultEvent : Nat := 216250
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216250.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216245.owner)
    (rightOwner := SemanticResult210893.owner)
    (leftResult := 216245) (rightResult := 210893)
    (leftActual := SemanticResult216245.actual selector witness)
    (rightActual := SemanticResult210893.actual selector witness)
    (leftRaw := SemanticResult216245.rawTerms)
    (rightRaw := SemanticResult210893.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 354089550391067611616654269349888)
    (rightMaximum := 32192146870060392302605751287808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216246) (rightBinding := 216247)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70182⟩) (rightExpression := ⟨30972⟩)
    (transferEvent := 216248) (summaryTransferEvent := 216249)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216245.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult210893.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult216250

namespace SemanticResult216255
def owner : Owner := ⟨.program ⟨257⟩, ⟨70184⟩⟩
def rawTerms : List Term := Proof.Events844.exact216255RawTerms
def summary : Bound := (.finite 418474237032079770976347551432704)
def resultEvent : Nat := 216255
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216255.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216250.owner)
    (rightOwner := SemanticResult210411.owner)
    (leftResult := 216250) (rightResult := 210411)
    (leftActual := SemanticResult216250.actual selector witness)
    (rightActual := SemanticResult210411.actual selector witness)
    (leftRaw := SemanticResult216250.rawTerms)
    (rightRaw := SemanticResult210411.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 386281697261128003919260020637696)
    (rightMaximum := 32192539770951767057087530795008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216251) (rightBinding := 216252)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70183⟩) (rightExpression := ⟨36632⟩)
    (transferEvent := 216253) (summaryTransferEvent := 216254)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216250.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult210411.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult216255

namespace SemanticResult216260
def owner : Owner := ⟨.program ⟨257⟩, ⟨70185⟩⟩
def rawTerms : List Term := Proof.Events844.exact216260RawTerms
def summary : Bound := (.finite 450666973253477225410675971981312)
def resultEvent : Nat := 216260
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216260.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216255.owner)
    (rightOwner := SemanticResult209929.owner)
    (leftResult := 216255) (rightResult := 209929)
    (leftActual := SemanticResult216255.actual selector witness)
    (rightActual := SemanticResult209929.actual selector witness)
    (leftRaw := SemanticResult216255.rawTerms)
    (rightRaw := SemanticResult209929.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 418474237032079770976347551432704)
    (rightMaximum := 32192736221397454434328420548608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216256) (rightBinding := 216257)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70184⟩) (rightExpression := ⟨39312⟩)
    (transferEvent := 216258) (summaryTransferEvent := 216259)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216255.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult209929.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult216260

namespace SemanticResult216265
def owner : Owner := ⟨.program ⟨257⟩, ⟨70186⟩⟩
def rawTerms : List Term := Proof.Events844.exact216265RawTerms
def summary : Bound := (.finite 482860102375766054599486172037120)
def resultEvent : Nat := 216265
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216265.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216260.owner)
    (rightOwner := SemanticResult209447.owner)
    (leftResult := 216260) (rightResult := 209447)
    (leftActual := SemanticResult216260.actual selector witness)
    (rightActual := SemanticResult209447.actual selector witness)
    (leftRaw := SemanticResult216260.rawTerms)
    (rightRaw := SemanticResult209447.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 450666973253477225410675971981312)
    (rightMaximum := 32193129122288829188810200055808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216261) (rightBinding := 216262)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70185⟩) (rightExpression := ⟨41992⟩)
    (transferEvent := 216263) (summaryTransferEvent := 216264)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216260.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult209447.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult216265

namespace SemanticResult216270
def owner : Owner := ⟨.program ⟨257⟩, ⟨70187⟩⟩
def rawTerms : List Term := Proof.Events844.exact216270RawTerms
def summary : Bound := (.finite 515053820849391945920019041353728)
def resultEvent : Nat := 216270
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult216270.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult216265.owner)
    (rightOwner := SemanticResult208965.owner)
    (leftResult := 216265) (rightResult := 208965)
    (leftActual := SemanticResult216265.actual selector witness)
    (rightActual := SemanticResult208965.actual selector witness)
    (leftRaw := SemanticResult216265.rawTerms)
    (rightRaw := SemanticResult208965.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 482860102375766054599486172037120)
    (rightMaximum := 32193718473625891320532869316608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 216266) (rightBinding := 216267)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70186⟩) (rightExpression := ⟨44672⟩)
    (transferEvent := 216268) (summaryTransferEvent := 216269)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult216265.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult208965.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult216270

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
