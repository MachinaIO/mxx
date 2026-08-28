import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard2160
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard250
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard350
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard451
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard551
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard652
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard752
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard853
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard954
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1054
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1155
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1255
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1356
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1456
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1557
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1657
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1758
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard2159

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult308125
def owner : Owner := ⟨.program ⟨257⟩, ⟨71183⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308125RawTerms
def summary : Bound := (.finite 30808454790312530031291914359231165163455306056856023605184929939366871092)
def resultEvent : Nat := 308125
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308125.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308120.owner)
    (rightOwner := SemanticResult251234.owner)
    (leftResult := 308120) (rightResult := 251234)
    (leftActual := SemanticResult308120.actual selector witness)
    (rightActual := SemanticResult251234.actual selector witness)
    (leftRaw := SemanticResult308120.rawTerms)
    (rightRaw := SemanticResult251234.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 23106341092913726332435000681198127318305096537183840368456281091158835252)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308121) (rightBinding := 308122)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71093⟩) (rightExpression := ⟨71182⟩)
    (transferEvent := 308123) (summaryTransferEvent := 308124)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308120.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult251234.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308125

namespace SemanticResult308130
def owner : Owner := ⟨.program ⟨257⟩, ⟨71215⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308130RawTerms
def summary : Bound := (.finite 38510568487711333730148828037264203008605515576528206841913578787574906932)
def resultEvent : Nat := 308130
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308130.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308125.owner)
    (rightOwner := SemanticResult236609.owner)
    (leftResult := 308125) (rightResult := 236609)
    (leftActual := SemanticResult308125.actual selector witness)
    (rightActual := SemanticResult236609.actual selector witness)
    (leftRaw := SemanticResult308125.rawTerms)
    (rightRaw := SemanticResult236609.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 30808454790312530031291914359231165163455306056856023605184929939366871092)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308126) (rightBinding := 308127)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71183⟩) (rightExpression := ⟨71214⟩)
    (transferEvent := 308128) (summaryTransferEvent := 308129)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308125.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult236609.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308130

namespace SemanticResult308135
def owner : Owner := ⟨.program ⟨257⟩, ⟨71247⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308135RawTerms
def summary : Bound := (.finite 46212682185110137429005741715297240853755725096200390078642227635782942772)
def resultEvent : Nat := 308135
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308135.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308130.owner)
    (rightOwner := SemanticResult221984.owner)
    (leftResult := 308130) (rightResult := 221984)
    (leftActual := SemanticResult308130.actual selector witness)
    (rightActual := SemanticResult221984.actual selector witness)
    (leftRaw := SemanticResult308130.rawTerms)
    (rightRaw := SemanticResult221984.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 38510568487711333730148828037264203008605515576528206841913578787574906932)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308131) (rightBinding := 308132)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71215⟩) (rightExpression := ⟨71246⟩)
    (transferEvent := 308133) (summaryTransferEvent := 308134)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308130.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult221984.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308135

namespace SemanticResult308140
def owner : Owner := ⟨.program ⟨257⟩, ⟨71308⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308140RawTerms
def summary : Bound := (.finite 53914795882508941127862655393330278698905934615872573315370876483990978612)
def resultEvent : Nat := 308140
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308140.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308135.owner)
    (rightOwner := SemanticResult207359.owner)
    (leftResult := 308135) (rightResult := 207359)
    (leftActual := SemanticResult308135.actual selector witness)
    (rightActual := SemanticResult207359.actual selector witness)
    (leftRaw := SemanticResult308135.rawTerms)
    (rightRaw := SemanticResult207359.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 46212682185110137429005741715297240853755725096200390078642227635782942772)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308136) (rightBinding := 308137)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71247⟩) (rightExpression := ⟨71307⟩)
    (transferEvent := 308138) (summaryTransferEvent := 308139)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308135.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult207359.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308140

namespace SemanticResult308145
def owner : Owner := ⟨.program ⟨257⟩, ⟨71340⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308145RawTerms
def summary : Bound := (.finite 61616909579907744826719569071363316544056144135544756552099525332199014452)
def resultEvent : Nat := 308145
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308145.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308140.owner)
    (rightOwner := SemanticResult192734.owner)
    (leftResult := 308140) (rightResult := 192734)
    (leftActual := SemanticResult308140.actual selector witness)
    (rightActual := SemanticResult192734.actual selector witness)
    (leftRaw := SemanticResult308140.rawTerms)
    (rightRaw := SemanticResult192734.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 53914795882508941127862655393330278698905934615872573315370876483990978612)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308141) (rightBinding := 308142)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71308⟩) (rightExpression := ⟨71339⟩)
    (transferEvent := 308143) (summaryTransferEvent := 308144)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308140.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult192734.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308145

namespace SemanticResult308150
def owner : Owner := ⟨.program ⟨257⟩, ⟨71376⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308150RawTerms
def summary : Bound := (.finite 69319023277306548525576482749396354389206353655216939788828174180407050292)
def resultEvent : Nat := 308150
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308150.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308145.owner)
    (rightOwner := SemanticResult178109.owner)
    (leftResult := 308145) (rightResult := 178109)
    (leftActual := SemanticResult308145.actual selector witness)
    (rightActual := SemanticResult178109.actual selector witness)
    (leftRaw := SemanticResult308145.rawTerms)
    (rightRaw := SemanticResult178109.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 61616909579907744826719569071363316544056144135544756552099525332199014452)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308146) (rightBinding := 308147)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71340⟩) (rightExpression := ⟨71375⟩)
    (transferEvent := 308148) (summaryTransferEvent := 308149)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308145.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult178109.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308150

namespace SemanticResult308155
def owner : Owner := ⟨.program ⟨257⟩, ⟨71377⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308155RawTerms
def summary : Bound := (.finite 77021136974705352224433396427429392234356563174889123025556823028615086132)
def resultEvent : Nat := 308155
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308155.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308150.owner)
    (rightOwner := SemanticResult163484.owner)
    (leftResult := 308150) (rightResult := 163484)
    (leftActual := SemanticResult308150.actual selector witness)
    (rightActual := SemanticResult163484.actual selector witness)
    (leftRaw := SemanticResult308150.rawTerms)
    (rightRaw := SemanticResult163484.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 69319023277306548525576482749396354389206353655216939788828174180407050292)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308151) (rightBinding := 308152)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71376⟩) (rightExpression := ⟨71152⟩)
    (transferEvent := 308153) (summaryTransferEvent := 308154)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308150.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult163484.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308155

namespace SemanticResult308160
def owner : Owner := ⟨.program ⟨257⟩, ⟨71378⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308160RawTerms
def summary : Bound := (.finite 84723250672104155923290310105462430079506772694561306262285471876823121972)
def resultEvent : Nat := 308160
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308160.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308155.owner)
    (rightOwner := SemanticResult148859.owner)
    (leftResult := 308155) (rightResult := 148859)
    (leftActual := SemanticResult308155.actual selector witness)
    (rightActual := SemanticResult148859.actual selector witness)
    (leftRaw := SemanticResult308155.rawTerms)
    (rightRaw := SemanticResult148859.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 77021136974705352224433396427429392234356563174889123025556823028615086132)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308156) (rightBinding := 308157)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71377⟩) (rightExpression := ⟨71027⟩)
    (transferEvent := 308158) (summaryTransferEvent := 308159)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308155.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult148859.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308160

namespace SemanticResult308165
def owner : Owner := ⟨.program ⟨257⟩, ⟨71379⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308165RawTerms
def summary : Bound := (.finite 92425364369502959622147223783495467924656982214233489499014120725031157812)
def resultEvent : Nat := 308165
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308165.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308160.owner)
    (rightOwner := SemanticResult134234.owner)
    (leftResult := 308160) (rightResult := 134234)
    (leftActual := SemanticResult308160.actual selector witness)
    (rightActual := SemanticResult134234.actual selector witness)
    (leftRaw := SemanticResult308160.rawTerms)
    (rightRaw := SemanticResult134234.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 84723250672104155923290310105462430079506772694561306262285471876823121972)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308161) (rightBinding := 308162)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71378⟩) (rightExpression := ⟨71123⟩)
    (transferEvent := 308163) (summaryTransferEvent := 308164)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308160.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult134234.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308165

namespace SemanticResult308170
def owner : Owner := ⟨.program ⟨257⟩, ⟨71380⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308170RawTerms
def summary : Bound := (.finite 100127478066901763321004137461528505769807191733905672735742769573239193652)
def resultEvent : Nat := 308170
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308170.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308165.owner)
    (rightOwner := SemanticResult119609.owner)
    (leftResult := 308165) (rightResult := 119609)
    (leftActual := SemanticResult308165.actual selector witness)
    (rightActual := SemanticResult119609.actual selector witness)
    (leftRaw := SemanticResult308165.rawTerms)
    (rightRaw := SemanticResult119609.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 92425364369502959622147223783495467924656982214233489499014120725031157812)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308166) (rightBinding := 308167)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71379⟩) (rightExpression := ⟨71277⟩)
    (transferEvent := 308168) (summaryTransferEvent := 308169)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult119609.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308170

namespace SemanticResult308175
def owner : Owner := ⟨.program ⟨257⟩, ⟨71416⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308175RawTerms
def summary : Bound := (.finite 107829591764300567019861051139561543614957401253577855972471418421447229492)
def resultEvent : Nat := 308175
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308175.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308170.owner)
    (rightOwner := SemanticResult104984.owner)
    (leftResult := 308170) (rightResult := 104984)
    (leftActual := SemanticResult308170.actual selector witness)
    (rightActual := SemanticResult104984.actual selector witness)
    (leftRaw := SemanticResult308170.rawTerms)
    (rightRaw := SemanticResult104984.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 100127478066901763321004137461528505769807191733905672735742769573239193652)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308171) (rightBinding := 308172)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71380⟩) (rightExpression := ⟨71415⟩)
    (transferEvent := 308173) (summaryTransferEvent := 308174)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308170.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult104984.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308175

namespace SemanticResult308180
def owner : Owner := ⟨.program ⟨257⟩, ⟨71448⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308180RawTerms
def summary : Bound := (.finite 115531705461699370718717964817594581460107610773250039209200067269655265332)
def resultEvent : Nat := 308180
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308180.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308175.owner)
    (rightOwner := SemanticResult90359.owner)
    (leftResult := 308175) (rightResult := 90359)
    (leftActual := SemanticResult308175.actual selector witness)
    (rightActual := SemanticResult90359.actual selector witness)
    (leftRaw := SemanticResult308175.rawTerms)
    (rightRaw := SemanticResult90359.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 107829591764300567019861051139561543614957401253577855972471418421447229492)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308176) (rightBinding := 308177)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71416⟩) (rightExpression := ⟨71447⟩)
    (transferEvent := 308178) (summaryTransferEvent := 308179)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308175.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult90359.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308180

namespace SemanticResult308185
def owner : Owner := ⟨.program ⟨257⟩, ⟨71480⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308185RawTerms
def summary : Bound := (.finite 123233819159098174417574878495627619305257820292922222445928716117863301172)
def resultEvent : Nat := 308185
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308185.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308180.owner)
    (rightOwner := SemanticResult75734.owner)
    (leftResult := 308180) (rightResult := 75734)
    (leftActual := SemanticResult308180.actual selector witness)
    (rightActual := SemanticResult75734.actual selector witness)
    (leftRaw := SemanticResult308180.rawTerms)
    (rightRaw := SemanticResult75734.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 115531705461699370718717964817594581460107610773250039209200067269655265332)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308181) (rightBinding := 308182)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71448⟩) (rightExpression := ⟨71479⟩)
    (transferEvent := 308183) (summaryTransferEvent := 308184)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308180.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75734.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308185

namespace SemanticResult308190
def owner : Owner := ⟨.program ⟨257⟩, ⟨71512⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308190RawTerms
def summary : Bound := (.finite 130935932856496978116431792173660657150408029812594405682657364966071337012)
def resultEvent : Nat := 308190
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308190.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308185.owner)
    (rightOwner := SemanticResult61109.owner)
    (leftResult := 308185) (rightResult := 61109)
    (leftActual := SemanticResult308185.actual selector witness)
    (rightActual := SemanticResult61109.actual selector witness)
    (leftRaw := SemanticResult308185.rawTerms)
    (rightRaw := SemanticResult61109.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 123233819159098174417574878495627619305257820292922222445928716117863301172)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308186) (rightBinding := 308187)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71480⟩) (rightExpression := ⟨71511⟩)
    (transferEvent := 308188) (summaryTransferEvent := 308189)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308185.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult61109.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308190

namespace SemanticResult308195
def owner : Owner := ⟨.program ⟨257⟩, ⟨71545⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308195RawTerms
def summary : Bound := (.finite 138638046553895781815288705851693694995558239332266588919386013814279372852)
def resultEvent : Nat := 308195
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308195.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308190.owner)
    (rightOwner := SemanticResult46484.owner)
    (leftResult := 308190) (rightResult := 46484)
    (leftActual := SemanticResult308190.actual selector witness)
    (rightActual := SemanticResult46484.actual selector witness)
    (leftRaw := SemanticResult308190.rawTerms)
    (rightRaw := SemanticResult46484.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 130935932856496978116431792173660657150408029812594405682657364966071337012)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308191) (rightBinding := 308192)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71512⟩) (rightExpression := ⟨71544⟩)
    (transferEvent := 308193) (summaryTransferEvent := 308194)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308190.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult46484.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308195

namespace SemanticResult308200
def owner : Owner := ⟨.program ⟨257⟩, ⟨71546⟩⟩
def rawTerms : List Term := Proof.Events1203.exact308200RawTerms
def summary : Bound := (.finite 146340160251294585514145619529726732840708448851938772156114662662487408692)
def resultEvent : Nat := 308200
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308200.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult308195.owner)
    (rightOwner := SemanticResult31859.owner)
    (leftResult := 308195) (rightResult := 31859)
    (leftActual := SemanticResult308195.actual selector witness)
    (rightActual := SemanticResult31859.actual selector witness)
    (leftRaw := SemanticResult308195.rawTerms)
    (rightRaw := SemanticResult31859.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 138638046553895781815288705851693694995558239332266588919386013814279372852)
    (rightMaximum := 7702113697398803698856913678033037845150209519672183236728648848208035840) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308196) (rightBinding := 308197)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71545⟩) (rightExpression := ⟨70978⟩)
    (transferEvent := 308198) (summaryTransferEvent := 308199)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308195.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31859.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult308200

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
