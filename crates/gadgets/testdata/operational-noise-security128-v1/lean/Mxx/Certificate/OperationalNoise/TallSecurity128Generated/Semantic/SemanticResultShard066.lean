import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard066
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard060
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard063
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard064
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard065

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult8236
def owner : Owner := ⟨.program ⟨257⟩, ⟨57203⟩⟩
def rawTerms : List Term := Proof.Events032.exact8236RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8236
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8236.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8233) (rightBinding := 8234)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54223⟩) (rightExpression := ⟨57202⟩)
    (transferEvent := 8235)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8232.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8160.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8236

namespace SemanticResult8240
def owner : Owner := ⟨.program ⟨257⟩, ⟨60183⟩⟩
def rawTerms : List Term := Proof.Events032.exact8240RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8240.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8237) (rightBinding := 8238)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57203⟩) (rightExpression := ⟨60182⟩)
    (transferEvent := 8239)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8236.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8152.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8240

namespace SemanticResult8244
def owner : Owner := ⟨.program ⟨257⟩, ⟨63163⟩⟩
def rawTerms : List Term := Proof.Events032.exact8244RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8244
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8244.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8241) (rightBinding := 8242)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60183⟩) (rightExpression := ⟨63162⟩)
    (transferEvent := 8243)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8240.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8144.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8244

namespace SemanticResult8248
def owner : Owner := ⟨.program ⟨257⟩, ⟨66870⟩⟩
def rawTerms : List Term := Proof.Events032.exact8248RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8248
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8248.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8245) (rightBinding := 8246)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63163⟩) (rightExpression := ⟨66869⟩)
    (transferEvent := 8247)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8244.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8136.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8248

namespace SemanticResult8252
def owner : Owner := ⟨.program ⟨257⟩, ⟨66871⟩⟩
def rawTerms : List Term := Proof.Events032.exact8252RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8252
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8252.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8249) (rightBinding := 8250)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66870⟩) (rightExpression := ⟨26675⟩)
    (transferEvent := 8251)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8248.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8128.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8252

namespace SemanticResult8256
def owner : Owner := ⟨.program ⟨257⟩, ⟨66872⟩⟩
def rawTerms : List Term := Proof.Events032.exact8256RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8256
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8256.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8253) (rightBinding := 8254)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66871⟩) (rightExpression := ⟨29355⟩)
    (transferEvent := 8255)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8252.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8120.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8256

namespace SemanticResult8260
def owner : Owner := ⟨.program ⟨257⟩, ⟨66873⟩⟩
def rawTerms : List Term := Proof.Events032.exact8260RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8260
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8260.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8257) (rightBinding := 8258)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66872⟩) (rightExpression := ⟨35012⟩)
    (transferEvent := 8259)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8256.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8112.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8260

namespace SemanticResult8264
def owner : Owner := ⟨.program ⟨257⟩, ⟨66874⟩⟩
def rawTerms : List Term := Proof.Events032.exact8264RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8264
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8264.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8261) (rightBinding := 8262)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66873⟩) (rightExpression := ⟨37692⟩)
    (transferEvent := 8263)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8260.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8104.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8264

namespace SemanticResult8268
def owner : Owner := ⟨.program ⟨257⟩, ⟨66875⟩⟩
def rawTerms : List Term := Proof.Events032.exact8268RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8268
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8268.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8265) (rightBinding := 8266)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66874⟩) (rightExpression := ⟨40375⟩)
    (transferEvent := 8267)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8264.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8096.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8268

namespace SemanticResult8272
def owner : Owner := ⟨.program ⟨257⟩, ⟨66876⟩⟩
def rawTerms : List Term := Proof.Events032.exact8272RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8272
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8272.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8269) (rightBinding := 8270)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66875⟩) (rightExpression := ⟨43055⟩)
    (transferEvent := 8271)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8268.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8088.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8272

namespace SemanticResult8276
def owner : Owner := ⟨.program ⟨257⟩, ⟨66877⟩⟩
def rawTerms : List Term := Proof.Events032.exact8276RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8276
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8276.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8273) (rightBinding := 8274)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66876⟩) (rightExpression := ⟨45732⟩)
    (transferEvent := 8275)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8272.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8080.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8276

namespace SemanticResult8280
def owner : Owner := ⟨.program ⟨257⟩, ⟨66878⟩⟩
def rawTerms : List Term := Proof.Events032.exact8280RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8280
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8280.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8277) (rightBinding := 8278)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66877⟩) (rightExpression := ⟨48412⟩)
    (transferEvent := 8279)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8276.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8072.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8280

namespace SemanticResult8284
def owner : Owner := ⟨.program ⟨257⟩, ⟨67541⟩⟩
def rawTerms : List Term := Proof.Events032.exact8284RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8284
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8284.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 8281) (rightBinding := 8282)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66878⟩) (rightExpression := ⟨67539⟩)
    (transferEvent := 8283)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult8280.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8064.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult8284

namespace SemanticResult8307
def owner : Owner := ⟨.program ⟨257⟩, ⟨67542⟩⟩
def rawTerms : List Term := Proof.Events032.exact8307RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8307
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8307.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge8288.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge8288.frameStart)
    (transferEvent := 8287) (owner := owner)
    (leftResult := 8284) (rightResult := 7561)
    (working := LeftOperatorMerge8288.working)
    (reconstruction := LeftOperatorMerge8288.reconstruction)
    (leftReference := .predecessor 0 8285 .coefficient) (rightReference := .predecessor 1 8286 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult8284.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7561.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge8288.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult8307

namespace SemanticResult8309
def owner : Owner := ⟨.program ⟨257⟩, ⟨6806⟩⟩
def rawTerms : List Term := Proof.Events032.exact8309RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8309
def producerEvent : Nat := 8308
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8309.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 0, .finite 392208910876296843290869724658024391949918004018017135461780498791886113798803788492196058508406193818925718552453952606142266741361954240447112917026659566933549801769, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult8309

namespace SemanticResult8322
def owner : Owner := ⟨.program ⟨257⟩, ⟨47906⟩⟩
def rawTerms : List Term := Proof.Events032.exact8322RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 8322
def producerEvent : Nat := 8321
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult8322.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 0, .finite 60, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult8322

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
