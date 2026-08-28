import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard042
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard039
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard040
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard041

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult5228
def owner : Owner := ⟨.program ⟨257⟩, ⟨22102⟩⟩
def rawTerms : List Term := Proof.Events020.exact5228RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5228
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5228.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5225) (rightBinding := 5226)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18882⟩) (rightExpression := ⟨22101⟩)
    (transferEvent := 5227)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5224.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5200.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5228

namespace SemanticResult5232
def owner : Owner := ⟨.program ⟨257⟩, ⟨32122⟩⟩
def rawTerms : List Term := Proof.Events020.exact5232RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5232
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5232.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5229) (rightBinding := 5230)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22102⟩) (rightExpression := ⟨32121⟩)
    (transferEvent := 5231)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5228.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5192.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5232

namespace SemanticResult5236
def owner : Owner := ⟨.program ⟨257⟩, ⟨51186⟩⟩
def rawTerms : List Term := Proof.Events020.exact5236RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5236
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5236.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5233) (rightBinding := 5234)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32122⟩) (rightExpression := ⟨51185⟩)
    (transferEvent := 5235)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5232.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5184.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5236

namespace SemanticResult5240
def owner : Owner := ⟨.program ⟨257⟩, ⟨54166⟩⟩
def rawTerms : List Term := Proof.Events020.exact5240RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5240.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5237) (rightBinding := 5238)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51186⟩) (rightExpression := ⟨54165⟩)
    (transferEvent := 5239)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5236.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5176.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5240

namespace SemanticResult5244
def owner : Owner := ⟨.program ⟨257⟩, ⟨57146⟩⟩
def rawTerms : List Term := Proof.Events020.exact5244RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5244
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5244.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5241) (rightBinding := 5242)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54166⟩) (rightExpression := ⟨57145⟩)
    (transferEvent := 5243)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5240.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5168.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5244

namespace SemanticResult5248
def owner : Owner := ⟨.program ⟨257⟩, ⟨60126⟩⟩
def rawTerms : List Term := Proof.Events020.exact5248RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5248
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5248.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5245) (rightBinding := 5246)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57146⟩) (rightExpression := ⟨60125⟩)
    (transferEvent := 5247)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5244.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5160.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5248

namespace SemanticResult5252
def owner : Owner := ⟨.program ⟨257⟩, ⟨63106⟩⟩
def rawTerms : List Term := Proof.Events020.exact5252RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5252
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5252.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5249) (rightBinding := 5250)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60126⟩) (rightExpression := ⟨63105⟩)
    (transferEvent := 5251)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5248.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5152.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5252

namespace SemanticResult5256
def owner : Owner := ⟨.program ⟨257⟩, ⟨66660⟩⟩
def rawTerms : List Term := Proof.Events020.exact5256RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5256
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5256.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5253) (rightBinding := 5254)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63106⟩) (rightExpression := ⟨66659⟩)
    (transferEvent := 5255)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5252.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5144.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5256

namespace SemanticResult5260
def owner : Owner := ⟨.program ⟨257⟩, ⟨66661⟩⟩
def rawTerms : List Term := Proof.Events020.exact5260RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5260
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5260.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5257) (rightBinding := 5258)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66660⟩) (rightExpression := ⟨26636⟩)
    (transferEvent := 5259)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5256.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5136.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5260

namespace SemanticResult5264
def owner : Owner := ⟨.program ⟨257⟩, ⟨66662⟩⟩
def rawTerms : List Term := Proof.Events020.exact5264RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5264
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5264.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5261) (rightBinding := 5262)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66661⟩) (rightExpression := ⟨29316⟩)
    (transferEvent := 5263)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5260.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5128.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5264

namespace SemanticResult5268
def owner : Owner := ⟨.program ⟨257⟩, ⟨66663⟩⟩
def rawTerms : List Term := Proof.Events020.exact5268RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5268
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5268.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5265) (rightBinding := 5266)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66662⟩) (rightExpression := ⟨34973⟩)
    (transferEvent := 5267)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5264.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5120.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5268

namespace SemanticResult5272
def owner : Owner := ⟨.program ⟨257⟩, ⟨66664⟩⟩
def rawTerms : List Term := Proof.Events020.exact5272RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5272
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5272.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5269) (rightBinding := 5270)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66663⟩) (rightExpression := ⟨37653⟩)
    (transferEvent := 5271)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5268.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5112.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5272

namespace SemanticResult5276
def owner : Owner := ⟨.program ⟨257⟩, ⟨66665⟩⟩
def rawTerms : List Term := Proof.Events020.exact5276RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5276
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5276.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5273) (rightBinding := 5274)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66664⟩) (rightExpression := ⟨40336⟩)
    (transferEvent := 5275)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5272.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5104.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5276

namespace SemanticResult5280
def owner : Owner := ⟨.program ⟨257⟩, ⟨66666⟩⟩
def rawTerms : List Term := Proof.Events020.exact5280RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5280
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5280.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5277) (rightBinding := 5278)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66665⟩) (rightExpression := ⟨43016⟩)
    (transferEvent := 5279)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5276.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5096.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5280

namespace SemanticResult5284
def owner : Owner := ⟨.program ⟨257⟩, ⟨66667⟩⟩
def rawTerms : List Term := Proof.Events020.exact5284RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5284
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5284.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5281) (rightBinding := 5282)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66666⟩) (rightExpression := ⟨45693⟩)
    (transferEvent := 5283)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5280.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5088.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5284

namespace SemanticResult5288
def owner : Owner := ⟨.program ⟨257⟩, ⟨66668⟩⟩
def rawTerms : List Term := Proof.Events020.exact5288RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5288
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5288.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5285) (rightBinding := 5286)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66667⟩) (rightExpression := ⟨48373⟩)
    (transferEvent := 5287)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5284.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5080.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5288

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
