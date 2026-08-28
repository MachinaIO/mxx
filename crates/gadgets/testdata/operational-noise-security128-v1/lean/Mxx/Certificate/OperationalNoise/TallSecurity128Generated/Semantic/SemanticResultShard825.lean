import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard825
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard823
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard824

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult115193
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def rawTerms : List Term := Proof.Events449.exact115193RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115193
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115193.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115190) (rightBinding := 115191)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7198⟩) (rightExpression := ⟨7200⟩)
    (transferEvent := 115192)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115189.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115186.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115193

namespace SemanticResult115197
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def rawTerms : List Term := Proof.Events449.exact115197RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115197
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115197.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115194) (rightBinding := 115195)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7309⟩) (rightExpression := ⟨7202⟩)
    (transferEvent := 115196)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115193.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115183.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115197

namespace SemanticResult115201
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def rawTerms : List Term := Proof.Events450.exact115201RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115201
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115201.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115198) (rightBinding := 115199)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7310⟩) (rightExpression := ⟨7204⟩)
    (transferEvent := 115200)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115197.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115180.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115201

namespace SemanticResult115205
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def rawTerms : List Term := Proof.Events450.exact115205RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115205
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115205.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115202) (rightBinding := 115203)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7311⟩) (rightExpression := ⟨7206⟩)
    (transferEvent := 115204)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115201.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115177.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115205

namespace SemanticResult115209
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def rawTerms : List Term := Proof.Events450.exact115209RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115209
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115209.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115206) (rightBinding := 115207)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7312⟩) (rightExpression := ⟨7208⟩)
    (transferEvent := 115208)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115205.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115174.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115209

namespace SemanticResult115213
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def rawTerms : List Term := Proof.Events450.exact115213RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115213
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115213.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115210) (rightBinding := 115211)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7313⟩) (rightExpression := ⟨7210⟩)
    (transferEvent := 115212)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115209.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115171.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115213

namespace SemanticResult115217
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def rawTerms : List Term := Proof.Events450.exact115217RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115217
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115217.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115214) (rightBinding := 115215)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7314⟩) (rightExpression := ⟨7212⟩)
    (transferEvent := 115216)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115213.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115168.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115217

namespace SemanticResult115221
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def rawTerms : List Term := Proof.Events450.exact115221RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115221
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115221.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115218) (rightBinding := 115219)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7315⟩) (rightExpression := ⟨7214⟩)
    (transferEvent := 115220)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115217.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115165.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115221

namespace SemanticResult115225
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def rawTerms : List Term := Proof.Events450.exact115225RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115225
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115225.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115222) (rightBinding := 115223)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7316⟩) (rightExpression := ⟨7216⟩)
    (transferEvent := 115224)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115221.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115162.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115225

namespace SemanticResult115229
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def rawTerms : List Term := Proof.Events450.exact115229RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115229
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115229.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115226) (rightBinding := 115227)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7317⟩) (rightExpression := ⟨7218⟩)
    (transferEvent := 115228)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115225.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115159.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115229

namespace SemanticResult115233
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def rawTerms : List Term := Proof.Events450.exact115233RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115233
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115233.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115230) (rightBinding := 115231)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7318⟩) (rightExpression := ⟨7220⟩)
    (transferEvent := 115232)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115229.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115156.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115233

namespace SemanticResult115237
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def rawTerms : List Term := Proof.Events450.exact115237RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115237
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115237.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115234) (rightBinding := 115235)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7319⟩) (rightExpression := ⟨7222⟩)
    (transferEvent := 115236)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115233.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115153.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115237

namespace SemanticResult115241
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def rawTerms : List Term := Proof.Events450.exact115241RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115241
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115241.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115238) (rightBinding := 115239)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7320⟩) (rightExpression := ⟨7224⟩)
    (transferEvent := 115240)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115237.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115150.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115241

namespace SemanticResult115245
def owner : Owner := ⟨.program ⟨257⟩, ⟨7322⟩⟩
def rawTerms : List Term := Proof.Events450.exact115245RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115245
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115245.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115242) (rightBinding := 115243)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7321⟩) (rightExpression := ⟨7226⟩)
    (transferEvent := 115244)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115241.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115147.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115245

namespace SemanticResult115249
def owner : Owner := ⟨.program ⟨257⟩, ⟨7323⟩⟩
def rawTerms : List Term := Proof.Events450.exact115249RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115249
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115249.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115246) (rightBinding := 115247)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7322⟩) (rightExpression := ⟨7228⟩)
    (transferEvent := 115248)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115245.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115144.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115249

namespace SemanticResult115253
def owner : Owner := ⟨.program ⟨257⟩, ⟨7324⟩⟩
def rawTerms : List Term := Proof.Events450.exact115253RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 115253
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult115253.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 115250) (rightBinding := 115251)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7323⟩) (rightExpression := ⟨7230⟩)
    (transferEvent := 115252)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult115249.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115141.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult115253

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
