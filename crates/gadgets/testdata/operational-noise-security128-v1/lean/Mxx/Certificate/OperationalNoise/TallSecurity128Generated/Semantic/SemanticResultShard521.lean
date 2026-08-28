import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard521
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard519
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard520

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult71161
def owner : Owner := ⟨.program ⟨257⟩, ⟨51295⟩⟩
def rawTerms : List Term := Proof.Events277.exact71161RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71161
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71161.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71158) (rightBinding := 71159)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32240⟩) (rightExpression := ⟨51294⟩)
    (transferEvent := 71160)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71157.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult71053.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71161

namespace SemanticResult71165
def owner : Owner := ⟨.program ⟨257⟩, ⟨54275⟩⟩
def rawTerms : List Term := Proof.Events277.exact71165RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71165
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71165.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71162) (rightBinding := 71163)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51295⟩) (rightExpression := ⟨54274⟩)
    (transferEvent := 71164)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71161.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult71030.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71165

namespace SemanticResult71169
def owner : Owner := ⟨.program ⟨257⟩, ⟨57255⟩⟩
def rawTerms : List Term := Proof.Events278.exact71169RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71169
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71169.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71166) (rightBinding := 71167)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54275⟩) (rightExpression := ⟨57254⟩)
    (transferEvent := 71168)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult71007.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71169

namespace SemanticResult71173
def owner : Owner := ⟨.program ⟨257⟩, ⟨60235⟩⟩
def rawTerms : List Term := Proof.Events278.exact71173RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71173
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71173.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71170) (rightBinding := 71171)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57255⟩) (rightExpression := ⟨60234⟩)
    (transferEvent := 71172)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71169.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70984.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71173

namespace SemanticResult71177
def owner : Owner := ⟨.program ⟨257⟩, ⟨63215⟩⟩
def rawTerms : List Term := Proof.Events278.exact71177RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71177
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71177.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71174) (rightBinding := 71175)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60235⟩) (rightExpression := ⟨63214⟩)
    (transferEvent := 71176)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71173.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70961.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71177

namespace SemanticResult71181
def owner : Owner := ⟨.program ⟨257⟩, ⟨67092⟩⟩
def rawTerms : List Term := Proof.Events278.exact71181RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71181
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71181.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71178) (rightBinding := 71179)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63215⟩) (rightExpression := ⟨67091⟩)
    (transferEvent := 71180)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71177.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70938.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71181

namespace SemanticResult71185
def owner : Owner := ⟨.program ⟨257⟩, ⟨67093⟩⟩
def rawTerms : List Term := Proof.Events278.exact71185RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71185
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71185.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71182) (rightBinding := 71183)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67092⟩) (rightExpression := ⟨26710⟩)
    (transferEvent := 71184)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71181.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70915.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71185

namespace SemanticResult71189
def owner : Owner := ⟨.program ⟨257⟩, ⟨67094⟩⟩
def rawTerms : List Term := Proof.Events278.exact71189RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71189
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71189.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71186) (rightBinding := 71187)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67093⟩) (rightExpression := ⟨29390⟩)
    (transferEvent := 71188)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71185.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70892.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71189

namespace SemanticResult71193
def owner : Owner := ⟨.program ⟨257⟩, ⟨67095⟩⟩
def rawTerms : List Term := Proof.Events278.exact71193RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71193
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71193.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71190) (rightBinding := 71191)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67094⟩) (rightExpression := ⟨35054⟩)
    (transferEvent := 71192)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71189.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70869.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71193

namespace SemanticResult71197
def owner : Owner := ⟨.program ⟨257⟩, ⟨67096⟩⟩
def rawTerms : List Term := Proof.Events278.exact71197RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71197
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71197.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71194) (rightBinding := 71195)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67095⟩) (rightExpression := ⟨37734⟩)
    (transferEvent := 71196)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71193.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70846.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71197

namespace SemanticResult71201
def owner : Owner := ⟨.program ⟨257⟩, ⟨67097⟩⟩
def rawTerms : List Term := Proof.Events278.exact71201RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71201
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71201.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71198) (rightBinding := 71199)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67096⟩) (rightExpression := ⟨40410⟩)
    (transferEvent := 71200)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71197.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70823.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71201

namespace SemanticResult71205
def owner : Owner := ⟨.program ⟨257⟩, ⟨67098⟩⟩
def rawTerms : List Term := Proof.Events278.exact71205RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71205
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71205.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71202) (rightBinding := 71203)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67097⟩) (rightExpression := ⟨43090⟩)
    (transferEvent := 71204)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71201.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70800.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71205

namespace SemanticResult71209
def owner : Owner := ⟨.program ⟨257⟩, ⟨67099⟩⟩
def rawTerms : List Term := Proof.Events278.exact71209RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71209
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71209.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71206) (rightBinding := 71207)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67098⟩) (rightExpression := ⟨45774⟩)
    (transferEvent := 71208)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71205.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70777.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71209

namespace SemanticResult71213
def owner : Owner := ⟨.program ⟨257⟩, ⟨67100⟩⟩
def rawTerms : List Term := Proof.Events278.exact71213RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71213
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71213.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71210) (rightBinding := 71211)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67099⟩) (rightExpression := ⟨48454⟩)
    (transferEvent := 71212)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71209.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70754.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71213

namespace SemanticResult71224
def owner : Owner := ⟨.program ⟨257⟩, ⟨68872⟩⟩
def rawTerms : List Term := Proof.Events278.exact71224RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71224
def producerEvent : Nat := 71223
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71224.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 70711, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult71224

namespace SemanticResult71227
def owner : Owner := ⟨.program ⟨257⟩, ⟨71469⟩⟩
def rawTerms : List Term := Proof.Events278.exact71227RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71227
def producerEvent : Nat := 71226
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult71227.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 70711, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult71227

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
