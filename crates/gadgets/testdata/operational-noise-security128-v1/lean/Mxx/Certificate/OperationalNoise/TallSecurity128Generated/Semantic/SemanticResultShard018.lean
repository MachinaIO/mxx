import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard018
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard005
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard016
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard017

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult2219
def owner : Owner := ⟨.program ⟨257⟩, ⟨16158⟩⟩
def rawTerms : List Term := Proof.Events008.exact2219RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2219
def producerEvent : Nat := 2218
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2219.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 0, .finite 2, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult2219

namespace SemanticResult2224
def owner : Owner := ⟨.program ⟨257⟩, ⟨16159⟩⟩
def rawTerms : List Term := Proof.Events008.exact2224RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2224
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2224.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge2223.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge2223.frameStart)
    (transferEvent := 2222) (owner := owner)
    (leftResult := 2219) (rightResult := 713)
    (working := LeftOperatorMerge2223.working)
    (reconstruction := LeftOperatorMerge2223.reconstruction)
    (leftReference := .predecessor 0 2220 .coefficient) (rightReference := .predecessor 1 2221 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult2219.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult713.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge2223.operationAgreement
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
end SemanticResult2224

namespace SemanticResult2228
def owner : Owner := ⟨.program ⟨257⟩, ⟨16160⟩⟩
def rawTerms : List Term := Proof.Events008.exact2228RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2228
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2228.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2225) (rightBinding := 2226)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6728⟩) (rightExpression := ⟨16159⟩)
    (transferEvent := 2227)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2224.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2228

namespace SemanticResult2232
def owner : Owner := ⟨.program ⟨257⟩, ⟨19015⟩⟩
def rawTerms : List Term := Proof.Events008.exact2232RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2232
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2232.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2229) (rightBinding := 2230)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16160⟩) (rightExpression := ⟨19014⟩)
    (transferEvent := 2231)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2228.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2216.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2232

namespace SemanticResult2236
def owner : Owner := ⟨.program ⟨257⟩, ⟨22235⟩⟩
def rawTerms : List Term := Proof.Events008.exact2236RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2236
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2236.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2233) (rightBinding := 2234)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨19015⟩) (rightExpression := ⟨22234⟩)
    (transferEvent := 2235)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2232.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2208.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2236

namespace SemanticResult2240
def owner : Owner := ⟨.program ⟨257⟩, ⟨32255⟩⟩
def rawTerms : List Term := Proof.Events008.exact2240RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2240.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2237) (rightBinding := 2238)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22235⟩) (rightExpression := ⟨32254⟩)
    (transferEvent := 2239)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2236.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2200.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2240

namespace SemanticResult2244
def owner : Owner := ⟨.program ⟨257⟩, ⟨51319⟩⟩
def rawTerms : List Term := Proof.Events008.exact2244RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2244
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2244.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2241) (rightBinding := 2242)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32255⟩) (rightExpression := ⟨51318⟩)
    (transferEvent := 2243)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2240.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2192.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2244

namespace SemanticResult2248
def owner : Owner := ⟨.program ⟨257⟩, ⟨54299⟩⟩
def rawTerms : List Term := Proof.Events008.exact2248RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2248
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2248.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2245) (rightBinding := 2246)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51319⟩) (rightExpression := ⟨54298⟩)
    (transferEvent := 2247)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2244.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2184.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2248

namespace SemanticResult2252
def owner : Owner := ⟨.program ⟨257⟩, ⟨57279⟩⟩
def rawTerms : List Term := Proof.Events008.exact2252RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2252
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2252.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2249) (rightBinding := 2250)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54299⟩) (rightExpression := ⟨57278⟩)
    (transferEvent := 2251)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2248.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2176.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2252

namespace SemanticResult2256
def owner : Owner := ⟨.program ⟨257⟩, ⟨60259⟩⟩
def rawTerms : List Term := Proof.Events008.exact2256RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2256
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2256.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2253) (rightBinding := 2254)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57279⟩) (rightExpression := ⟨60258⟩)
    (transferEvent := 2255)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2252.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2168.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2256

namespace SemanticResult2260
def owner : Owner := ⟨.program ⟨257⟩, ⟨63239⟩⟩
def rawTerms : List Term := Proof.Events008.exact2260RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2260
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2260.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2257) (rightBinding := 2258)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60259⟩) (rightExpression := ⟨63238⟩)
    (transferEvent := 2259)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2256.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2160.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2260

namespace SemanticResult2264
def owner : Owner := ⟨.program ⟨257⟩, ⟨67150⟩⟩
def rawTerms : List Term := Proof.Events008.exact2264RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2264
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2264.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2261) (rightBinding := 2262)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63239⟩) (rightExpression := ⟨67149⟩)
    (transferEvent := 2263)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2260.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2152.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2264

namespace SemanticResult2268
def owner : Owner := ⟨.program ⟨257⟩, ⟨67151⟩⟩
def rawTerms : List Term := Proof.Events008.exact2268RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2268
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2268.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2265) (rightBinding := 2266)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67150⟩) (rightExpression := ⟨26727⟩)
    (transferEvent := 2267)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2264.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2144.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2268

namespace SemanticResult2272
def owner : Owner := ⟨.program ⟨257⟩, ⟨67152⟩⟩
def rawTerms : List Term := Proof.Events008.exact2272RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2272
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2272.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2269) (rightBinding := 2270)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67151⟩) (rightExpression := ⟨29407⟩)
    (transferEvent := 2271)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2268.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2136.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2272

namespace SemanticResult2276
def owner : Owner := ⟨.program ⟨257⟩, ⟨67153⟩⟩
def rawTerms : List Term := Proof.Events008.exact2276RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2276
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2276.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2273) (rightBinding := 2274)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67152⟩) (rightExpression := ⟨35064⟩)
    (transferEvent := 2275)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2272.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2128.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2276

namespace SemanticResult2280
def owner : Owner := ⟨.program ⟨257⟩, ⟨67154⟩⟩
def rawTerms : List Term := Proof.Events008.exact2280RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2280
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult2280.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2277) (rightBinding := 2278)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67153⟩) (rightExpression := ⟨37744⟩)
    (transferEvent := 2279)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2276.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2120.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2280

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
