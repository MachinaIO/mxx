import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard219
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard218

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult27267
def owner : Owner := ⟨.program ⟨257⟩, ⟨15895⟩⟩
def rawTerms : List Term := Proof.Events106.exact27267RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27267
def producerEvent : Nat := 27266
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27267.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 26833, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult27267

namespace SemanticResult27271
def owner : Owner := ⟨.program ⟨257⟩, ⟨18701⟩⟩
def rawTerms : List Term := Proof.Events106.exact27271RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27271
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27271.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27268) (rightBinding := 27269)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15895⟩) (rightExpression := ⟨18700⟩)
    (transferEvent := 27270)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27267.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27244.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27271

namespace SemanticResult27275
def owner : Owner := ⟨.program ⟨257⟩, ⟨21921⟩⟩
def rawTerms : List Term := Proof.Events106.exact27275RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27275
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27275.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27272) (rightBinding := 27273)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18701⟩) (rightExpression := ⟨21920⟩)
    (transferEvent := 27274)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27271.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27221.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27275

namespace SemanticResult27279
def owner : Owner := ⟨.program ⟨257⟩, ⟨31941⟩⟩
def rawTerms : List Term := Proof.Events106.exact27279RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27279
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27279.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27276) (rightBinding := 27277)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21921⟩) (rightExpression := ⟨31940⟩)
    (transferEvent := 27278)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27275.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27198.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27279

namespace SemanticResult27283
def owner : Owner := ⟨.program ⟨257⟩, ⟨50996⟩⟩
def rawTerms : List Term := Proof.Events106.exact27283RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27283
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27283.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27280) (rightBinding := 27281)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨31941⟩) (rightExpression := ⟨50995⟩)
    (transferEvent := 27282)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27279.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27175.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27283

namespace SemanticResult27287
def owner : Owner := ⟨.program ⟨257⟩, ⟨53976⟩⟩
def rawTerms : List Term := Proof.Events106.exact27287RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27287
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27287.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27284) (rightBinding := 27285)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨50996⟩) (rightExpression := ⟨53975⟩)
    (transferEvent := 27286)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27283.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27152.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27287

namespace SemanticResult27291
def owner : Owner := ⟨.program ⟨257⟩, ⟨56956⟩⟩
def rawTerms : List Term := Proof.Events106.exact27291RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27291
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27291.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27288) (rightBinding := 27289)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53976⟩) (rightExpression := ⟨56955⟩)
    (transferEvent := 27290)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27287.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27129.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27291

namespace SemanticResult27295
def owner : Owner := ⟨.program ⟨257⟩, ⟨59936⟩⟩
def rawTerms : List Term := Proof.Events106.exact27295RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27295
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27295.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27292) (rightBinding := 27293)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56956⟩) (rightExpression := ⟨59935⟩)
    (transferEvent := 27294)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27291.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27106.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27295

namespace SemanticResult27299
def owner : Owner := ⟨.program ⟨257⟩, ⟨62916⟩⟩
def rawTerms : List Term := Proof.Events106.exact27299RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27299
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27299.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27296) (rightBinding := 27297)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59936⟩) (rightExpression := ⟨62915⟩)
    (transferEvent := 27298)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27295.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27083.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27299

namespace SemanticResult27303
def owner : Owner := ⟨.program ⟨257⟩, ⟨65994⟩⟩
def rawTerms : List Term := Proof.Events106.exact27303RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27303
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27303.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27300) (rightBinding := 27301)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62916⟩) (rightExpression := ⟨65993⟩)
    (transferEvent := 27302)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27299.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27060.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27303

namespace SemanticResult27307
def owner : Owner := ⟨.program ⟨257⟩, ⟨65995⟩⟩
def rawTerms : List Term := Proof.Events106.exact27307RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27307
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27307.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27304) (rightBinding := 27305)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65994⟩) (rightExpression := ⟨26505⟩)
    (transferEvent := 27306)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27303.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27037.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27307

namespace SemanticResult27311
def owner : Owner := ⟨.program ⟨257⟩, ⟨65996⟩⟩
def rawTerms : List Term := Proof.Events106.exact27311RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27311
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27311.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27308) (rightBinding := 27309)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65995⟩) (rightExpression := ⟨29185⟩)
    (transferEvent := 27310)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27307.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27014.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27311

namespace SemanticResult27315
def owner : Owner := ⟨.program ⟨257⟩, ⟨65997⟩⟩
def rawTerms : List Term := Proof.Events106.exact27315RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27315
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27315.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27312) (rightBinding := 27313)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65996⟩) (rightExpression := ⟨34849⟩)
    (transferEvent := 27314)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27311.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult26991.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27315

namespace SemanticResult27319
def owner : Owner := ⟨.program ⟨257⟩, ⟨65998⟩⟩
def rawTerms : List Term := Proof.Events106.exact27319RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27319
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27319.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27316) (rightBinding := 27317)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65997⟩) (rightExpression := ⟨37529⟩)
    (transferEvent := 27318)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27315.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult26968.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27319

namespace SemanticResult27323
def owner : Owner := ⟨.program ⟨257⟩, ⟨65999⟩⟩
def rawTerms : List Term := Proof.Events106.exact27323RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27323
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27323.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27320) (rightBinding := 27321)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65998⟩) (rightExpression := ⟨40205⟩)
    (transferEvent := 27322)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27319.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult26945.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27323

namespace SemanticResult27327
def owner : Owner := ⟨.program ⟨257⟩, ⟨66000⟩⟩
def rawTerms : List Term := Proof.Events106.exact27327RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27327
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult27327.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27324) (rightBinding := 27325)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65999⟩) (rightExpression := ⟨42885⟩)
    (transferEvent := 27326)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27323.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult26922.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27327

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
