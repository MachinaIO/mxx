import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard619
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard559
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard563
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard567
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard571
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard574
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard578
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard582
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard585
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard589
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard593
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard596
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard600
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard604
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard607
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard611
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard615
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard618

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult84575
def owner : Owner := ⟨.program ⟨257⟩, ⟨20842⟩⟩
def rawTerms : List Term := Proof.Events330.exact84575RawTerms
def summary : Bound := (.finite 64377712650190257467641695830016)
def resultEvent : Nat := 84575
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84575.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84570.owner)
    (rightOwner := SemanticResult84088.owner)
    (leftResult := 84570) (rightResult := 84088)
    (leftActual := SemanticResult84570.actual selector witness)
    (rightActual := SemanticResult84088.actual selector witness)
    (leftRaw := SemanticResult84570.rawTerms)
    (rightRaw := SemanticResult84088.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 32188807212483706889510625476608)
    (rightMaximum := 32188905437706550578131070353408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84571) (rightBinding := 84572)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17932⟩) (rightExpression := ⟨20841⟩)
    (transferEvent := 84573) (summaryTransferEvent := 84574)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84570.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult84088.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84575

namespace SemanticResult84580
def owner : Owner := ⟨.program ⟨257⟩, ⟨24062⟩⟩
def rawTerms : List Term := Proof.Events330.exact84580RawTerms
def summary : Bound := (.finite 96566716313119651734393211060224)
def resultEvent : Nat := 84580
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84580.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84575.owner)
    (rightOwner := SemanticResult83606.owner)
    (leftResult := 84575) (rightResult := 83606)
    (leftActual := SemanticResult84575.actual selector witness)
    (rightActual := SemanticResult83606.actual selector witness)
    (leftRaw := SemanticResult84575.rawTerms)
    (rightRaw := SemanticResult83606.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 64377712650190257467641695830016)
    (rightMaximum := 32189003662929394266751515230208) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84576) (rightBinding := 84577)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20842⟩) (rightExpression := ⟨24061⟩)
    (transferEvent := 84578) (summaryTransferEvent := 84579)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84575.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult83606.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84580

namespace SemanticResult84585
def owner : Owner := ⟨.program ⟨257⟩, ⟨34082⟩⟩
def rawTerms : List Term := Proof.Events330.exact84585RawTerms
def summary : Bound := (.finite 128755916426494733378385616044032)
def resultEvent : Nat := 84585
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84585.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84580.owner)
    (rightOwner := SemanticResult83124.owner)
    (leftResult := 84580) (rightResult := 83124)
    (leftActual := SemanticResult84580.actual selector witness)
    (rightActual := SemanticResult83124.actual selector witness)
    (leftRaw := SemanticResult84580.rawTerms)
    (rightRaw := SemanticResult83124.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 96566716313119651734393211060224)
    (rightMaximum := 32189200113375081643992404983808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84581) (rightBinding := 84582)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨24062⟩) (rightExpression := ⟨34081⟩)
    (transferEvent := 84583) (summaryTransferEvent := 84584)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84580.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult83124.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84585

namespace SemanticResult84590
def owner : Owner := ⟨.program ⟨257⟩, ⟨53142⟩⟩
def rawTerms : List Term := Proof.Events330.exact84590RawTerms
def summary : Bound := (.finite 160945509440761189776859800535040)
def resultEvent : Nat := 84590
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84590.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84585.owner)
    (rightOwner := SemanticResult82642.owner)
    (leftResult := 84585) (rightResult := 82642)
    (leftActual := SemanticResult84585.actual selector witness)
    (rightActual := SemanticResult82642.actual selector witness)
    (leftRaw := SemanticResult84585.rawTerms)
    (rightRaw := SemanticResult82642.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 128755916426494733378385616044032)
    (rightMaximum := 32189593014266456398474184491008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84586) (rightBinding := 84587)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨34082⟩) (rightExpression := ⟨53141⟩)
    (transferEvent := 84588) (summaryTransferEvent := 84589)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84585.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult82642.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84590

namespace SemanticResult84595
def owner : Owner := ⟨.program ⟨257⟩, ⟨56122⟩⟩
def rawTerms : List Term := Proof.Events330.exact84595RawTerms
def summary : Bound := (.finite 193135298905473333552574874779648)
def resultEvent : Nat := 84595
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84595.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84590.owner)
    (rightOwner := SemanticResult82160.owner)
    (leftResult := 84590) (rightResult := 82160)
    (leftActual := SemanticResult84590.actual selector witness)
    (rightActual := SemanticResult82160.actual selector witness)
    (leftRaw := SemanticResult84590.rawTerms)
    (rightRaw := SemanticResult82160.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 160945509440761189776859800535040)
    (rightMaximum := 32189789464712143775715074244608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84591) (rightBinding := 84592)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53142⟩) (rightExpression := ⟨56121⟩)
    (transferEvent := 84593) (summaryTransferEvent := 84594)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84590.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult82160.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84595

namespace SemanticResult84600
def owner : Owner := ⟨.program ⟨257⟩, ⟨59102⟩⟩
def rawTerms : List Term := Proof.Events330.exact84600RawTerms
def summary : Bound := (.finite 225325481271076852082771728531456)
def resultEvent : Nat := 84600
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84600.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84595.owner)
    (rightOwner := SemanticResult81678.owner)
    (leftResult := 84595) (rightResult := 81678)
    (leftActual := SemanticResult84595.actual selector witness)
    (rightActual := SemanticResult81678.actual selector witness)
    (leftRaw := SemanticResult84595.rawTerms)
    (rightRaw := SemanticResult81678.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 193135298905473333552574874779648)
    (rightMaximum := 32190182365603518530196853751808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84596) (rightBinding := 84597)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56122⟩) (rightExpression := ⟨59101⟩)
    (transferEvent := 84598) (summaryTransferEvent := 84599)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84595.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult81678.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84600

namespace SemanticResult84605
def owner : Owner := ⟨.program ⟨257⟩, ⟨62082⟩⟩
def rawTerms : List Term := Proof.Events330.exact84605RawTerms
def summary : Bound := (.finite 257515860087126057990209472036864)
def resultEvent : Nat := 84605
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84605.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84600.owner)
    (rightOwner := SemanticResult81196.owner)
    (leftResult := 84600) (rightResult := 81196)
    (leftActual := SemanticResult84600.actual selector witness)
    (rightActual := SemanticResult81196.actual selector witness)
    (leftRaw := SemanticResult84600.rawTerms)
    (rightRaw := SemanticResult81196.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 225325481271076852082771728531456)
    (rightMaximum := 32190378816049205907437743505408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84601) (rightBinding := 84602)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59102⟩) (rightExpression := ⟨62081⟩)
    (transferEvent := 84603) (summaryTransferEvent := 84604)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84600.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult81196.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84605

namespace SemanticResult84610
def owner : Owner := ⟨.program ⟨257⟩, ⟨65062⟩⟩
def rawTerms : List Term := Proof.Events330.exact84610RawTerms
def summary : Bound := (.finite 289706631804066638652128995049472)
def resultEvent : Nat := 84610
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84610.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84605.owner)
    (rightOwner := SemanticResult80714.owner)
    (leftResult := 84605) (rightResult := 80714)
    (leftActual := SemanticResult84605.actual selector witness)
    (rightActual := SemanticResult80714.actual selector witness)
    (leftRaw := SemanticResult84605.rawTerms)
    (rightRaw := SemanticResult80714.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 257515860087126057990209472036864)
    (rightMaximum := 32190771716940580661919523012608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84606) (rightBinding := 84607)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62082⟩) (rightExpression := ⟨65061⟩)
    (transferEvent := 84608) (summaryTransferEvent := 84609)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84605.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult80714.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84610

namespace SemanticResult84615
def owner : Owner := ⟨.program ⟨257⟩, ⟨70655⟩⟩
def rawTerms : List Term := Proof.Events330.exact84615RawTerms
def summary : Bound := (.finite 321897992872344281445771187322880)
def resultEvent : Nat := 84615
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84615.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84610.owner)
    (rightOwner := SemanticResult80232.owner)
    (leftResult := 84610) (rightResult := 80232)
    (leftActual := SemanticResult84610.actual selector witness)
    (rightActual := SemanticResult80232.actual selector witness)
    (leftRaw := SemanticResult84610.rawTerms)
    (rightRaw := SemanticResult80232.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 289706631804066638652128995049472)
    (rightMaximum := 32191361068277642793642192273408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84611) (rightBinding := 84612)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65062⟩) (rightExpression := ⟨70654⟩)
    (transferEvent := 84613) (summaryTransferEvent := 84614)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84610.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult80232.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84615

namespace SemanticResult84620
def owner : Owner := ⟨.program ⟨257⟩, ⟨70656⟩⟩
def rawTerms : List Term := Proof.Events330.exact84620RawTerms
def summary : Bound := (.finite 354089550391067611616654269349888)
def resultEvent : Nat := 84620
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84620.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84615.owner)
    (rightOwner := SemanticResult79750.owner)
    (leftResult := 84615) (rightResult := 79750)
    (leftActual := SemanticResult84615.actual selector witness)
    (rightActual := SemanticResult79750.actual selector witness)
    (leftRaw := SemanticResult84615.rawTerms)
    (rightRaw := SemanticResult79750.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 321897992872344281445771187322880)
    (rightMaximum := 32191557518723330170883082027008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84616) (rightBinding := 84617)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70655⟩) (rightExpression := ⟨28442⟩)
    (transferEvent := 84618) (summaryTransferEvent := 84619)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84615.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79750.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84620

namespace SemanticResult84625
def owner : Owner := ⟨.program ⟨257⟩, ⟨70657⟩⟩
def rawTerms : List Term := Proof.Events330.exact84625RawTerms
def summary : Bound := (.finite 386281697261128003919260020637696)
def resultEvent : Nat := 84625
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84625.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84620.owner)
    (rightOwner := SemanticResult79268.owner)
    (leftResult := 84620) (rightResult := 79268)
    (leftActual := SemanticResult84620.actual selector witness)
    (rightActual := SemanticResult79268.actual selector witness)
    (leftRaw := SemanticResult84620.rawTerms)
    (rightRaw := SemanticResult79268.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 354089550391067611616654269349888)
    (rightMaximum := 32192146870060392302605751287808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84621) (rightBinding := 84622)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70656⟩) (rightExpression := ⟨31122⟩)
    (transferEvent := 84623) (summaryTransferEvent := 84624)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84620.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79268.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84625

namespace SemanticResult84630
def owner : Owner := ⟨.program ⟨257⟩, ⟨70658⟩⟩
def rawTerms : List Term := Proof.Events330.exact84630RawTerms
def summary : Bound := (.finite 418474237032079770976347551432704)
def resultEvent : Nat := 84630
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84630.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84625.owner)
    (rightOwner := SemanticResult78786.owner)
    (leftResult := 84625) (rightResult := 78786)
    (leftActual := SemanticResult84625.actual selector witness)
    (rightActual := SemanticResult78786.actual selector witness)
    (leftRaw := SemanticResult84625.rawTerms)
    (rightRaw := SemanticResult78786.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 386281697261128003919260020637696)
    (rightMaximum := 32192539770951767057087530795008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84626) (rightBinding := 84627)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70657⟩) (rightExpression := ⟨36782⟩)
    (transferEvent := 84628) (summaryTransferEvent := 84629)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84625.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult78786.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84630

namespace SemanticResult84635
def owner : Owner := ⟨.program ⟨257⟩, ⟨70659⟩⟩
def rawTerms : List Term := Proof.Events330.exact84635RawTerms
def summary : Bound := (.finite 450666973253477225410675971981312)
def resultEvent : Nat := 84635
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84635.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84630.owner)
    (rightOwner := SemanticResult78304.owner)
    (leftResult := 84630) (rightResult := 78304)
    (leftActual := SemanticResult84630.actual selector witness)
    (rightActual := SemanticResult78304.actual selector witness)
    (leftRaw := SemanticResult84630.rawTerms)
    (rightRaw := SemanticResult78304.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 418474237032079770976347551432704)
    (rightMaximum := 32192736221397454434328420548608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84631) (rightBinding := 84632)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70658⟩) (rightExpression := ⟨39462⟩)
    (transferEvent := 84633) (summaryTransferEvent := 84634)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84630.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult78304.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84635

namespace SemanticResult84640
def owner : Owner := ⟨.program ⟨257⟩, ⟨70660⟩⟩
def rawTerms : List Term := Proof.Events330.exact84640RawTerms
def summary : Bound := (.finite 482860102375766054599486172037120)
def resultEvent : Nat := 84640
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84640.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84635.owner)
    (rightOwner := SemanticResult77822.owner)
    (leftResult := 84635) (rightResult := 77822)
    (leftActual := SemanticResult84635.actual selector witness)
    (rightActual := SemanticResult77822.actual selector witness)
    (leftRaw := SemanticResult84635.rawTerms)
    (rightRaw := SemanticResult77822.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 450666973253477225410675971981312)
    (rightMaximum := 32193129122288829188810200055808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84636) (rightBinding := 84637)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70659⟩) (rightExpression := ⟨42142⟩)
    (transferEvent := 84638) (summaryTransferEvent := 84639)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84635.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult77822.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84640

namespace SemanticResult84645
def owner : Owner := ⟨.program ⟨257⟩, ⟨70661⟩⟩
def rawTerms : List Term := Proof.Events330.exact84645RawTerms
def summary : Bound := (.finite 515053820849391945920019041353728)
def resultEvent : Nat := 84645
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84645.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84640.owner)
    (rightOwner := SemanticResult77340.owner)
    (leftResult := 84640) (rightResult := 77340)
    (leftActual := SemanticResult84640.actual selector witness)
    (rightActual := SemanticResult77340.actual selector witness)
    (leftRaw := SemanticResult84640.rawTerms)
    (rightRaw := SemanticResult77340.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 482860102375766054599486172037120)
    (rightMaximum := 32193718473625891320532869316608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84641) (rightBinding := 84642)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70660⟩) (rightExpression := ⟨44822⟩)
    (transferEvent := 84643) (summaryTransferEvent := 84644)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84640.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult77340.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84645

namespace SemanticResult84650
def owner : Owner := ⟨.program ⟨257⟩, ⟨70662⟩⟩
def rawTerms : List Term := Proof.Events330.exact84650RawTerms
def summary : Bound := (.finite 547248128674354899372274579931136)
def resultEvent : Nat := 84650
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult84650.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult84645.owner)
    (rightOwner := SemanticResult76858.owner)
    (leftResult := 84645) (rightResult := 76858)
    (leftActual := SemanticResult84645.actual selector witness)
    (rightActual := SemanticResult76858.actual selector witness)
    (leftRaw := SemanticResult84645.rawTerms)
    (rightRaw := SemanticResult76858.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 515053820849391945920019041353728)
    (rightMaximum := 32194307824962953452255538577408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 84646) (rightBinding := 84647)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70661⟩) (rightExpression := ⟨47502⟩)
    (transferEvent := 84648) (summaryTransferEvent := 84649)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult84645.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult76858.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult84650

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
