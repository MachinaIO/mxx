import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard722
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard720
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard721

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult100403
def owner : Owner := ⟨.program ⟨257⟩, ⟨22182⟩⟩
def rawTerms : List Term := Proof.Events392.exact100403RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100403
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100403.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100400) (rightBinding := 100401)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18962⟩) (rightExpression := ⟨22181⟩)
    (transferEvent := 100402)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100399.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100349.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100403

namespace SemanticResult100407
def owner : Owner := ⟨.program ⟨257⟩, ⟨32202⟩⟩
def rawTerms : List Term := Proof.Events392.exact100407RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100407
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100407.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100404) (rightBinding := 100405)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22182⟩) (rightExpression := ⟨32201⟩)
    (transferEvent := 100406)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100403.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100326.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100407

namespace SemanticResult100411
def owner : Owner := ⟨.program ⟨257⟩, ⟨51257⟩⟩
def rawTerms : List Term := Proof.Events392.exact100411RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100411
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100411.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100408) (rightBinding := 100409)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32202⟩) (rightExpression := ⟨51256⟩)
    (transferEvent := 100410)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100407.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100303.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100411

namespace SemanticResult100415
def owner : Owner := ⟨.program ⟨257⟩, ⟨54237⟩⟩
def rawTerms : List Term := Proof.Events392.exact100415RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100415
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100415.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100412) (rightBinding := 100413)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51257⟩) (rightExpression := ⟨54236⟩)
    (transferEvent := 100414)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100411.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100280.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100415

namespace SemanticResult100419
def owner : Owner := ⟨.program ⟨257⟩, ⟨57217⟩⟩
def rawTerms : List Term := Proof.Events392.exact100419RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100419
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100419.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100416) (rightBinding := 100417)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54237⟩) (rightExpression := ⟨57216⟩)
    (transferEvent := 100418)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100415.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100257.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100419

namespace SemanticResult100423
def owner : Owner := ⟨.program ⟨257⟩, ⟨60197⟩⟩
def rawTerms : List Term := Proof.Events392.exact100423RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100423
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100423.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100420) (rightBinding := 100421)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57217⟩) (rightExpression := ⟨60196⟩)
    (transferEvent := 100422)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100419.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100234.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100423

namespace SemanticResult100427
def owner : Owner := ⟨.program ⟨257⟩, ⟨63177⟩⟩
def rawTerms : List Term := Proof.Events392.exact100427RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100427
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100427.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100424) (rightBinding := 100425)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60197⟩) (rightExpression := ⟨63176⟩)
    (transferEvent := 100426)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100423.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100211.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100427

namespace SemanticResult100431
def owner : Owner := ⟨.program ⟨257⟩, ⟨66952⟩⟩
def rawTerms : List Term := Proof.Events392.exact100431RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100431
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100431.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100428) (rightBinding := 100429)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63177⟩) (rightExpression := ⟨66951⟩)
    (transferEvent := 100430)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100427.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100188.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100431

namespace SemanticResult100435
def owner : Owner := ⟨.program ⟨257⟩, ⟨66953⟩⟩
def rawTerms : List Term := Proof.Events392.exact100435RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100435
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100435.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100432) (rightBinding := 100433)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66952⟩) (rightExpression := ⟨26684⟩)
    (transferEvent := 100434)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100431.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100165.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100435

namespace SemanticResult100439
def owner : Owner := ⟨.program ⟨257⟩, ⟨66954⟩⟩
def rawTerms : List Term := Proof.Events392.exact100439RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100439
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100439.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100436) (rightBinding := 100437)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66953⟩) (rightExpression := ⟨29364⟩)
    (transferEvent := 100438)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100435.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100142.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100439

namespace SemanticResult100443
def owner : Owner := ⟨.program ⟨257⟩, ⟨66955⟩⟩
def rawTerms : List Term := Proof.Events392.exact100443RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100443
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100443.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100440) (rightBinding := 100441)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66954⟩) (rightExpression := ⟨35028⟩)
    (transferEvent := 100442)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100439.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100119.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100443

namespace SemanticResult100447
def owner : Owner := ⟨.program ⟨257⟩, ⟨66956⟩⟩
def rawTerms : List Term := Proof.Events392.exact100447RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100447
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100447.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100444) (rightBinding := 100445)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66955⟩) (rightExpression := ⟨37708⟩)
    (transferEvent := 100446)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100443.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100096.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100447

namespace SemanticResult100451
def owner : Owner := ⟨.program ⟨257⟩, ⟨66957⟩⟩
def rawTerms : List Term := Proof.Events392.exact100451RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100451
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100451.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100448) (rightBinding := 100449)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66956⟩) (rightExpression := ⟨40384⟩)
    (transferEvent := 100450)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100447.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100073.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100451

namespace SemanticResult100455
def owner : Owner := ⟨.program ⟨257⟩, ⟨66958⟩⟩
def rawTerms : List Term := Proof.Events392.exact100455RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100455
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100455.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100452) (rightBinding := 100453)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66957⟩) (rightExpression := ⟨43064⟩)
    (transferEvent := 100454)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100451.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100050.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100455

namespace SemanticResult100459
def owner : Owner := ⟨.program ⟨257⟩, ⟨66959⟩⟩
def rawTerms : List Term := Proof.Events392.exact100459RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100459
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100459.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100456) (rightBinding := 100457)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66958⟩) (rightExpression := ⟨45748⟩)
    (transferEvent := 100458)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100455.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100027.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100459

namespace SemanticResult100463
def owner : Owner := ⟨.program ⟨257⟩, ⟨66960⟩⟩
def rawTerms : List Term := Proof.Events392.exact100463RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100463
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult100463.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100460) (rightBinding := 100461)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66959⟩) (rightExpression := ⟨48428⟩)
    (transferEvent := 100462)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100459.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100004.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100463

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
